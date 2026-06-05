"""Pure aggregation, persistence and LaTeX helpers for the 1D BVP
activation sweep (companion of the section 4.1 interpolation sweep).

Deliberately torch-free so it can be unit-tested without a GPU. The raw
per-seed JSON written here is the reproducible source of truth that the
BVP activation-comparison prose cites.

Unlike the interpolation sweep, which records a single error norm, every
BVP cell carries three metrics evaluated on a dense grid after training:

    * ``sol_linf``    -- solution error  max|u_hat - u_exact|        (epsilon_inf)
    * ``sol_rel_l2``  -- relative solution error in L2 norm          (epsilon^rel_L2)
    * ``residual_l2`` -- strong-form PDE residual  ||u_hat'' - f||_L2

The heatmap colour metric is ``sol_linf`` (the direct analogue of the
interpolation epsilon_inf); the other two are plotted as companion panels
and reported in the cross-activation summary table.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

# Metric registry: maps a metric key to the cell attribute, the aggregated
# mean/std grid attributes on BVPSweepResult, a colour-bar / axis label
# (LaTeX math), a short human name, and the figure-filename token. The sweep,
# the shared-scale regenerator and the summary table all iterate over this so
# a new metric is added in exactly one place.
METRICS: dict[str, dict[str, str]] = {
    "sol_linf": {
        "cell_attr": "sol_linf",
        "mean_attr": "sol_linf_mean",
        "std_attr": "sol_linf_std",
        "label": r"$\log_{10}\,\varepsilon_\infty$ (mean over seeds)",
        "short": r"Solution $L^\infty$ error",
        "token": "sol_linf",
    },
    "sol_rel_l2": {
        "cell_attr": "sol_rel_l2",
        "mean_attr": "sol_rel_l2_mean",
        "std_attr": "sol_rel_l2_std",
        "label": r"$\log_{10}\,\varepsilon^{\mathrm{rel}}_{L^2}$ (mean over seeds)",
        "short": r"Relative $L^2$ error",
        "token": "sol_rel_l2",
    },
    "residual_l2": {
        "cell_attr": "residual_l2",
        "mean_attr": "residual_l2_mean",
        "std_attr": "residual_l2_std",
        "label": r"$\log_{10}\,\|\widehat{u}_\theta''-f\|_{L^2}$ (mean over seeds)",
        "short": r"PDE residual $L^2$",
        "token": "residual_l2",
    },
}

# The metric that defines "best cell" and the failure flag, mirroring the
# interpolation sweep where epsilon_inf plays both roles.
PRIMARY_METRIC = "sol_linf"


@dataclass(frozen=True)
class BVPCellResult:
    """One trained BVP network at a (layers, neurons, seed) point."""
    layers: int
    neurons: int
    seed: int
    sol_linf: float
    sol_rel_l2: float
    residual_l2: float
    train_time_s: float
    epochs_run: int


@dataclass(frozen=True)
class BVPSweepResult:
    """Aggregated sweep for a single activation.

    Every ``*_mean`` / ``*_std`` / ``time_mean`` / ``n_failed`` field is a
    row-major grid indexed ``[i_layers][j_neurons]``.
    """
    activation: str
    layers: tuple[int, ...]
    neurons: tuple[int, ...]
    seeds: tuple[int, ...]
    failure_log_threshold: float
    machine_eps: float
    sol_linf_mean: tuple[tuple[float, ...], ...]
    sol_linf_std: tuple[tuple[float, ...], ...]
    sol_rel_l2_mean: tuple[tuple[float, ...], ...]
    sol_rel_l2_std: tuple[tuple[float, ...], ...]
    residual_l2_mean: tuple[tuple[float, ...], ...]
    residual_l2_std: tuple[tuple[float, ...], ...]
    time_mean: tuple[tuple[float, ...], ...]
    n_failed: tuple[tuple[int, ...], ...]
    cells: tuple[BVPCellResult, ...]
    created_utc: str


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _pop_std(xs: Sequence[float]) -> float:
    """Population standard deviation (divides by N, not N-1).

    Matches the interpolation sweep so the two studies report the seed
    spread on the same convention.
    """
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _finite_mean(xs: Sequence[float]) -> float:
    """Mean over finite entries only; NaN if none are finite.

    Used for the timing column so a crashed run (recorded with
    ``train_time_s=nan``) does not bias mean training time.
    """
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def aggregate(
    activation: str,
    layers: Sequence[int],
    neurons: Sequence[int],
    cells: Sequence[BVPCellResult],
    *,
    failure_log_threshold: float,
    machine_eps: float,
) -> BVPSweepResult:
    """Build a BVPSweepResult from raw per-seed cells. Never mutates input."""
    layers = list(layers)
    neurons = list(neurons)
    seeds = sorted({c.seed for c in cells})

    def grid(fn: Callable[[list[BVPCellResult]], float | int]):
        out = []
        for L in layers:
            row = []
            for W in neurons:
                pts = [c for c in cells if c.layers == L and c.neurons == W]
                if not pts:
                    raise ValueError(f"no cells for L={L}, W={W}")
                row.append(fn(pts))
            out.append(tuple(row))
        return tuple(out)

    def mean_of(attr: str):
        return grid(lambda p: _mean([getattr(c, attr) for c in p]))

    def std_of(attr: str):
        return grid(lambda p: _pop_std([getattr(c, attr) for c in p]))

    n_failed = grid(
        lambda p: int(
            math.log10(max(_mean([getattr(c, PRIMARY_METRIC) for c in p]),
                           machine_eps))
            > failure_log_threshold
        )
    )

    return BVPSweepResult(
        activation=activation,
        layers=tuple(layers),
        neurons=tuple(neurons),
        seeds=tuple(seeds),
        failure_log_threshold=failure_log_threshold,
        machine_eps=machine_eps,
        sol_linf_mean=mean_of("sol_linf"),
        sol_linf_std=std_of("sol_linf"),
        sol_rel_l2_mean=mean_of("sol_rel_l2"),
        sol_rel_l2_std=std_of("sol_rel_l2"),
        residual_l2_mean=mean_of("residual_l2"),
        residual_l2_std=std_of("residual_l2"),
        time_mean=grid(lambda p: _finite_mean([c.train_time_s for c in p])),
        n_failed=n_failed,
        cells=tuple(cells),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def _tuplify(x):
    """Recursively convert nested lists to nested tuples; other types pass through.

    JSON has no tuple type, so a loaded grid comes back as nested lists;
    frozen-dataclass equality is type-sensitive, so the grids must be
    re-tuplified for ``load_json(...) == original`` to hold.
    """
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    return x


def save_json(sweep: BVPSweepResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(sweep), fh, indent=2)


def load_json(path: str) -> BVPSweepResult:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        cells = tuple(BVPCellResult(**c) for c in d["cells"])
        return BVPSweepResult(
            activation=d["activation"],
            layers=tuple(d["layers"]),
            neurons=tuple(d["neurons"]),
            seeds=tuple(d["seeds"]),
            failure_log_threshold=d["failure_log_threshold"],
            machine_eps=d["machine_eps"],
            sol_linf_mean=_tuplify(d["sol_linf_mean"]),
            sol_linf_std=_tuplify(d["sol_linf_std"]),
            sol_rel_l2_mean=_tuplify(d["sol_rel_l2_mean"]),
            sol_rel_l2_std=_tuplify(d["sol_rel_l2_std"]),
            residual_l2_mean=_tuplify(d["residual_l2_mean"]),
            residual_l2_std=_tuplify(d["residual_l2_std"]),
            time_mean=_tuplify(d["time_mean"]),
            n_failed=_tuplify(d["n_failed"]),
            cells=cells,
            created_utc=d["created_utc"],
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to load BVPSweepResult from {path!r}: {exc}"
        ) from exc


def _escape_latex(s: str) -> str:
    """Escape LaTeX specials for plain-text table cells."""
    s = s.replace("\\", r"\textbackslash{}")
    for char, repl in (
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        s = s.replace(char, repl)
    return s


def _fmt_pm(mean: float, std: float) -> str:
    r"""Format ``mean ± std`` as a math-mode LaTeX cell with a shared exponent.

    Non-finite means (e.g. an all-failed activation) render as
    ``$\mathrm{n/a}$``.
    """
    if not math.isfinite(mean):
        return r"$\mathrm{n/a}$"
    exp = 0 if mean == 0.0 else math.floor(math.log10(abs(mean)))
    scale = 10.0 ** exp
    return rf"$({mean / scale:.2f} \pm {std / scale:.2f})\times 10^{{{exp}}}$"


def _best_cell(sweep: BVPSweepResult) -> tuple[int, int, int, int]:
    """Indices and (L, W) of the lowest mean primary-metric cell."""
    mean_grid = getattr(sweep, METRICS[PRIMARY_METRIC]["mean_attr"])
    best = None
    for i, L in enumerate(sweep.layers):
        for j, W in enumerate(sweep.neurons):
            m = mean_grid[i][j]
            if best is None or m < best[0]:
                best = (m, i, j, L, W)
    if best is None:
        raise ValueError("BVPSweepResult has no grid cells.")
    _, i, j, L, W = best
    return i, j, L, W


def to_latex_summary(sweeps: Sequence[BVPSweepResult]) -> str:
    """Cross-activation summary table for the 1D BVP activation sweep.

    Columns: activation, best (L, W) by solution L-infinity, the three
    metrics (mean +- std at that best cell), failed-cell count, mean time
    per cell.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Activation & Best $(L,W)$ & $\varepsilon_\infty$ & "
        r"$\varepsilon^{\mathrm{rel}}_{L^2}$ & $\|\widehat{u}_\theta''-f\|_{L^2}$ "
        r"& Failed & Time/cell [s] \\",
        r"\hline",
    ]
    for sw in sweeps:
        i, j, L, W = _best_cell(sw)
        total = len(sw.layers) * len(sw.neurons)
        failed = sum(sum(row) for row in sw.n_failed)
        time_all = [t for row in sw.time_mean for t in row if math.isfinite(t)]
        tmean = sum(time_all) / len(time_all) if time_all else float("nan")
        lines.append(
            f"{_escape_latex(sw.activation)} & ({L}, {W}) & "
            f"{_fmt_pm(sw.sol_linf_mean[i][j], sw.sol_linf_std[i][j])} & "
            f"{_fmt_pm(sw.sol_rel_l2_mean[i][j], sw.sol_rel_l2_std[i][j])} & "
            f"{_fmt_pm(sw.residual_l2_mean[i][j], sw.residual_l2_std[i][j])} & "
            f"{failed}/{total} & {tmean:.2f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{One-dimensional BVP $-u''=(4\pi)^2\sin(4\pi x)$: best "
        r"architecture per activation over the $L\in\{1,\ldots,7\}\times "
        r"W\in\{5,10,20,40,80\}$ grid, with solution $L^\infty$ error, "
        r"relative $L^2$ error and PDE residual $L^2$ norm (mean $\pm$ std "
        r"over the seed ensemble), number of cells flagged as failed, and "
        r"mean training time per cell.}",
        r"\label{tab:bvp-activation-summary}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
