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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Sequence

# Chapter-wide per-seed success cutoff on the relative L2 solution error
# (thesis section 4.2.3): a run succeeds iff sol_rel_l2 < 1e-2.
SUCCESS_REL_L2 = 1e-2

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
    wavenumber: float
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
    wavenumber: float = 1.0,
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
        wavenumber=float(wavenumber),
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


def with_rell2_failures(
    sweep: BVPSweepResult,
    *,
    success_rel_l2: float = SUCCESS_REL_L2,
) -> BVPSweepResult:
    """Return a copy of ``sweep`` with ``n_failed`` recomputed per seed.

    A run succeeds iff its relative L2 solution error ``sol_rel_l2`` is
    below ``success_rel_l2``; a cell is flagged failed (1) when no strict
    majority of its seeds succeeds. This is the chapter-wide criterion of
    thesis section 4.2.3, replacing the legacy mean-log10(linf) flag.
    """
    def cell_flag(pts: Sequence[BVPCellResult]) -> int:
        n_succ = sum(
            1 for c in pts
            if math.isfinite(c.sol_rel_l2) and c.sol_rel_l2 < success_rel_l2
        )
        return int(2 * n_succ <= len(pts))

    n_failed = []
    for L in sweep.layers:
        row = []
        for W in sweep.neurons:
            pts = [c for c in sweep.cells if c.layers == L and c.neurons == W]
            if not pts:
                raise ValueError(f"no cells for L={L}, W={W}")
            row.append(cell_flag(pts))
        n_failed.append(tuple(row))
    return replace(sweep, n_failed=tuple(n_failed))


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
            # pre-wavenumber JSONs were all k=1 runs (Adam cannot train k=4).
            wavenumber=float(d.get("wavenumber", 1.0)),
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
    r"""Cross-activation summary for the 1D BVP activation sweep, as two tables.

    The combined seven-column table is too wide for the text block, so it is
    split into two narrower tables printed one after the other. The first
    (\label{tab:bvp-activation-summary}) carries the best architecture, the
    solution $L^\infty$ and relative $L^2$ errors, and the failure count; the
    second (\label{tab:bvp-activation-summary-diag}) carries the PDE residual
    norm and the mean training time per cell, for the same best cells.
    """
    # Drop the redundant unit coefficient so k=1 reads "\pi", not "1\pi".
    k = sweeps[0].wavenumber
    k_str = "" if k == 1.0 else f"{k:g}"

    # One pass: best cell and aggregates per activation, reused by both tables.
    rows = []
    for sw in sweeps:
        i, j, L, W = _best_cell(sw)
        total = len(sw.layers) * len(sw.neurons)
        failed = sum(sum(row) for row in sw.n_failed)
        time_all = [t for row in sw.time_mean for t in row if math.isfinite(t)]
        tmean = sum(time_all) / len(time_all) if time_all else float("nan")
        rows.append({
            "name": _escape_latex(sw.activation),
            "LW": f"({L}, {W})",
            "linf": _fmt_pm(sw.sol_linf_mean[i][j], sw.sol_linf_std[i][j]),
            "rel": _fmt_pm(sw.sol_rel_l2_mean[i][j], sw.sol_rel_l2_std[i][j]),
            "res": _fmt_pm(sw.residual_l2_mean[i][j], sw.residual_l2_std[i][j]),
            "failed": f"{failed}/{total}",
            "time": f"{tmean:.2f}",
        })

    lines = [
        # --- Table 1: best architecture + solution errors + failure count ---
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Activation & Best $(L,W)$ & $\varepsilon_\infty$ & "
        r"$\varepsilon^{\mathrm{rel}}_{L^2}$ & Failed \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['name']} & {r['LW']} & {r['linf']} & {r['rel']} & "
            f"{r['failed']} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{One-dimensional BVP $-u''=({k_str}\pi)^2\sin({k_str}\pi x)$"
        r", solution errors: best architecture per activation over the "
        r"$L\in\{1,\ldots,7\}\times W\in\{5,10,20,40,80\}$ grid, with solution "
        r"$L^\infty$ error $\varepsilon_\infty$ and relative $L^2$ error "
        r"$\varepsilon^{\mathrm{rel}}_{L^2}$ (mean $\pm$ std over the seed "
        r"ensemble) at the best cell, and the number of failed cells (no "
        r"majority of seeds reaches $\varepsilon^{\mathrm{rel}}_{L^2}<10^{-2}$). "
        r"Residual norm and timing for the same runs are in "
        r"Table~\ref{tab:bvp-activation-summary-diag}.}",
        r"\label{tab:bvp-activation-summary}",
        r"\end{table}",
        "",
        # --- Table 2: PDE residual norm + mean training time ---
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Activation & $\|\widehat{u}_\theta''-f\|_{L^2}$ & Time/cell [s] \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(f"{r['name']} & {r['res']} & {r['time']} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Companion to Table~\ref{tab:bvp-activation-summary}: "
        r"PDE residual $L^2$ norm $\|\widehat{u}_\theta''-f\|_{L^2}$ (mean "
        r"$\pm$ std over the seed ensemble) at the best cell, and mean "
        r"training time per cell, for the same activation runs.}",
        r"\label{tab:bvp-activation-summary-diag}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
