"""Pure aggregation, persistence and LaTeX helpers for the 1D
interpolation architecture sweep (thesis section 4.1).

Deliberately torch-free so it can be unit-tested without a GPU.
The raw per-seed JSON written here is the reproducible source of
truth that the section 4.1 prose cites.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Sequence

# Chapter-wide per-seed success cutoff on the relative L2 error (thesis
# section 4.2.3): a run succeeds iff ||u_hat - f||_L2 / ||f||_L2 < 1e-2.
SUCCESS_REL_L2 = 1e-2

# ||f||_L2 on [-1,1] for the sweep target f(x) = (x - 1/2)^2:
# integral of (x-1/2)^4 over [-1,1] is 61/40, so the norm is sqrt(61/40).
TARGET_L2_NORM = math.sqrt(61.0 / 40.0)


@dataclass(frozen=True)
class CellResult:
    """One trained network at a (layers, neurons, seed) point."""
    layers: int
    neurons: int
    seed: int
    linf: float
    l2: float
    train_time_s: float
    epochs_run: int


@dataclass(frozen=True)
class SweepResult:
    """Aggregated sweep for a single activation.

    The `*_mean`, `*_std`, `time_mean`, `n_failed` fields are row-major
    grids indexed `[i_layers][j_neurons]`.
    """
    activation: str
    layers: tuple[int, ...]
    neurons: tuple[int, ...]
    seeds: tuple[int, ...]
    failure_log_threshold: float
    machine_eps: float
    linf_mean: tuple[tuple[float, ...], ...]
    linf_std: tuple[tuple[float, ...], ...]
    l2_mean: tuple[tuple[float, ...], ...]
    l2_std: tuple[tuple[float, ...], ...]
    time_mean: tuple[tuple[float, ...], ...]
    n_failed: tuple[tuple[int, ...], ...]
    cells: tuple[CellResult, ...]
    created_utc: str


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _pop_std(xs: Sequence[float]) -> float:
    """Population standard deviation (divides by N, not N-1).

    The thesis cites these as 'std over the seed ensemble'; the
    population estimator is the deliberate, documented choice.
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
    cells: Sequence[CellResult],
    *,
    failure_log_threshold: float,
    machine_eps: float,
) -> SweepResult:
    """Build a SweepResult from raw per-seed cells. Never mutates input."""
    layers = list(layers)
    neurons = list(neurons)
    seeds = sorted({c.seed for c in cells})

    def grid(fn):
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

    linf_mean = grid(lambda p: _mean([c.linf for c in p]))
    linf_std = grid(lambda p: _pop_std([c.linf for c in p]))
    l2_mean = grid(lambda p: _mean([c.l2 for c in p]))
    l2_std = grid(lambda p: _pop_std([c.l2 for c in p]))
    time_mean = grid(lambda p: _finite_mean([c.train_time_s for c in p]))
    n_failed = grid(
        lambda p: int(math.log10(max(_mean([c.linf for c in p]), machine_eps))
                      > failure_log_threshold)
    )

    return SweepResult(
        activation=activation,
        layers=tuple(layers),
        neurons=tuple(neurons),
        seeds=tuple(seeds),
        failure_log_threshold=failure_log_threshold,
        machine_eps=machine_eps,
        linf_mean=linf_mean,
        linf_std=linf_std,
        l2_mean=l2_mean,
        l2_std=l2_std,
        time_mean=time_mean,
        n_failed=n_failed,
        cells=tuple(cells),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def with_rell2_failures(
    sweep: SweepResult,
    *,
    success_rel_l2: float = SUCCESS_REL_L2,
    target_l2_norm: float = TARGET_L2_NORM,
) -> SweepResult:
    """Return a copy of ``sweep`` with ``n_failed`` recomputed per seed.

    A run succeeds iff its relative L2 error ``l2 / target_l2_norm`` is
    below ``success_rel_l2``; a cell is flagged failed (1) when no strict
    majority of its seeds succeeds. This is the chapter-wide criterion of
    thesis section 4.2.3, replacing the legacy mean-log10(linf) flag.
    """
    def cell_flag(pts: Sequence[CellResult]) -> int:
        n_succ = sum(
            1 for c in pts
            if math.isfinite(c.l2) and c.l2 / target_l2_norm < success_rel_l2
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


def _tuplify(x):
    """Recursively convert nested lists to nested tuples; other types pass through.

    JSON has no tuple type, so a loaded grid comes back as nested lists;
    frozen-dataclass equality is type-sensitive, so the grids must be
    re-tuplified for ``load_json(...) == original`` to hold.
    """
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    return x


def save_json(sweep: SweepResult, path: str) -> None:
    payload = asdict(sweep)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: str) -> SweepResult:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        cells = tuple(CellResult(**c) for c in d["cells"])
        return SweepResult(
            activation=d["activation"],
            layers=tuple(d["layers"]),
            neurons=tuple(d["neurons"]),
            seeds=tuple(d["seeds"]),
            failure_log_threshold=d["failure_log_threshold"],
            machine_eps=d["machine_eps"],
            linf_mean=_tuplify(d["linf_mean"]),
            linf_std=_tuplify(d["linf_std"]),
            l2_mean=_tuplify(d["l2_mean"]),
            l2_std=_tuplify(d["l2_std"]),
            time_mean=_tuplify(d["time_mean"]),
            n_failed=_tuplify(d["n_failed"]),
            cells=cells,
            created_utc=d["created_utc"],
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to load SweepResult from {path!r}: {exc}"
        ) from exc


def _escape_latex(s: str) -> str:
    """Escape LaTeX specials for plain-text table cells.

    Backslash is replaced first so the replacement strings (which
    themselves contain no other specials handled here) are not
    re-escaped.
    """
    s = s.replace("\\", r"\textbackslash{}")
    for char, repl in (
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        s = s.replace(char, repl)
    return s


def _fmt_pm(mean: float, std: float) -> str:
    r"""Format ``mean ± std`` as a math-mode LaTeX cell, TFM-4 style.

    Produces ``$(m.mm \pm s.ss)\times 10^{e}$`` with a shared exponent
    taken from the mean, matching the reference table 4.1. Non-finite
    means (e.g. an all-failed activation) render as ``$\mathrm{n/a}$``.
    Must be math-mode delimited: the table columns are text-mode, so a
    bare ``\pm`` would break ``pdflatex``.
    """
    if not math.isfinite(mean):
        return r"$\mathrm{n/a}$"
    if mean == 0.0:
        exp = 0
    else:
        exp = math.floor(math.log10(abs(mean)))
    scale = 10.0 ** exp
    return rf"$({mean / scale:.2f} \pm {std / scale:.2f})\times 10^{{{exp}}}$"


def _best_cell(sweep: SweepResult) -> tuple[int, int, float, float, float, float]:
    best = None
    for i, L in enumerate(sweep.layers):
        for j, W in enumerate(sweep.neurons):
            m = sweep.linf_mean[i][j]
            if best is None or m < best[2]:
                best = (L, W, m, sweep.linf_std[i][j],
                        sweep.l2_mean[i][j], sweep.l2_std[i][j])
    if best is None:
        raise ValueError(
            "SweepResult has no grid cells; cannot determine best cell."
        )
    return best


def to_latex_summary(sweeps: Sequence[SweepResult]) -> str:
    """Cross-activation summary table, analogous to TFM-4 Table 4.1.

    Columns: activation, best (L, W), L-inf (mean +- std),
    L2 (mean +- std), failed cells / total, mean time per cell [s].
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Activation & Best $(L,W)$ & $\varepsilon_\infty$ "
        r"& $L^2$ & Failed & Time/cell [s] \\",
        r"\hline",
    ]
    for sw in sweeps:
        L, W, lm, ls, l2m, l2s = _best_cell(sw)
        total = len(sw.layers) * len(sw.neurons)
        failed = sum(sum(row) for row in sw.n_failed)
        time_all = [t for row in sw.time_mean for t in row]
        tmean = sum(time_all) / len(time_all)
        lines.append(
            f"{_escape_latex(sw.activation)} & ({L}, {W}) & {_fmt_pm(lm, ls)} & "
            f"{_fmt_pm(l2m, l2s)} & {failed}/{total} & {tmean:.2f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{One-dimensional interpolation benchmark: best "
        r"architecture per activation over the $L\in\{1,\ldots,7\}\times "
        r"W\in\{5,10,20,40,80\}$ grid, with $L^\infty$ and $L^2$ errors "
        r"(mean $\pm$ std over the seed ensemble), number of failed cells "
        r"(no majority of seeds reaches "
        r"$\varepsilon^{\mathrm{rel}}_{L^2}<10^{-2}$), and mean training "
        r"time per cell.}",
        r"\label{tab:interp-summary}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
