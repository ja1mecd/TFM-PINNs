"""Pure aggregation, persistence and LaTeX helpers for the 1D
interpolation architecture sweep (thesis section 4.1).

Deliberately torch-free so it can be unit-tested without a GPU.
The raw per-seed JSON written here is the reproducible source of
truth that the section 4.1 prose cites.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence


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
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


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
    time_mean = grid(lambda p: _mean([c.train_time_s for c in p]))
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
