"""Optimiser comparison on the 2D Poisson benchmark (unit square).

Sibling of ``BVP/one_d/optimiser_comparison_1d.py``: same four pipelines, same
success criterion, same figure/summary machinery (imported, not duplicated),
but trained on the 2D Poisson problem

    u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y),    u = 0 on the boundary,

with the hard box ansatz u_hat = x(1-x)y(1-y)N(x,y) and a 3x32 Tanh network.

    - "adam"             pure Adam for the full budget;
    - "adam_bfgs"        Adam warm start, then standard BFGS;
    - "adam_ssbfgs"      Adam warm start, then self-scaled BFGS;
    - "adam_ssbroyden"   Adam warm start, then self-scaled Broyden.

The quasi-Newton phase runs the Urban engine (float64 dense Hessian + strong
Wolfe). Unlike the legacy 1D run, ``initial_scale`` is ON by default here, so
the Oren-Luenberger initial Hessian scaling that Urban's headline SSBroyden
relies on is applied on the first step AND after every mid-run reset (the
optimiser fix in ``ssbroyden_urban.py``).

This is a heavy, GPU-bound sweep (four pipelines x N seeds x up to 10^4 epochs
with an O(n^2) dense-Hessian step and a 2D Laplacian backward). Run it on the
training machine, not interactively.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

# --- make the shared 1D machinery, the 2D solver, and the optimisers importable
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../BVP/two_d
_BVP = os.path.dirname(_HERE)                                  # .../BVP
_ONE_D = os.path.join(_BVP, "one_d")
_OPT = os.path.join(_BVP, "optimizers")
for _p in (_HERE, _ONE_D, _OPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Problem-agnostic comparison machinery is reused verbatim from the 1D driver.
from optimiser_comparison_1d import (  # noqa: E402
    PIPELINES,
    SUCCESS_REL_L2_DEFAULT,
    PipelineResult,
    SeedRun,
    plot_comparison,
    write_summary,
)
from pinn_poisson_2d_unitsquare import Net, PoissonPINN  # noqa: E402

PROBLEM_LABEL = "2D Poisson, unit square"


# =============================================================================
# Single seed run for a pipeline
# =============================================================================
def run_pipeline_once(
    pipeline: str,
    seed: int,
    total_epochs: int,
    adam_warmup: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    loss_transform: str,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    early_stop: bool,
    es_patience: int,
    es_window: int,
    es_min_delta: float,
    es_stop_loss: float,
    qn_engine: str,
    qn_initial_scale: bool,
    qn_wolfe_c1: float,
    qn_wolfe_c2: float,
    qn_wolfe_max_ls: int,
) -> SeedRun:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; valid: {PIPELINES}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Net(hidden=tuple(hidden))

    qn_variant = {
        "adam": "ssbroyden",      # never reached (QN phase is empty)
        "adam_bfgs": "bfgs",
        "adam_ssbfgs": "ssbfgs",
        "adam_ssbroyden": "ssbroyden",
    }[pipeline]

    pinn = PoissonPINN(
        model=model,
        lr=lr,
        loss_transform=loss_transform,
        qn_variant=qn_variant,
        qn_engine=qn_engine,
        qn_wolfe_c1=qn_wolfe_c1,
        qn_wolfe_c2=qn_wolfe_c2,
        qn_wolfe_max_ls=qn_wolfe_max_ls,
        qn_initial_scale=qn_initial_scale,
    )

    if pipeline == "adam":
        # Pure Adam: adam_epochs == n_epochs so the QN phase never engages,
        # and the post-handover early stop can never fire. Run the full budget.
        pinn.train(
            n_epochs=total_epochs,
            adam_epochs=total_epochs,
            n_collocation=n_collocation,
            train_split=0.8,
            resample_every=500,
            verbose_freq=max(1, total_epochs // 5),
            diag_grid_n=200,
            handover_strategy="fixed",
            handover_max_adam_epochs=handover_max_adam_epochs,
            early_stop=False,
        )
    else:
        pinn.train(
            n_epochs=total_epochs,
            adam_epochs=adam_warmup,
            n_collocation=n_collocation,
            train_split=0.8,
            resample_every=500,
            verbose_freq=max(1, total_epochs // 5),
            diag_grid_n=200,
            handover_strategy=handover_strategy,
            handover_max_adam_epochs=handover_max_adam_epochs,
            plateau_patience=plateau_patience,
            plateau_min_delta=plateau_min_delta,
            early_stop=early_stop,
            es_patience=es_patience,
            es_window=es_window,
            es_min_delta=es_min_delta,
            es_stop_loss=es_stop_loss,
        )

    return SeedRun(
        seed=seed,
        J_val_history=np.asarray(pinn.J_val, dtype=np.float64),
        sol_l2_history=np.asarray(pinn.sol_l2, dtype=np.float64),
        final_J_val=float(pinn.J_val[-1]),
        final_pde_l2=float("nan"),  # PoissonPINN does not log a separate pde_l2
        final_sol_l2=float(pinn.sol_l2[-1]),
        final_sol_rel_l2=float(pinn.sol_rel_l2[-1]),
    )


def run_comparison(
    pipelines: tuple[str, ...],
    seeds: tuple[int, ...],
    **kwargs,
) -> tuple[PipelineResult, ...]:
    results: list[PipelineResult] = []
    for p in pipelines:
        runs: list[SeedRun] = []
        for s in seeds:
            print(f"\n[pipeline={p}, seed={s}]")
            runs.append(run_pipeline_once(pipeline=p, seed=s, **kwargs))
        results.append(PipelineResult(pipeline=p, seeds=tuple(runs)))
    return tuple(results)


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adam / Adam->BFGS / Adam->SSBFGS / Adam->SSBroyden "
        "comparison on the 2D Poisson benchmark."
    )
    p.add_argument(
        "--pipelines", type=str, nargs="+",
        default=list(PIPELINES), choices=list(PIPELINES),
    )
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                            52, 53, 54, 55, 56, 57, 58, 59, 60, 61])
    p.add_argument("--total-epochs", type=int, default=10000,
                   help="Total budget; matches the 2D Poisson protocol.")
    p.add_argument("--adam-warmup", type=int, default=2000,
                   help="Fixed Adam warm-up length before QN handover.")
    p.add_argument("--n-collocation", type=int, default=2000)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32],
                   help="Hidden-layer widths; default 3x32 matches the thesis.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss-transform", type=str, default="identity",
                   choices=["identity", "sqrt", "log", "boxcox"])
    p.add_argument("--handover-strategy", type=str, default="fixed",
                   choices=["fixed", "plateau", "loss_threshold", "gradnorm"])
    p.add_argument("--handover-max-adam-epochs", type=int, default=10000)
    p.add_argument("--plateau-patience", type=int, default=200)
    p.add_argument("--plateau-min-delta", type=float, default=1e-4)
    # QN-phase early stopping (urban-style relative-MA criterion).
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="Disable QN-phase early stopping (run the full budget).")
    p.set_defaults(early_stop=True)
    p.add_argument("--es-patience", type=int, default=300)
    p.add_argument("--es-window", type=int, default=20)
    p.add_argument("--es-min-delta", type=float, default=1e-4)
    p.add_argument("--es-stop-loss", type=float, default=0.0)
    # Urban QN engine knobs.
    p.add_argument("--engine", choices=["inhouse", "urban"], default="urban",
                   help="QN engine. Default 'urban' (float64 + strong-Wolfe).")
    p.add_argument("--no-initial-scale", dest="initial_scale",
                   action="store_false",
                   help="Disable Urban's Oren-Luenberger initial Hessian "
                        "scaling (urban engine only). On by default.")
    p.set_defaults(initial_scale=True)
    p.add_argument("--wolfe-c1", type=float, default=1e-4)
    p.add_argument("--wolfe-c2", type=float, default=0.9)
    p.add_argument("--wolfe-max-ls", type=int, default=25)
    p.add_argument("--results-dir", type=str,
                   default=os.path.join("..", "results"))
    p.add_argument("--success-rel-l2-threshold", type=float,
                   default=SUCCESS_REL_L2_DEFAULT,
                   help="Final relative L^2 error below which a seed counts as "
                        "successful. Default 1.0 (trivial zero predictor); pass "
                        "0.01 for the chapter-wide accuracy criterion.")
    p.add_argument("--portrait", action="store_true",
                   help="Use a portrait (10x12) figure geometry for full-page "
                        "thesis rendering instead of the default 14x10.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    engine_tag = "_urban" if args.engine == "urban" else ""
    out_dir = os.path.join(
        args.results_dir,
        f"poisson2d_optim_compare{engine_tag}_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\nOptimiser comparison on the 2D Poisson benchmark "
        f"(total_epochs={args.total_epochs}, adam_warmup={args.adam_warmup}).\n"
        f"  pipelines:        {args.pipelines}\n"
        f"  seeds:            {args.seeds}\n"
        f"  engine:           {args.engine}"
        f"  (initial_scale={args.initial_scale})\n"
        f"  handover:         {args.handover_strategy}\n"
        f"  success rel L^2:  < {args.success_rel_l2_threshold:g}\n"
    )

    results = run_comparison(
        pipelines=tuple(args.pipelines),
        seeds=tuple(args.seeds),
        total_epochs=args.total_epochs,
        adam_warmup=args.adam_warmup,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden),
        lr=args.lr,
        loss_transform=args.loss_transform,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        early_stop=args.early_stop,
        es_patience=args.es_patience,
        es_window=args.es_window,
        es_min_delta=args.es_min_delta,
        es_stop_loss=args.es_stop_loss,
        qn_engine=args.engine,
        qn_initial_scale=args.initial_scale,
        qn_wolfe_c1=args.wolfe_c1,
        qn_wolfe_c2=args.wolfe_c2,
        qn_wolfe_max_ls=args.wolfe_max_ls,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        k=0.0,  # unused; problem_label overrides the header
        total_epochs=args.total_epochs,
        adam_warmup=args.adam_warmup,
        seeds=tuple(args.seeds),
        rel_l2_threshold=args.success_rel_l2_threshold,
        problem_label=PROBLEM_LABEL,
    )
    plot_comparison(
        results=results,
        out_path=os.path.join(out_dir, "optimiser_comparison.png"),
        k=0.0,  # unused; problem_label overrides the residual-panel title
        adam_warmup=args.adam_warmup,
        rel_l2_threshold=args.success_rel_l2_threshold,
        figsize=(10.0, 12.0) if args.portrait else (14.0, 10.0),
        problem_label=PROBLEM_LABEL,
    )

    np.savez(
        os.path.join(out_dir, "raw_histories.npz"),
        pipelines=np.asarray(args.pipelines),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        **{
            f"J_val_{r.pipeline}_seed{s.seed}": s.J_val_history
            for r in results for s in r.seeds
        },
        **{
            f"sol_l2_{r.pipeline}_seed{s.seed}": s.sol_l2_history
            for r in results for s in r.seeds
        },
    )
    print(f"\nAll comparison artefacts written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
