"""Ad-hoc comparison: urban (fp64 + strong Wolfe) vs inhouse (fp32 + Armijo)
on the 20-seed 1D optimiser comparison. Parses the per-seed tables from both
summary_table.txt files and reports (1) matched depth statistics over seeds
that succeed under BOTH engines, and (2) a divergence-vs-stall breakdown of
the failures. Read-only; prints to stdout."""
from __future__ import annotations

import os
import re
import statistics as st

URBAN = "bvp1d_k4_optim_compare_urban_20260613_152100/summary_table.txt"
INHOUSE = "bvp1d_k4_optim_compare_merged_20seeds/summary_table.txt"

ROW = re.compile(
    r"^\s*(adam(?:_\w+)?)\s+(\d+)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(ok|FAIL)\s*$"
)


def parse(path):
    """Return {pipeline: {seed: dict(J, sol, rel, ok)}} from the per-seed block."""
    out = {}
    in_block = False
    with open(path) as fh:
        for line in fh:
            if "Per-seed final metrics" in line:
                in_block = True
                continue
            if not in_block:
                continue
            m = ROW.match(line)
            if not m:
                continue
            pipe, seed, J, sol, rel, status = m.groups()
            out.setdefault(pipe, {})[int(seed)] = {
                "J": float(J), "sol": float(sol), "rel": float(rel),
                "ok": status == "ok",
            }
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    u = parse(os.path.join(here, URBAN))
    h = parse(os.path.join(here, INHOUSE))
    pipes = ["adam_bfgs", "adam_ssbfgs", "adam_ssbroyden"]

    print("=" * 78)
    print("(1) MATCHED DEPTH  — seeds that succeed under BOTH engines")
    print("=" * 78)
    print(f"{'pipeline':>16} {'both_ok':>8} "
          f"{'med solL2 IN':>14} {'med solL2 URB':>14}   "
          f"{'med J IN':>12} {'med J URB':>12}")
    for p in pipes:
        hh, uu = h.get(p, {}), u.get(p, {})
        both = sorted(s for s in hh
                      if hh[s]["ok"] and s in uu and uu[s]["ok"])
        sol_in = [hh[s]["sol"] for s in both]
        sol_ur = [uu[s]["sol"] for s in both]
        J_in = [hh[s]["J"] for s in both]
        J_ur = [uu[s]["J"] for s in both]
        print(f"{p:>16} {len(both):>8} "
              f"{med(sol_in):>14.3e} {med(sol_ur):>14.3e}   "
              f"{med(J_in):>12.3e} {med(J_ur):>12.3e}")
        print(f"{'  (seeds)':>16} {','.join(str(s) for s in both)}")

    print()
    print("=" * 78)
    print("(2) BEST-RUN DEPTH  — min over each engine's own successful seeds")
    print("=" * 78)
    print(f"{'pipeline':>16} {'min J IN':>12} {'min J URB':>12}   "
          f"{'min sol IN':>12} {'min sol URB':>12}")
    for p in pipes:
        hh, uu = h.get(p, {}), u.get(p, {})
        hok = [hh[s] for s in hh if hh[s]["ok"]]
        uok = [uu[s] for s in uu if uu[s]["ok"]]
        mjin = min((d["J"] for d in hok), default=float("nan"))
        mjur = min((d["J"] for d in uok), default=float("nan"))
        msin = min((d["sol"] for d in hok), default=float("nan"))
        msur = min((d["sol"] for d in uok), default=float("nan"))
        print(f"{p:>16} {mjin:>12.3e} {mjur:>12.3e}   "
              f"{msin:>12.3e} {msur:>12.3e}")

    print()
    print("=" * 78)
    print("(3) FAILURE BREAKDOWN  — divergence (final J > 10) vs stall (J <= 10)")
    print("    handover hands off around J ~ 0.8, so J > 10 means it blew UP.")
    print("=" * 78)
    print(f"{'pipeline':>16} {'engine':>8} {'n_fail':>7} "
          f"{'diverge':>8} {'stall':>7} {'max final J':>14}")
    for p in pipes:
        for name, data in (("inhouse", h), ("urban", u)):
            dd = data.get(p, {})
            fails = [dd[s] for s in dd if not dd[s]["ok"]]
            diverge = sum(1 for d in fails if d["J"] > 10.0)
            stall = sum(1 for d in fails if d["J"] <= 10.0)
            maxJ = max((d["J"] for d in fails), default=float("nan"))
            print(f"{p:>16} {name:>8} {len(fails):>7} "
                  f"{diverge:>8} {stall:>7} {maxJ:>14.3e}")


if __name__ == "__main__":
    main()
