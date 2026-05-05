# Box-Cox investigation note

## Summary

The Box-Cox implementation in `pinn_ssbroyden_1d.py:_transform_objective` and
`pinn_helmholtz_2d.py:_transform_objective` is **mathematically correct**.
The reason `lambda = 1` wins the sweep of section 4.2.3 of the thesis is the
*regime*, not a bug:

| location of `J` during SSBroyden | curvature factor `|g''_lambda(J)|` for `lambda < 1`              |
|----------------------------------|------------------------------------------------------------------|
| `J >> 1`                         | `|lambda - 1| * J^{lambda - 2}` — vanishes as `lambda - 2 < 0`   |
| `J << 1`                         | `|lambda - 1| * J^{lambda - 2}` — diverges as `J -> 0`           |

In the user's 1D BVP run with `k = 4`, after the 2000-epoch Adam phase the
validation `J` plateaus around `10^4`. With `J = 10^4`,

| `lambda` | `g'_lambda(J)`    | `|g''_lambda(J)|`     |
|----------|-------------------|-----------------------|
| `1.0`    | `1`               | `0`                   |
| `0.5`    | `1.0e-2`          | `5.0e-7`              |
| `0.0`    | `1.0e-4`          | `1.0e-8`              |
| `-0.25`  | `3.16e-5`         | `3.16e-9`             |

So no curvature gets amplified anywhere in the SSBroyden phase. The
self-scaled Broyden update just sees a contracted version of the same
landscape, with smaller secant pairs `(s_k, y_k)`, which the
`damping = 1e-12` cutoff of `ssbroyden.py:181` shaves off more aggressively
than for `lambda = 1`. This explains the four-orders-of-magnitude gap in
the reported results.

Urban et al. (2025) hit `J ~ 10^{-3}` early in their SSBroyden phase by
running 10 000 Adam iterations on a smaller network with 10 000 collocation
points; in that regime the same `lambda = 0.5, 0.0` runs do amplify
curvature and beat `lambda = 1`. The 2D Helmholtz benchmark in their
section 5 is the configuration to test this empirically.

## What the new scripts do about it

* `boxcox_diagnostic.py` — independently verifies (i) numerical
  equivalence of the implemented `expm1(lam log s) / lam` form against the
  naive `(s^lam - 1) / lam`, (ii) autograd derivatives against the
  analytical `g'_lambda(s) = s^{lambda - 1}` and `g''_lambda(s) = (lambda - 1) s^{lambda - 2}`,
  (iii) that the empirical `J` trajectory of an Adam->SSBroyden run
  on the 1D BVP lives outside the curvature-amplification window. Prints
  PASS / CHECK and emits a four-panel figure.

* `boxcox_sweep_1d_finegrained.py` — replaces the five-point `lambda` grid
  with eleven points over `[-0.5, 1.0]` (so the sweep can see whether
  negative `lambda` ever helps), averages over five seeds, and exposes a
  `--engage-threshold` knob. With `--engage-threshold 1.0` the
  transformation stays at identity until validation `J` first crosses 1, and
  only then switches to Box-Cox; this is the cleanest way to test whether
  Box-Cox helps once the optimiser is in the small-loss regime.

* `boxcox_sweep_2d_helmholtz.py` — the same fine-grained, multi-seed sweep
  on the 2D Helmholtz benchmark of Urban et al. 2025 section 5. This is the
  test that the published claim about Box-Cox actually replicates in our
  own pipeline.

## How the implementation works (line-by-line check)

The transformation in `pinn_ssbroyden_1d.py:160-174` reads:

```python
def _transform_objective(self, J_raw):
    eps = self.loss_eps
    if self.loss_transform == "boxcox":
        lam = self.loss_lambda
        shifted = J_raw + eps
        if lam == 0.0:
            return torch.log(shifted)
        return torch.expm1(lam * torch.log(shifted)) / lam
    ...
```

Identities:

* `expm1(x) = e^x - 1` exactly, evaluated stably for small `|x|`.
* `expm1(lam * log(s)) / lam = (s^lam - 1) / lam` for `lam != 0`.
* As `lam -> 0`, `(s^lam - 1) / lam -> log(s)` — handled by the explicit
  branch.
* The shift by `eps = 1e-12` regularises `log(0)` on the (rare) iterations
  where `J_raw` underflows to zero on float32; for `J_raw >> eps` the shift
  is operationally invisible.

Autograd of this expression:

* `d/dJ [expm1(lam * log(J + eps)) / lam] = (J + eps)^{lam - 1}`,
  which `boxcox_diagnostic.py` cross-checks at five values of `s` spanning
  `[10^{-6}, 10^4]` to better than `1e-9` relative precision (float64).

Line-search interaction:

* `loss_eval()` in `pinn_ssbroyden_1d.py:335-337` returns the *transformed*
  `J_obj`, so the Armijo condition compares transformed values across
  candidate steps. Gradients fed into the Broyden update come from
  `J_obj.backward()`, i.e. they are `g'_lambda(J) * grad J`. The secant pair
  `(s_k, y_k)` is therefore consistent with the Hessian of the transformed
  objective.

* The implementation is correct end to end; the failure mode in the 1D BVP
  is structural (regime), not algorithmic.

## Recommendations for the thesis

* Replace the 5-point Box-Cox sweep figure with the 11-point one from
  `boxcox_sweep_1d_finegrained.py`. Report mean +/- std across seeds
  (currently a single seed is plotted, so any reviewer can object to the
  monotone trend being seed-specific).

* Add a paragraph after Fig. 4.5 explaining the regime argument: Box-Cox
  amplifies curvature only when `J << 1`, which the 1D BVP at `k = 4` does
  not reach during the 3000-epoch SSBroyden phase. This recasts the
  negative result as a *regime statement*, not as a refutation of [8].

* Add the 2D Helmholtz Box-Cox figure (`boxcox_sweep_2d_helmholtz.py`) to
  section 4.3 as the test of whether the Urban et al. claim replicates in
  the regime the paper targets.

* Optionally, run the delayed-engagement variant
  (`boxcox_sweep_1d_finegrained.py --engage-threshold 1.0`) and report it as
  a third panel: does Box-Cox help once we engage it inside the small-loss
  regime, even on the 1D BVP? This isolates the regime claim cleanly.
