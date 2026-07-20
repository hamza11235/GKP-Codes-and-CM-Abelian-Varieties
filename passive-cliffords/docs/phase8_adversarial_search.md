# Phase 8: adversarial local search around CM champions

Phase 7 showed that the mean of three generic directions lies below the CM
baseline.  That does not rule out an exceptional direction in which the
relative systole increases.  Phase 8 therefore replaces average-direction
sampling by a fixed-budget adversarial search for counterexamples.

## Frozen candidate set and search question

For each of the five Phase-5 polarization types, the selected baseline is the
record with the largest stored `ell^2`.  Ties prefer the larger logical image
and then the lexicographically earliest stable candidate ID.  All five selected
champions have enhanced passive logical symmetry.

At each radius `r`, the numerical question is

$$
M_{X_0}(r)
=
\max_{d_{\rm RMS}(G_0,G)=r}
\frac{\ell^2(G)}{\ell^2(G_0)}.
$$

Finding any value greater than one falsifies local maximality at that radius.
Failure to find one is evidence, but not a proof, that the CM baseline is a
local peak.

## Geometry-respecting coordinates

Let `A` be the fixed polarization form and `G` the compatible metric.  The
symplectic Lie algebra satisfies

$$
K^T A + A K = 0.
$$

Writing `K=A^{-1}H` with `H=H^T` parameterizes this Lie algebra.  Its induced
metric tangent is

$$
\dot G = K^T G + G K.
$$

Directions for which `dot G=0` only stabilize the same metric.  Phase 8 removes
that kernel with an SVD and orthonormalizes the remaining directions in the
RMS affine-invariant tangent norm.  The effective dimensions are therefore

$$
\dim_{\mathbb R}\mathcal A_g = g(g+1),
$$

namely six coordinates for the surfaces and twelve for the threefolds.

A unit tangent direction `xi` determines a Lie-algebra element `K(xi)`.  The
finite deformation is

$$
S(\alpha,\xi)=\exp(\alpha K(\xi)),
\qquad
G(\alpha,\xi)=S(\alpha,\xi)^T G_0S(\alpha,\xi).
$$

Bisection chooses `alpha` so that the RMS affine-invariant distance is exactly
one of

$$
r\in\{0.005,0.01,0.02,0.05,0.10\}.
$$

Since `S` is symplectic, the full polarization matrix, type, kernel, and volume
remain fixed.

## Matched search budgets

Each candidate-radius pair receives two equal 64-query search arms:

1. **Pure Sobol:** 32 shared space-filling directions plus 32 held-out Sobol
   directions.
2. **Bayesian optimization:** the same 32 shared directions plus 32 adaptive
   directions.

The Bayesian surrogate is a fixed Matérn-5/2 Gaussian process on unit tangent
coordinates.  At each step it selects the unused point maximizing

$$
a(\xi)=\mu(\xi)+2\sigma(\xi),
$$

where `mu` and `sigma` are the posterior mean and standard deviation.  Thus it
balances directions predicted to have high `ell^2` against uncertain regions.
All Sobol scrambles and acquisition pools have deterministic SHA-256 seeds.

## Results

There are 25 searches and 2,400 exact relative-systole evaluations.  No search
found `ell^2/ell^2_CM > 1`.

| radius | mean best Sobol ratio | mean best Bayesian ratio | closest ratio found |
|---:|---:|---:|---:|
| 0.005 | 0.997159 | 0.997506 | 0.999022 |
| 0.010 | 0.994645 | 0.995359 | 0.998993 |
| 0.020 | 0.989096 | 0.989536 | 0.994569 |
| 0.050 | 0.972262 | 0.975657 | 0.991764 |
| 0.100 | 0.946107 | 0.959611 | 0.992069 |

Bayesian optimization strictly improves on the matched Sobol best in 19 of 25
searches, ties it in three, and trails it in three.  Its advantage is largest at
radius 0.10.  This confirms that the adaptive arm is finding harder directions:
the negative result is not an artifact of relying only on three random rays.

The closest approach to a counterexample occurs for type `(1,5)` at radius
0.005, where the best Bayesian direction reaches

$$
\ell^2/\ell^2_{\rm CM}=0.9990217926.
$$

Even at radius 0.10 the type `(1,5)` optimizer finds a comparatively flat
direction with ratio `0.9920691706`, but it remains below the CM value.

## Validation

Across all 2,400 evaluations:

- maximum target-radius error is below `2.0e-11`;
- maximum polarization residual is below `2.3e-13`;
- maximum log-volume residual is below `7.0e-14`;
- all tangent bases have the expected dimension and are orthonormal to the
  tested tolerance;
- all 25 winning metrics were independently recomputed by a 60-digit mpmath
  closest-vector search, with maximum discrepancy `3.34e-15`.

## Claim boundary

Phase 8 supports:

> Within the preregistered five-radius, 64-query adversarial protocol, neither
> space-filling Sobol search nor Bayesian optimization found a deformation
> beating any of the five within-type CM champions.  The adaptive optimizer
> usually found harder directions than Sobol, while remaining below the CM
> baseline.

It does not establish a mathematical local maximum, a global maximum over
moduli, or optimality of CM points as a class.  The acquisition pool is finite,
the GP is a numerical surrogate, and the relative-systole objective is
nonsmooth where shortest logical classes exchange.  A later computer-assisted
certificate would need interval upper bounds over a covering of the entire
tangent sphere.
