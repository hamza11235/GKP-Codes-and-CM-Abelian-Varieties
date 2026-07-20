# Phase 10: blind bounded global search

## Question

Phases 7--9 started at known CM points. Phase 10 removes that advantage and
asks whether generic numerical optimization, seeing only the relative systole
objective, can discover a point as strong as the bounded-CM champion.

For each fixed polarization type `D`, the search maximizes

```text
ell(X,L)^2
```

over compatible metrics in a bounded intrinsic ball. The search is blind:
CM labels, discriminants, automorphism groups, and passive-gate counts are not
loaded until every objective query has completed.

## Frozen protocol

- Types: `(1,3)`, `(1,5)`, `(1,1,2)`, `(1,1,3)`, `(1,2,2)`.
- Reference: the canonical product metric `diag(D,D)`, determined only by `D`.
- Coordinates: the full compatible-metric space `Sp/U`, with dimension
  `g(g+1)` (six for surfaces and twelve for threefolds).
- Radius: RMS affine-invariant metric distance, with balls of radius
  `0.25`, `0.50`, `1.00`, and `1.50`.
- Objective: relative systole squared, computed by exhaustive kernel-class
  enumeration and floating-point branch-and-bound CVP.
- Equal budget: 96 objective calls for each of three methods:
  - scrambled Sobol uniform-in-ball search;
  - rank-mu CMA-ES with two deterministic restarts;
  - Gaussian-process UCB, using 32 initial Sobol points and 64 adaptive calls.
- Total: `5 * 4 * 3 * 96 = 5,760` blind objective evaluations.

The intrinsic ball is implemented exactly in the Cartan chart. If `x` is a
coordinate vector and `g` is the complex dimension, then

```text
d_RMS(G_0,G(x)) = 2 ||x|| / sqrt(2g).
```

Thus Euclidean projection in coordinate space is projection onto the stated
affine-invariant ball.

## Results

At the largest radius, the original frozen comparison against the Phase-5
population champion is:

| type `D` | best method | best blind `ell^2` | bounded-CM champion `ell^2` | ratio |
|---|---:|---:|---:|---:|
| `(1,3)` | Bayesian UCB | 0.730917844 | 0.816496581 | 0.895188 |
| `(1,5)` | CMA-ES | 0.470379128 | 0.539359890 | 0.872106 |
| `(1,1,2)` | CMA-ES | 0.869062246 | 1.000000000 | 0.869062 |
| `(1,1,3)` | Bayesian UCB | 0.777729500 | 1.154700538 | 0.673534 |
| `(1,2,2)` | Bayesian UCB | 0.735414635 | 1.000000000 | 0.735415 |

No blind evaluation beat or tied its Phase-5 population champion. Across the
20 type-radius comparisons, CMA-ES supplied 10 winners, Bayesian UCB supplied
9, and Sobol supplied 1. This matters because the conclusion is not dependent
on a single optimization method.

The mean best-blind/CM ratio increases monotonically with radius:

| intrinsic radius | mean ratio |
|---:|---:|
| 0.25 | 0.430522 |
| 0.50 | 0.487067 |
| 1.00 | 0.632141 |
| 1.50 | 0.809061 |

The earlier systole work contains stronger exact CM reconstructions for
`(1,5)` (`ell^2=sqrt(2/5)`) and `(1,1,2)` (`ell^2=2/sqrt(3)`). Consequently,
the blind/strongest-known-CM ratios for those two types are 0.744 and 0.753,
respectively, rather than the 0.872 and 0.869 frozen Phase-5 ratios above.
The original protocol ledger is retained unchanged; the consolidated release
adds this stricter cross-workstream comparison.

Most radius-1.5 winners lie close to the boundary. Therefore the scan has not
saturated; a larger expanding-ball phase is justified. The present result is
strong evidence that the known CM records are not easy artifacts of a weak
generic search, but it is not global optimality certification.

## Numerical validation

- All 5,760 metrics preserve the fixed polarization and volume by construction.
- Maximum observed radius overrun: less than `7e-15`.
- Every one of the 60 method winners was recomputed independently using a
  70-digit finite-kernel CVP calculation.
- Maximum high-precision discrepancy in `ell^2`: less than `2.3e-16`.
- The full sorted kernel-distance spectrum is stored for every evaluation.
  It is used as an isomorphism diagnostic, not as proof of arithmetic identity.

## Claim boundary

This is a deterministic bounded numerical search in a global coordinate chart.
It does not cover the entire noncompact moduli space, reduce endpoints to a
fundamental domain, or prove that a floating-point endpoint is CM. The CM
comparison is against the strongest records in the bounded Phase-5 CM
population, not against all CM polarized abelian varieties.

The next search should enlarge the intrinsic balls adaptively and add
fundamental-domain reduction/endpoint clustering. Arithmetic recognition is
appropriately deferred until a blind endpoint numerically coincides with a
known or new extremal spectrum.
