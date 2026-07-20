# Phase 9: radial robustness of CM passive Clifford enhancements

Exact automorphism counts are discontinuous.  A CM point may have many exact
passive logical gates, while an arbitrarily small generic deformation usually
leaves only the unavoidable image of `{+I,-I}`.  Phase 9 therefore measures both
exact survival and a continuous notion of approximate passivity.

## Logical-action defect

For a CM automorphism `U` and deformed compatible metric `G`, define

$$
\delta_U(G)
=
\frac{1}{\sqrt{2g}}
\left\|
G^{-1/2}(U^TGU-G)G^{-1/2}
\right\|_F.
$$

The polarization is fixed exactly throughout, so `delta_U=0` means that `U`
again preserves both the polarization and metric and is an exact passive
automorphism.  Several automorphisms can implement the same logical Clifford.
For a logical action `a`, Phase 9 therefore uses

$$
\delta_a(G)=\min_{U\mapsto a}\delta_U(G).
$$

This counts logical gates rather than redundant geometric representatives.

## Removing the generic subgroup

The logical actions induced by `{+I,-I}` are present for every metric.  They
form one action when the kernel exponent is two and two actions at odd level.
The primary score excludes this complete generic image and measures only the
additional logical actions supplied at the CM point:

$$
R_\tau(G)
=
\frac{1}{|\mathcal I_{\rm CM}\setminus\mathcal I_{\rm gen}|}
\sum_{a\in\mathcal I_{\rm CM}\setminus\mathcal I_{\rm gen}}
\exp\left[-\left(\frac{\delta_a(G)}{\tau}\right)^2\right],
\qquad \tau=0.02.
$$

Thus `R_tau=1` at the CM baseline and tends to zero as the CM-only actions
become strongly nonpassive.  Literal epsilon-passive counts are also recorded
for

$$
\varepsilon\in
\{10^{-8},0.0025,0.005,0.01,0.02,0.05,0.10\}.
$$

## Search protocol

The experiment uses the same five within-type CM champions, tangent coordinates,
and radii as Phase 8:

$$
r\in\{0.005,0.01,0.02,0.05,0.10\}.
$$

At every candidate-radius pair:

- 32 shared Sobol directions initialize all searches;
- 32 held-out Sobol directions provide a pure space-filling control;
- 32 Bayesian UCB directions maximize gate retention;
- 32 independent Bayesian UCB directions minimize gate retention.

There are therefore 128 unique deformations per search, 25 searches, and 3,200
total evaluations.  Every deformation also receives an exact relative-systole
calculation, allowing the gate and distance objectives to be compared along
the same paths.

## Exact-gate result

Every one of the 3,200 nonzero generic deformations loses every exact CM-only
logical action at tolerance `1e-8`.  The surviving exact logical image always
equals the expected generic minimum:

- one action for exponent-two kernels, because `-I` acts trivially;
- two actions when `-I` is logically distinct.

This confirms numerically that exact CM gate enhancement sits on a special
locus rather than persisting throughout an open neighborhood.

## Approximate-gate result

The best- and worst-retention directions show substantial anisotropy:

| radius | mean best enhanced retention | mean worst enhanced retention | mean ell ratio, best direction | mean ell ratio, worst direction |
|---:|---:|---:|---:|---:|
| 0.005 | 0.953364 | 0.875493 | 0.996162 | 0.994171 |
| 0.010 | 0.839443 | 0.605564 | 0.992311 | 0.989424 |
| 0.020 | 0.579197 | 0.184883 | 0.985295 | 0.979602 |
| 0.050 | 0.196606 | 0.000811 | 0.960856 | 0.952520 |
| 0.100 | 0.028553 | below `5e-10` | 0.919405 | 0.904030 |

At radius 0.005, the best directions retain essentially all CM-only actions as
`epsilon=0.01` approximate gates for all five types.  By radius 0.05, none of
the types retains an enhanced action below that threshold, even in the best
direction.  Exact gates disappear immediately, but approximate usefulness has
a finite and highly directional decay scale.

The best-retention Bayesian arm improves on matched Sobol search in 21 of 25
cases.  The worst-retention arm improves in 11 cases and ties in six; at larger
radii many directions have already saturated extremely close to zero.

## Relation to relative systole

Within each fixed candidate-radius search, the Spearman correlation between
gate retention and `ell^2` is positive in 22 of 25 cases.  The mean within-search
Spearman coefficient is `0.228`.  The best-retention directions also have a
higher mean `ell^2` ratio than the worst-retention directions at every radius.

This is not evidence of a sharp functional relationship, but it argues against
a simple tradeoff in which preserving passive gates necessarily damages code
distance.  In these local experiments the two desirable properties are mildly
aligned.

## Validation and claim boundary

Across the complete ledger:

- maximum radius error is below `2.0e-11`;
- maximum polarization residual is below `3.5e-13`;
- maximum log-volume residual is below `9.8e-14`;
- all 50 best/worst metrics were recomputed with the independent 60-digit CVP
  implementation;
- maximum `ell^2` discrepancy is below `1.4e-15`;
- vectorized automorphism defects agree with an independent scalar calculation
  below `2.2e-16`.

Phase 9 tracks the degradation of the logical actions that exist at each CM
baseline.  It does not exhaustively enumerate potentially new automorphisms of
every deformed metric.  Such new exact symmetries are nongeneric, but ruling
them out rigorously would require a separate exact lattice-automorphism audit.
