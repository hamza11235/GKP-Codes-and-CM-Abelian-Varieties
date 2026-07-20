# Phase 7: equal-distance generic controls

Phase 7 resolves the main control-design limitation found in Phase 6. Fixed
transvection coefficients produced very different actual metric displacements
on different CM baselines. Here every generic control is placed at an exactly
matched, coordinate-invariant distance from its paired CM metric.

## Preregistered geometry

For positive-definite metrics `G0` and `G1`, define the RMS affine-invariant
distance

$$
d_{\rm RMS}(G_0,G_1)
=
\frac{1}{\sqrt{2g}}
\left\|\log\left(G_0^{-1/2}G_1G_0^{-1/2}\right)\right\|_F.
$$

This distance is invariant under simultaneous changes of lattice coordinates
by congruence. The factor `sqrt(2g)` permits the same numerical radii to be used
in real dimensions four and six.

For each CM candidate, three deterministic symplectic directions were sampled.
If a direction consists of four integer vectors `v_j` and fixed signed weights
`w_j`, then

$$
S(\alpha)=
\prod_{j=1}^{4}
\left(I+\alpha\pi w_j v_j(v_j^T A)\right),
\qquad
G(\alpha)=S(\alpha)^T G_0 S(\alpha).
$$

The scalar `alpha` was determined by bisection so that the metric landed at one
of the two preregistered radii

$$
d_{\rm RMS}(G_0,G(\alpha))=0.02
\quad\text{or}\quad
0.10.
$$

The same three directions were used at both radii. Seeds, directions, weights,
radii, and the primary response were fixed before outcomes were inspected.
There was no rejection or adaptive resampling.

Because every factor is symplectic,

$$
S^TAS=A.
$$

Consequently, dimension, full polarization matrix, type `D`, kernel `K(L)`,
and volume remain fixed. Only the compatible metric/complex structure changes.

## Results

The primary response is candidate-level

$$
\Delta\ell^2
=
\frac{1}{3}\sum_{j=1}^{3}\ell^2_{\rm control,j}
-\ell^2_{\rm CM}.
$$

Negative values favor the CM baseline. Every tested type has a negative mean at
both radii, and every descriptive 95% interval lies strictly below zero.

| type | candidates | radius | mean control minus CM ell^2 | 95% interval | mean control/CM ratio |
|---|---:|---:|---:|---:|---:|
| `(1,3)` | 876 | 0.02 | -0.001779 | [-0.002018, -0.001540] | 0.996557 |
| `(1,3)` | 876 | 0.10 | -0.008233 | [-0.009352, -0.007113] | 0.985766 |
| `(1,5)` | 915 | 0.02 | -0.000452 | [-0.000595, -0.000310] | 0.999132 |
| `(1,5)` | 915 | 0.10 | -0.001670 | [-0.002347, -0.000993] | 0.999081 |
| `(1,1,2)` | 1,051 | 0.02 | -0.002158 | [-0.002413, -0.001903] | 0.997144 |
| `(1,1,2)` | 1,051 | 0.10 | -0.008823 | [-0.010037, -0.007609] | 0.990023 |
| `(1,1,3)` | 1,070 | 0.02 | -0.001666 | [-0.001902, -0.001430] | 0.997198 |
| `(1,1,3)` | 1,070 | 0.10 | -0.006641 | [-0.007761, -0.005521] | 0.990831 |
| `(1,2,2)` | 253 | 0.02 | -0.002224 | [-0.002719, -0.001729] | 0.995724 |
| `(1,2,2)` | 253 | 0.10 | -0.009400 | [-0.011740, -0.007060] | 0.983274 |

Pooling all 4,165 paired candidates gives:

| radius | mean control minus CM ell^2 | 95% interval | mean control/CM ratio |
|---:|---:|---:|---:|
| 0.02 | -0.001581 | [-0.001693, -0.001469] | 0.997385 |
| 0.10 | -0.006602 | [-0.007132, -0.006072] | 0.990915 |

The mean relative advantage of the CM baseline is therefore about 0.26% at
radius 0.02 and 0.91% at radius 0.10. These effects are smaller than the raw
Phase-6 estimate, as expected after removing its displacement tail, but their
sign is completely consistent across types and radii.

The corresponding radial profiles are saved as
`figures/phase7_radial_profiles_by_type.png` and
`figures/phase7_radial_profiles_by_symmetry.png` and are reproduced directly
in the Phase-7 notebook.

## Connection with passive symmetry

Phase 5 classified 521 of the 4,165 CM candidates as having a logical image
larger than the unavoidable generic image. These enhanced-symmetry CM points
are more resistant to equal-distance deformation:

| CM subset | candidates | radius | mean control/CM ratio |
|---|---:|---:|---:|
| minimal passive image | 3,644 | 0.02 | 0.998138 |
| enhanced passive image | 521 | 0.02 | 0.992120 |
| minimal passive image | 3,644 | 0.10 | 0.994616 |
| enhanced passive image | 521 | 0.10 | 0.965033 |

At radius 0.10, a generic deformation reduces `ell^2` by about 3.50% on average
around enhanced-symmetry CM candidates, compared with about 0.54% around CM
candidates having only the minimal passive image. The enhanced subset has the
more negative paired difference in every one of the five polarization types.

This provides a sharper version of the Phase-5 association: the CM points with
extra passive Clifford symmetry also behave as more pronounced local peaks of
the relative-systole function under the sampled directions.

## Numerical validation

The final ledger contains 24,990 controls. Across all controls:

- the maximum target-radius error is below `2e-11`;
- the maximum polarization residual is below `5.7e-14`;
- the maximum log-volume residual is below `8.7e-14`;
- ten extremal controls were recomputed with an independent 60-digit mpmath CVP
  search, with maximum discrepancy `1.11e-16`.

## Claim boundary

Phase 7 supports:

> Under the preregistered equal-affine-distance protocol, the bounded CM
> candidates have higher mean relative systole than their generic-real
> deformations in every tested polarization type and at both tested radii.
> The effect is stronger around CM candidates with enhanced passive logical
> symmetry.

It does not prove global or local optimality of individual CM points, a
canonical measure on moduli space, or that every generated control has a
separately certified non-CM endomorphism ring. Directional sampling also tests
three rays per candidate rather than the full tangent sphere.

The frozen protocol, row-level ledger, aggregate summary, and precision audit
are stored under `data/phase7_*`.
