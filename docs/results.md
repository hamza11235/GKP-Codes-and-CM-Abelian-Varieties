# Consolidated results and claim ledger

This document separates global theorems, exact constructions, bounded survey
records, and numerical controls. Throughout, `ell^2` denotes the squared
relative systole. Comparisons are meaningful only at fixed polarization type and
metric convention.

## Verification levels

- **Global optimum:** an exact candidate plus an external or internal upper
  bound proving optimality over the stated moduli problem.
- **Exact construction:** exact polarized-abelian validation and exhaustive
  exact CVP/SVP enumeration.
- **Interval certified:** exhaustive outward-rounded interval enumeration.
- **Bounded record:** exact best value in a fully stated finite candidate list.
- **Numerical control:** floating-point search, rechecked at high precision but
  not by interval arithmetic.

## Known benchmarks

| model | type | convention | `ell^2` | classes/lifts | status |
|---|---:|---|---:|---:|---|
| square elliptic curve | `(2)` | fixed principal | `1/4` | `2 / 4` | exact benchmark |
| hexagonal elliptic curve | `(2)` | fixed principal | `1/(2 sqrt(3))` | `3 / 6` | algebraic benchmark; known optimum |
| verified `D4` PPAV | `(2,2)` | fixed principal | `1/(2 sqrt(2))` | `12 / 24` | exact period model |
| Klein-quartic Jacobian | `(2,2,2)` | fixed principal | `1/sqrt(7)` | `21 / 42` | exact period model |
| `E8` PPAV | `(2,2,2,2)` | fixed principal | `1/2` | `120 / 240` | exact global optimum |

The uniform SVP shortcut and full kernel/CVP path agree independently on every
benchmark for which both paths are practical.

## Dimension two: bounded imaginary-quadratic survey

The Phase-7 survey covers imaginary-quadratic order discriminants
`3 <= |Delta| <= 160`, Hermitian determinants 2 through 8, and diagonal bound
16. Elementary integral shears, mode exchange, and units are deduplicated. The
6,302 retained candidates are not a complete classification under the full
Hermitian isometry group.

| type | discriminant | exact `ell^2` | claim |
|---|---:|---:|---|
| `(1,2)` | `-4` | `1` | bounded exact record |
| `(1,3)` | `-24` | `sqrt(2/3)` | bounded exact record |
| `(1,4)` | `-36` | `7/12` | bounded exact record |
| `(1,5)` | `-55` | `4/sqrt(55)` | bounded exact record, later improved |
| `(1,6)` | `-3` | `1/sqrt(3)` | bounded exact record |
| `(1,7)` | `-68` | `4/sqrt(68)` | bounded exact record |
| `(1,8)` | `-47` | `3/sqrt(47)` | bounded exact record |
| `(2,2)` | `-8` | `1/sqrt(2)` | bounded exact record |
| `(2,4)` | `-4` | `1/2` | bounded exact record |

### Exact type `(1,5)` reconstruction

A full six-dimensional compatible-metric search found a numerical improvement
near `0.6324555`. Equal-distance equations for its 24 active lifts determine an
exact metric

\[
G=G_{\rm core}/\sqrt{10}.
\]

Exact validation and exhaustive CVP give

\[
\ell^2=2/\sqrt{10}=\sqrt{2/5},
\qquad N_{\rm class}=N_{\rm lift}=24.
\]

The Euclidean dual is a scaled `D4` lattice. An integral endomorphism squares to
`-10 I`, and a rational index-47 decomposition proves that the surface is
isogenous to `E_(i sqrt(10))^2`, hence CM by an order of discriminant `-40`.
Global optimality is open.

### Simple quartic-CM type `(1,5)` construction

The `Q(zeta_5)` candidate has

\[
\ell^2=
\sqrt{4/25+8\sqrt5/125}
=0.550552768188469\ldots.
\]

Outward-rounded interval CVP certifies ten shortest classes/lifts, interval width
below `5e-69`, and a positive gap greater than `0.1299` to the next candidate.
It is beaten by the exact reconstructed type-`(1,5)` surface above.

## Dimension three: bounded CM survey

The ternary-Hermitian scan covers `3 <= |Delta| <= 40`, diagonal entries at
most 5, and off-diagonal order-basis coefficients in `[-1,1]`. It is not a
complete classification under `GL(3,O_Delta)`.

| type | exact `ell^2` | classes/lifts | claim |
|---|---:|---:|---|
| `(1,1,2)` | `1` | `3 / 24` | bounded record, later improved |
| `(1,1,3)` | `2/sqrt(3)` | `8 / 72` | bounded exact record |
| `(1,2,2)` | `1` | `15 / 60` | bounded exact record |

### Exact type `(1,1,2)` reconstruction

A full 12-dimensional search improved the initial bounded value. Max--min
refinement equalized all three nonzero logical classes and suggested

\[
G=G_{\rm core}/\sqrt3.
\]

Independent exact validation and exhaustive CVP give

\[
\ell^2=2/\sqrt3,
\qquad N_{\rm class}=3,
\qquad N_{\rm lift}=36.
\]

The rational endomorphism satisfies `S^2=-3 I`. A degree-72 rational
decomposition proves that the threefold is isogenous to `E_(i sqrt(3))^3` and
is CM by `Q(sqrt(-3))`. Global optimality is open.

## Numerical controls

- Initial generic-real transvection controls: 10,000 total samples across types
  `(1,2)` and `(1,3)`; none beat the then-current CM baselines.
- Full `g=2` compatible-metric searches: six real directions, deterministic
  Halton screening, coordinate and simplex refinement, 60-digit rechecks.
- Full `g=3` compatible-metric searches: twelve real directions and analogous
  refinement. The type-`(1,1,2)` search led to the exact CM reconstruction.

These searches provide evidence and candidate discovery, not global search
certificates.

## Main empirical conclusion

In two independent nonuniform problems, a broader numerical search found a
stronger configuration than the initial bounded CM enumeration, and the stronger
configuration reconstructed as an exact CM variety. No certified non-CM winner
has yet been found. This motivates, but does not prove, a relationship between
CM arithmetic and extremal GKP relative systoles.
