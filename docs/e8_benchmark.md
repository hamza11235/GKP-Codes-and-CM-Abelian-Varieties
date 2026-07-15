# Phase 4: the exact E8 benchmark

The `E8` root lattice has real dimension eight, so a compatible complex
structure makes it a complex abelian fourfold.  We use the coordinate model

\[
E_8=D_8\cup\left(D_8+(1/2,\ldots,1/2)\right)
\]

and the paired quarter-turn

\[
R(x_1,x_2,\ldots,x_7,x_8)
=(x_2,-x_1,\ldots,x_8,-x_7).
\]

This transformation preserves both cosets in the coordinate description of
`E8`.  In the integral basis used by `src/gkp_systole/e8.py`, it becomes an
integral matrix `J` satisfying

\[
J^2=-I,
\qquad
J^T GJ=G.
\]

The resulting Riemann form

\[
A=GJ
\]

is integral, alternating, and unimodular.  The exact validator therefore
certifies polarization type

\[
(1,1,1,1).
\]

The order-four action gives Gaussian complex multiplication; as an unpolarized
complex torus the construction is isogenous to a fourth power of the square CM
elliptic curve.  The polarization is coupled rather than the orthogonal product
polarization.

## Qubit relative systole

The `E8` roots give

\[
\lambda_1^2(E_8)=2
\]

with 240 shortest vectors.  For type `(2,2,2,2)` under the fixed-principal
metric convention,

\[
\ell^2=\frac{\lambda_1^2}{2^2}=\frac12.
\]

Opposite roots determine the same class modulo `2`, so

\[
N_{\rm class}=120,
\qquad
N_{\rm lift}=240.
\]

Both the uniform SVP shortcut and the independent full 255-class CVP path
return these exact values.

## Optimality

[Viazovska's eight-dimensional sphere-packing theorem](https://arxiv.org/abs/1603.04246)
proves that the `E8` packing is densest even among all sphere packings.  In
particular, among covolume-one eight-dimensional lattices,

\[
\lambda_1^2\le 2.
\]

Principally polarized metrics in this normalization have covolume one, and our
exact construction shows that `E8` lies in that subclass.  Consequently

\[
\ell^2\le\frac12
\]

for the corresponding uniform type `(2,2,2,2)`, with equality at this `E8`
PPAV.  Thus this is a proven global optimum for the small-noise exponent within
the fixed-principal uniform-type problem, not merely the best candidate scanned.

The executed derivation is in `notebooks/03_e8_benchmark.ipynb`.
