# Exact reconstruction of the `g=3`, type `(1,1,2)` record

The first Phase-8 generic control improved the bounded ternary-Hermitian CM
record from `ell^2=1` to approximately `1.06896`, but only one logical class
was exactly active.  A broader 12-dimensional search reached `1.09914`, and a
constrained max--min refinement then equalized all three nonzero logical
classes near

\[
1.1547.
\]

## Algebraic recognition

Multiplying the refined numerical metric by `sqrt(3)` puts every entry within
`8.8e-5` of a half-integer matrix.  The recognized model is

\[
G=\frac{G_{\rm core}}{\sqrt3},
\]

with

\[
G_{\rm core}=
\begin{pmatrix}
12&-12&7&4&6&-11\\
-12&20&-5&-4&-10&15\\
7&-5&9&1/2&0&-4\\
4&-4&1/2&3&4&-11/2\\
6&-10&0&4&8&-10\\
-11&15&-4&-11/2&-10&15
\end{pmatrix}.
\]

Recognition proposes the matrix; it does not certify it.  Certification is
performed independently with exact rational arithmetic.

## Exact relative systole

The centralized PPAV validator proves that the exact matrices satisfy:

- polarization type `(1,1,2)`;
- positivity;
- `J^2=-I`;
- exact metric/polarization compatibility;
- the required determinant normalization.

Exhaustive exact closest-vector enumeration then gives

\[
\ell^2_{X,L}=\frac2{\sqrt3}.
\]

All three nonzero kernel classes attain the minimum, with 36 shortest lifts in
total.  Thus the reconstructed point improves the original bounded Gaussian
CM record `ell^2=1` by about 15.47 percent.

## CM certificate

Write

\[
J=\frac{S}{\sqrt3}.
\]

The reconstructed rational endomorphism satisfies

\[
S^2=-3I_6.
\]

An integral change of lattice of determinant 72 conjugates `S` to three copies
of

\[
\begin{pmatrix}0&-3\\1&0\end{pmatrix}.
\]

Therefore the rational complex torus splits as

\[
X\sim E_{i\sqrt3}^{,3}.
\]

It is a CM abelian threefold with field `Q(sqrt(-3))`; the displayed elliptic
order has discriminant `-12`.  The rational commutant of `S` has dimension 18,
as expected for three copies of a quadratic CM representation.

The apparently generic improvement is therefore **not** a non-CM
counterexample.  It is another CM point that lay outside the initial bounded
ternary-Hermitian presentation.

## Claim strength

This establishes the current exact project record for type `(1,1,2)`.  It does
not prove global optimality over the moduli space.

## Reproducibility

- Executed notebook: `notebooks/08_type112_exact.ipynb`
- Exact model: `src/gkp_systole/type112.py`
- Regression tests: `tests/test_type112.py`
- Updated ledger: `data/phase8_result_ledger.{json,csv}`
