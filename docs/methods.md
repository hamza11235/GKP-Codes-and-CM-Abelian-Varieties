# Computational methods

## Relative systole as a finite family of CVPs

Let `A` be an integral nonsingular alternating polarization matrix and `G` a
compatible positive-definite Gram matrix in the same lattice basis. The logical
displacement group is

\[
K(L)=A^{-T}\mathbf Z^{2g}/\mathbf Z^{2g}.
\]

For a representative `c`, its squared torus distance to zero is

\[
q_G(c)=\min_{n\in\mathbf Z^{2g}}(c+n)^T G(c+n).
\]

The relative systole is the minimum of `q_G` over the nonzero finite kernel.

## Exact arithmetic backend

Rational metrics are represented with `fractions.Fraction`. The code computes an
exact `LDL^T` decomposition and recursively enumerates every integer coordinate
whose exact partial quadratic contribution does not exceed the best complete
candidate. No floating-point pruning is used.

## Floating and high-precision backends

Floating metrics use Cholesky branch-and-bound. Leading candidates are rebuilt
with `mpmath` and rechecked at 60--70 decimal digits. Floating results are not
called certified unless they are subsequently reconstructed exactly or enclosed
by interval arithmetic.

## Uniform polarization shortcut

For fixed-principal type `(d,...,d)`,

\[
\ell^2=\lambda_1^2/d^2.
\]

One exact shortest-vector problem therefore replaces all kernel CVPs. The full
kernel and shortcut remain independent implementations and are cross-checked on
known models.

## Polarized-abelian validation

For algebraic-scale data

\[
G=G_{\rm core}/\sqrt r,
\qquad J=J_{\rm num}/\sqrt r,
\]

exact validation checks

\[
J_{\rm num}^2=-rI,
\quad J_{\rm num}^TG_{\rm core}J_{\rm num}=rG_{\rm core},
\quad A=G_{\rm core}J_{\rm num}/r,
\]

as well as positivity, integrality, Smith polarization type, and determinant
normalization.

## CM candidate generation

- CM elliptic curves are generated from reduced primitive positive-definite
  binary quadratic forms.
- Products and coupled polarizations are generated from positive binary or
  ternary Hermitian forms over imaginary-quadratic orders.
- A simple quartic-CM family is constructed over `Q(zeta_5)` using an exact trace
  pairing.

The present Hermitian scans are bounded coefficient searches with partial
isometry reduction.

## Compatible-metric searches

At fixed polarization `A` and reference metric `G0`, tangent matrices satisfy

\[
H^TA+AH=0,
\qquad H^TG_0=G_0H.
\]

The compatible symmetric space has real dimension `g(g+1)`. Metrics are
parameterized by

\[
G(x)=\exp(H(x))^TG_0\exp(H(x)).
\]

Searches use deterministic Halton screening, coordinate pattern refinement,
independent tangent directions, and simplex-style derivative-free refinement.
Derivative-free methods are appropriate because the minimum over logical classes
is nonsmooth when the active class changes.

## Exact reconstruction

At a symmetric numerical candidate, active lifts satisfy

\[
v_i^T Gv_i=\ell^2.
\]

These equations are linear in the symmetric entries of `G` and in `ell^2`.
Solving them over the rationals can recover the exact metric shape. Algebraic
recognition only proposes a model; exact polarized-abelian validation and exact
CVP provide the final certificate.
