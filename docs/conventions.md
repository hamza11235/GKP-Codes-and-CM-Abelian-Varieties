# Conventions from Mayrand--Royer

Source: M. Mayrand and B. Royer, *Complex abelian varieties and quantum error
correction: a mathematical framework for GKP codes*, arXiv:2605.28784v1
(27 May 2026).

This document records the conventions that affect the calculator. Page and
equation references refer to version 1 of the paper.

## Ambient symplectic and Hermitian spaces

The physical phase space is a real vector space `V` of dimension `2n` with a
symplectic form `E`. In standard real coordinates the paper uses

\[
E((x,y),(\widetilde x,\widetilde y))
=x\cdot\widetilde y-y\cdot\widetilde x.
\]

The translation operators are normalized as

\[
T_{(x,y)}=\exp\!\left[-i\sqrt{2\pi}
(y\cdot\widehat q-x\cdot\widehat p)\right]
\]

and satisfy

\[
T_uT_v=e^{-2\pi iE(u,v)}T_vT_u
=e^{-\pi iE(u,v)}T_{u+v}.
\]

This is equation (2.6). The factor `sqrt(2*pi)` is explicitly noted by the
authors as a nonstandard but convenient normalization.

A compatible complex structure `I` satisfies

\[
E(Iu,Iv)=E(u,v),\qquad E(Iu,u)>0\quad(u\ne0).
\]

The corresponding Hermitian form is

\[
H(z,w)=E(Iz,w)+iE(z,w).
\]

Hence the Euclidean metric used for lengths is

\[
g(z,w)=\operatorname{Re}H(z,w)=E(Iz,w),
\qquad |v|^2=H(v,v).
\]

## Lattices and polarization type

For a full lattice `Lambda` in `V`, the symplectic dual is

\[
\Lambda^\perp
=\{\mu\in V:E(\mu,\Lambda)\subset\mathbb Z\}.
\]

The lattice is symplectically integral when

\[
\Lambda\subseteq\Lambda^\perp.
\]

In a symplectic lattice basis, Frobenius normal form gives

\[
[E]_{\Lambda}=
\begin{pmatrix}0&D\\-D&0\end{pmatrix},
\qquad
D=\operatorname{diag}(d_1,\ldots,d_n),
\]

with positive integers

\[
d_1\mid d_2\mid\cdots\mid d_n.
\]

Definition 2.3 calls `(d_1,...,d_n)` the type of the lattice and of the GKP
code. The finite logical displacement group is

\[
K=\Lambda^\perp/\Lambda,
\]

and has order

\[
|K|=(d_1\cdots d_n)^2=|\det [E]_{\Lambda}|.
\]

In lattice coordinates `Lambda = Z^(2n)` with alternating matrix `A`,

\[
\Lambda^\perp=A^{-T}\mathbb Z^{2n}.
\]

This follows directly from the condition `c^T A m` integral for all integral
vectors `m`.

## Line bundles and polarization

With the compatible complex structure fixed, the quotient

\[
X=V/\Lambda
\]

is a complex abelian variety. A semicharacter `nu` and the Hermitian form `H`
define the Appell--Humbert line bundle `L(H,nu)`. Its Chern class is represented
by `E = Im(H)`. Thus the same alternating form that controls commuting GKP
translations is the integral polarization form.

The polarization kernel is identified with the logical displacement group:

\[
K(L)=\ker\phi_L\cong\Lambda^\perp/\Lambda.
\]

## Relative systole and multiplicity

Section 7.2 defines

\[
\ell(\Lambda)
=\min\{|\mu|:\mu\in\Lambda^\perp\setminus\Lambda\}
\]

and

\[
N_{\mathrm{lift}}(\Lambda)
=\#\{\mu\in\Lambda^\perp\setminus\Lambda:
|\mu|=\ell(\Lambda)\}.
\]

Theorem 7.6 states

\[
F_\sigma(\Lambda)\sim
\frac{2N_{\mathrm{lift}}(\Lambda)\sigma}
{\ell(\Lambda)\sqrt{2\pi}}
\exp\!\left[-\frac{\ell(\Lambda)^2}{8\sigma^2}\right].
\]

The Gaussian density is

\[
f_\sigma(v)=\frac{1}{(2\pi\sigma^2)^n}
\exp\!\left[-\frac{|v|^2}{2\sigma^2}\right]
\]

on the real `2n`-dimensional space.

Later, the paper geometrically describes `N` as the number of points of
`K(L)` at minimum torus distance. At torsion points, the number of minimum
lifts/geodesic directions can differ from the number of kernel classes.
Accordingly, the implementation will preserve both quantities:

- `N_lift`: the literal quantity in the lattice definition preceding
  Theorem 7.6;
- `N_class`: the number of nonzero kernel classes at minimum torus distance.

No equality between them will be assumed without an explicit check.

## Uniform type

Remark 7.8 states that if the lattice has type `(d,...,d)`, then

\[
\ell(\Lambda)=\frac{1}{d}\lambda_1(\Lambda),
\qquad
\lambda_1(\Lambda)=
\min\{|\lambda|:\lambda\in\Lambda\setminus\{0\}\}.
\]

This identity uses the same fixed norm on `V`: in a symplectic basis of
uniform type, `Lambda^perp = (1/d)Lambda`.

For a principally polarized pair `(X,L)`, the paper phrases the corresponding
geometric family as `(X,L^d)` of type `(d,...,d)`. Benchmark code should work
directly with the lattice, alternating form, and norm actually supplied; it
must not independently rescale the norm when changing `d`.

## Explicit software normalization labels

Every production relative-systole result carries one of two labels. These
labels describe how the supplied metric was selected; they do not change the
CVP calculation itself.

- `fixed_principal_metric`: keep the metric of a principal polarization fixed
  when replacing its alternating form `A_0` by `d*A_0`. This is the convention
  in Remark 7.8, and gives

  \[
  \ell_d^2=\lambda_1^2/d^2.
  \]

- `polarization_scaled_metric`: multiply both the alternating form and metric
  by `d`. This is the convention naturally produced when an integral
  Hermitian form itself specifies the polarization, and gives

  \[
  \ell_d^2=\lambda_1^2/d.
  \]

Thus, for the same principal metric and level `d`, the second squared systole
is `d` times the first. Values with different labels must not be numerically
ranked against each other.

The core `compute_relative_systole` API requires `metric_convention` as a
keyword argument. Its result exposes `normalization_record()`, containing the
label, complex dimension, polarization type, metric determinant, and `ell^2`.

## Benchmark normalization

Lattice comparisons must use a fixed covolume. For the elementary square and
hexagonal checks we use covolume one. Any later `D4` or Klein-quartic input must
state its basis, Gram matrix, covolume, and scaling explicitly.
