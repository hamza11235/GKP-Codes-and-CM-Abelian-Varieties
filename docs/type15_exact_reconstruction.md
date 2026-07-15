# Exact type `(1,5)` reconstruction

The strongest Phase-7 numerical control can be reconstructed exactly from its
24 active logical-displacement classes.

## Equal-distance reconstruction

Let `v_1,...,v_24` be the rational nearest lifts identified by the numerical
CVP calculation. Imposing

$$
v_i^T G v_i=\ell^2
$$

for every `i` gives a rank-10 rational linear system in the ten symmetric Gram
entries and `ell^2`. Its nullspace is one-dimensional and determines

$$
\frac{G}{\ell^2}=
\begin{pmatrix}
45&-115/2&60&-115/2\\
-115/2&75&-80&75\\
60&-80&110&-95\\
-115/2&75&-95&85
\end{pmatrix}.
$$

Compatibility with the type-`(1,5)` alternating form fixes the scale:

$$
G=\frac{G_{\rm core}}{\sqrt{10}},\qquad
J=\frac{S}{\sqrt{10}},
$$

where

$$
G_{\rm core}=
\begin{pmatrix}
90&-115&120&-115\\
-115&150&-160&150\\
120&-160&220&-190\\
-115&150&-190&170
\end{pmatrix}
$$

and

$$
S=
\begin{pmatrix}
68&-88&96&-90\\
35&-44&42&-42\\
41&-53&68&-61\\
61&-80&104&-92
\end{pmatrix}.
$$

Exact validation gives

$$
S^2=-10I,\qquad G_{\rm core}S/10=A,\qquad
\det(G_{\rm core})/10^2=|\det A|=25.
$$

The exact CVP solver finds core squared systole `2` for every nonzero kernel
class. Therefore

$$
\boxed{\ell^2=\frac2{\sqrt{10}}=\sqrt{\frac25}},
\qquad N_{\rm class}=N_{\rm lift}=24.
$$

## Scaled `D4` dual

The Euclidean dual Gram matrix is

$$
G^{-1}=\frac{Q}{\sqrt{10}},
\qquad
Q=10G_{\rm core}^{-1}.
$$

An explicit unimodular matrix `U` satisfies

$$
U^T Q U=G_{D_4}.
$$

Thus the 24 shortest logical lifts are precisely the 24 roots of a scaled
`D4`. This explains the exact value and the equal-distance multiplicity. It
does not, by itself, prove global optimality for the relative-systole problem.

## CM certificate

The integral endomorphism `S` commutes with `J` and satisfies

$$
S^2+10I=0.
$$

Hence the endomorphism algebra contains `Q(sqrt(-10))`. An explicit integral
matrix `P` of determinant `47` gives

$$
P^{-1}SP=
\begin{pmatrix}
0&-10&0&0\\
1&0&0&0\\
0&0&0&-10\\
0&0&1&0
\end{pmatrix}.
$$

Therefore the abelian surface is rationally isogenous to

$$
E_{i\sqrt{10}}^2.
$$

The elliptic factor has CM by the order of discriminant `-40`. The rational
commutant of `S` has dimension eight and is isomorphic to a matrix algebra over
`Q(sqrt(-10))`, which contains a commutative semisimple algebra of dimension
four. This proves that the reconstructed surface is of CM type.

The full calculation is executable in
`notebooks/06_type15_exact.ipynb`.
