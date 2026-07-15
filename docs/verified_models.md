# Verified period models

This document records the explicit principally polarized models now used by
the calculator. In both cases the period lattice is written in a symplectic
basis as

$$
\Lambda=\mathbb Z^g+\Omega\mathbb Z^g,
\qquad \Omega=X+iY\in\mathfrak H_g.
$$

For a principal polarization the flat Riemann metric in the lattice basis
`(I, Omega)` is

$$
G_\Omega=
\begin{pmatrix}
Y^{-1}&Y^{-1}X\\
XY^{-1}&XY^{-1}X+Y
\end{pmatrix}.
$$

The code derives this matrix from the period data rather than entering it as an
independent benchmark.

## D4 principally polarized abelian surface

Starting with the standard `D4` root-lattice Gram matrix

$$
G_0=\begin{pmatrix}
2&-1&0&0\\
-1&2&-1&-1\\
0&-1&2&0\\
0&-1&0&2
\end{pmatrix},
$$

scale it by `1/sqrt(2)` to obtain covolume one. An integral unimodular
alternating form compatible with this metric is

$$
A_0=\begin{pmatrix}
0&-1&0&1\\
1&0&0&-1\\
0&0&0&1\\
-1&1&-1&0
\end{pmatrix}.
$$

The integral basis change

$$
T=\begin{pmatrix}
-1&-1&-1&-1\\
-1&-1&-1&0\\
-1&0&0&0\\
-1&0&-1&0
\end{pmatrix}
$$

satisfies

$$
T^TA_0T=J_2.
$$

In this symplectic basis the normalized period matrix is

$$
\boxed{
\Omega_{D_4}=
\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}
+\frac{i}{\sqrt2}I_2.
}
$$

The corresponding metric is

$$
G_{D_4}=\frac1{\sqrt2}
\begin{pmatrix}
2&0&1&1\\
0&2&1&1\\
1&1&2&1\\
1&1&1&2
\end{pmatrix}.
$$

The code verifies exactly that the principal alternating form is unimodular,
the metric has covolume one, and the induced complex structure squares to
`-Id`. For the type-`(2,2)` qubit code:

$$
\ell^2=\frac1{2\sqrt2},\qquad
N_{\rm class}=12,\qquad N_{\rm lift}=24.
$$

The symplecticity of `D4` is given in Conway--Sloane, “D4, E8, Leech and
certain other lattices are symplectic,” Appendix 2 to Buser--Sarnak,
*Inventiones Mathematicae* 117 (1994), 53--55.

## Klein-quartic Jacobian

Example 4.16 of Bochnak--Kucharz--Silhol gives the normalized period matrix

$$
\boxed{
\Omega_K=\frac12I_3+
\frac{i}{2\sqrt7}
\begin{pmatrix}
3&2&2\\2&3&2\\2&2&3
\end{pmatrix}.
}
$$

The corresponding metric factors as

$$
G_K=\frac1{2\sqrt7}G_{K,\mathrm{core}},
$$

where

$$
G_{K,\mathrm{core}}=
\begin{pmatrix}
20&-8&-8&10&-4&-4\\
-8&20&-8&-4&10&-4\\
-8&-8&20&-4&-4&10\\
10&-4&-4&8&0&0\\
-4&10&-4&0&8&0\\
-4&-4&10&0&0&8
\end{pmatrix}.
$$

Again, the Riemann conditions and covolume are checked exactly. For the
type-`(2,2,2)` qubit code:

$$
\ell^2=\frac1{\sqrt7},\qquad
N_{\rm class}=21,\qquad N_{\rm lift}=42.
$$

Source: J. Bochnak, W. Kucharz, and R. Silhol, “Morphisms, line bundles and
moduli spaces in real algebraic geometry,” *Publications Mathématiques de
l'IHÉS* 86 (1997), Example 4.16, pp. 61--62.
