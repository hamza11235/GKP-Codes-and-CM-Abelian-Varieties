# Exact polarized-abelian validation

Before a Gram matrix is used in a systole scan, it must describe a metric that
is compatible with a complex structure and an integral Riemann form.  The
validator supports the algebraic representation used by the verified models:

$$
G=G_{\rm core}/\sqrt r,
\qquad
J=J_{\rm num}/\sqrt r.
$$

`validate_polarized_abelian_data(...)` certifies, using exact rational
arithmetic, that

$$
J_{\rm num}^2=-rI,
\qquad
J_{\rm num}^{T}G_{\rm core}J_{\rm num}=rG_{\rm core},
$$

and that

$$
A=G_{\rm core}J_{\rm num}/r
$$

is an integral, nondegenerate alternating form of the requested polarization
type.  It also checks the covolume identity

$$
\det(G_{\rm core})/r^g=|\det A|=(d_1\cdots d_g)^2.
$$

Use `validate_ppav_data(...)` when the polarization must be principal.  This
wrapper additionally requires type `(1,...,1)`.

Both functions reject floating-point inputs: they are certificate-producing
entry points, not tolerance-based numerical screeners.  Approximate period
matrices should first be reconstructed as exact algebraic data or handled by a
future interval-arithmetic validator.

Successful validation returns a `PPAVValidationResult`, including the derived
polarization, its type, the physical determinant, the scale radicand, and the
list of certified identities.  The existing period, Gaussian Hermitian,
Eisenstein Hermitian, and CM-product models expose this through
`validation_certificate()`.

See `notebooks/02_verified_benchmarks.ipynb` and `tests/test_ppav.py` for known
benchmarks and deliberately incompatible inputs.
