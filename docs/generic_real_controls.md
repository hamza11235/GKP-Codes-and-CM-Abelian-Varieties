# Generic real controls at fixed polarization type

This experiment compares the current CM winners with deformations designed to
leave their rational isogeny classes.

## Sampling construction

For the fixed alternating form `A`, use symplectic transvections

$$
S=I+t v v^T A,
\qquad S^TAS=A.
$$

Here every parameter has the form

$$
t=q\pi,
\qquad q\in\mathbb Q\setminus\{0\}.
$$

The irrational factor prevents `S` from being rational, unlike the exact
experiment in notebook 08. Dense rational coefficients approximate continuous
real sampling while remaining reproducible. The resulting metric is

$$
G'=S^TGS.
$$

CM points form a measure-zero subset of the moduli space, so generic real
samples are overwhelmingly expected to be non-CM. We do not claim a formal
endomorphism-ring certificate for every sampled point.

## Numerical checks

The initial screen uses double-precision Cholesky branch-and-bound. The best
sample from every type and sampling regime is then reconstructed from its
exact rational coefficients times `pi` and solved again using an independent
60-digit `mpmath` Cholesky branch-and-bound implementation.

For each type, the experiment contains:

- 2,000 local controls with `|t| <= 0.02`;
- 3,000 broader controls with `|t| <= 0.6`.

## Results

| Type | CM baseline `ell^2` | best local control | best broad control | controls beating CM |
|---|---:|---:|---:|---:|
| `(1,2)` | `1` | `0.9765886679992842` | `0.8940804101434524` | `0 / 5,000` |
| `(1,3)` | `4/(3 sqrt(3)) = 0.769800358919501` | `0.7544310245220359` | `0.6828805872009780` | `0 / 5,000` |

The 60-digit values agree with the screening values to at least fourteen
decimal places. None of the 10,000 generic-real controls beats the relevant
CM baseline.

This is meaningful numerical evidence that the two CM points are unusually
strong within their fixed types. It is not an exhaustive global search or a
proof that each individual control has trivial endomorphism ring.
