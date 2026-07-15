# CM constructions and relative systoles of GKP codes

This repository provides certified computational tools and reproducible results
for the relative systole

$$
\ell_{X,L}=\min_{0\ne x\in K(L)} d_X(0,x)
$$

of polarized complex abelian varieties, interpreted as the shortest
syndrome-invisible nontrivial logical displacement of a lattice GKP code.

The project was motivated by Mayrand and Royer, *Complex abelian varieties and
quantum error correction: a mathematical framework for GKP codes*,
[arXiv:2605.28784](https://arxiv.org/abs/2605.28784).

## Scope and claim status

The repository contains three kinds of results:

1. **Known benchmarks**, recovered independently to validate conventions and
   algorithms.
2. **Exact constructions**, certified by rational or interval arithmetic.
3. **Bounded survey records**, which are best within a documented finite search
   and are not claims of global optimality.

The main new observation is that full compatible-metric searches in types
`(1,5)` and `(1,1,2)` produced stronger configurations that subsequently
reconstructed as exact CM varieties. This is evidence for an arithmetic pattern,
not a theorem that CM points are globally optimal.

## Headline exact results

All nonuniform values below use the `polarization_scaled_metric` convention.
Values from different polarization types should not be ranked against one
another.

| dimension | type | exact $\ell^2$ | shortest classes/lifts | claim |
|---:|---:|---:|---:|---|
| 2 | `(1,3)` | $\sqrt{2/3}$ | `8 / 24` | exact bounded CM record |
| 2 | `(1,5)` | $\sqrt{2/5}$ | `24 / 24` | exact reconstructed CM record |
| 3 | `(1,1,2)` | $2/\sqrt{3}$ | `3 / 36` | exact reconstructed CM record |
| 3 | `(1,1,3)` | $2/\sqrt{3}$ | `8 / 72` | exact bounded CM record |
| 3 | `(1,2,2)` | $1$ | `15 / 60` | exact bounded CM record |

The type-`(1,5)` surface is isogenous to `E_(i sqrt(10))^2` and has a scaled
`D4` Euclidean dual. The type-`(1,1,2)` threefold is isogenous to
`E_(i sqrt(3))^3`.

For the fixed-principal uniform type `(2,2,2,2)`, the exact `E8` PPAV benchmark
has

$$
\lambda_1^2=2,\qquad \ell^2=\frac{1}{2},
\qquad N_{\mathrm{class}}=120,\quad N_{\mathrm{lift}}=240.
$$

Via the eight-dimensional sphere-packing theorem, this is a global optimum for
that uniform fixed-principal problem.

See [docs/results.md](docs/results.md) for the complete claim ledger.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To run the notebooks:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

## Reproduce the certified headline results

```bash
python scripts/reproduce_headline.py
python -m unittest discover -s tests -v
```

The headline script independently validates the polarized models and recomputes
their exact SVP/CVP data. The current suite contains 150 tests.

## Repository layout

```text
src/gkp_systole/   Python package
tests/             Exact and numerical regression tests
scripts/           One-command result reproduction
notebooks/         Curated executable derivations
docs/              Methods, conventions, and exact certificates
data/              Machine-readable result ledgers
```

## Numerical methodology

For a polarization matrix $A$ and compatible Gram matrix $G$, the code computes

$$
\ell^2=
\min_{0\ne[c]\in A^{-T}\mathbb{Z}^{2g}/\mathbb{Z}^{2g}}
\min_{n\in\mathbb{Z}^{2g}}(c+n)^T G(c+n).
$$

Rational metrics use exact `Fraction` arithmetic and exact `LDL^T`
branch-and-bound. Floating metrics use Cholesky branch-and-bound and are marked
uncertified until reconstructed exactly or checked with interval arithmetic.

Candidate generation combines imaginary-quadratic Hermitian forms, explicit CM
period data, and full-dimensional compatible-metric searches using deterministic
Halton screening and derivative-free refinement.

See [docs/methods.md](docs/methods.md) and
[docs/reproducibility.md](docs/reproducibility.md).

The dedicated non-CM-oriented control methodology and its limitations are
documented in [docs/generic_real_controls.md](docs/generic_real_controls.md).

Bibliographic entries for the principal mathematical and GKP references are in
[references.bib](references.bib).

## Limitations

- The nonuniform records are not proven global maxima.
- The Hermitian-form surveys use explicit finite bounds and partial isometry
  reduction rather than a complete classification.
- Generic floating controls are overwhelmingly expected to be non-CM, but the
  project does not certify the endomorphism ring of every sampled point.
- Results with different metric conventions or polarization types are not
  directly comparable.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Until an
associated paper is available, please cite this repository and the foundational
Mayrand--Royer paper.

## License

Released under the MIT License. See [LICENSE](LICENSE).
