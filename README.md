# GKP Codes and CM Abelian Varieties

This repository develops reproducible computational tools for polarized
abelian varieties interpreted as lattice GKP codes. It studies two connected
questions:

1. How can the GKP relative systole be computed and certified?
2. Do CM points show numerical evidence of simultaneously favorable GKP
   distance and passive logical Clifford symmetry?

The project was motivated by Mayrand and Royer,
*Complex abelian varieties and quantum error correction: a mathematical
framework for GKP codes*,
[arXiv:2605.28784](https://arxiv.org/abs/2605.28784).

## Main conclusion

> The computations provide **numerical evidence that CM points are enriched
> near extremal regions for GKP relative systole and passive Clifford
> symmetry**. They do not prove that CM points are local or global optimizers.

![Consolidated numerical evidence](passive-cliffords/figures/headline_numerical_evidence.png)

## The two invariants

For a polarized abelian variety `(X,L)`, the finite group

```text
K(L) = ker(phi_L)
```

is the logical Pauli group modulo phases. The GKP relative systole is

$$
\ell_{X,L}=\min_{0\ne x\in K(L)} d_X(0,x).
$$

It is the shortest syndrome-invisible nontrivial logical displacement. In the
small-isotropic-noise model, larger `ell` improves the leading logical-error
exponent.

The passive logical Clifford group is the finite image

$$
\mathrm{im}\!\left(\mathrm{Aut}_0(X,L)
\longrightarrow \mathrm{Sp}(K(L))\right).
$$

This is the part of the logical symplectic group induced by exact
polarization-preserving passive symmetries.

## Certified systole results

The core [`gkp_systole`](src/gkp_systole) package implements exact and
high-precision finite-kernel CVP calculations, period-matrix validation,
imaginary-quadratic and quartic-CM constructions, compatible-metric searches,
and exact reconstructions.

All nonuniform values below use the `polarization_scaled_metric` convention.
Values from different polarization types must not be ranked against one
another.

| dimension | type | exact `ell^2` | shortest classes/lifts | status |
|---:|---:|---:|---:|---|
| 2 | `(1,3)` | `sqrt(2/3)` | `8 / 24` | exact bounded CM record |
| 2 | `(1,5)` | `sqrt(2/5)` | `24 / 24` | exact reconstructed CM record |
| 3 | `(1,1,2)` | `2/sqrt(3)` | `3 / 36` | exact reconstructed CM record |
| 3 | `(1,1,3)` | `2/sqrt(3)` | `8 / 72` | exact bounded CM record |
| 3 | `(1,2,2)` | `1` | `15 / 60` | exact bounded CM record |

The type-`(1,5)` surface is isogenous to `E_(i sqrt(10))^2` and has a
scaled-`D4` Euclidean dual. The type-`(1,1,2)` threefold is isogenous to
`E_(i sqrt(3))^3`.

For the fixed-principal uniform type `(2,2,2,2)`, the exact `E8` PPAV
benchmark has

$$
\lambda_1^2=2,\qquad \ell^2=\frac12,
\qquad N_{\rm class}=120,\quad N_{\rm lift}=240.
$$

Via the eight-dimensional sphere-packing theorem, this is a global optimum for
that uniform fixed-principal problem. See [docs/results.md](docs/results.md)
for the complete certified-result ledger.

## Numerical CM-extremality study

The [`passive-cliffords`](passive-cliffords) workstream adds exact logical
kernel actions, bounded CM populations, matched generic controls, adversarial
search, passive-gate robustness, and blind bounded optimization.

| experiment | scale | result |
|---|---:|---|
| bounded CM population | 4,165 candidates | enhanced passive symmetry is associated with larger mean `ell^2` in every tested type |
| preregistered controls | 24,990 deformations | CM baselines have larger local mean `ell^2` in every tested type |
| equal-distance controls | 24,990 deformations | all ten type-radius mean changes and descriptive intervals lie below zero |
| adversarial local search | 2,400 evaluations | no searched deformation beats a tested Phase-5 CM champion |
| gate robustness | 3,200 evaluations | all nonzero generic deformations lose every exact CM-only logical action |
| blind bounded search | 5,760 evaluations | no blind endpoint reaches the strongest exact CM record known for its type |

At the largest blind-search radius:

| type | best blind `ell^2` | strongest known CM `ell^2` | ratio |
|---|---:|---:|---:|
| `(1,3)` | 0.730918 | 0.816497 | 89.5% |
| `(1,5)` | 0.470379 | 0.632456 | 74.4% |
| `(1,1,2)` | 0.869062 | 1.154701 | 75.3% |
| `(1,1,3)` | 0.777730 | 1.154701 | 67.4% |
| `(1,2,2)` | 0.735415 | 1.000000 | 73.5% |

The blind optimizer receives only `ell^2`, begins at the canonical product
metric determined by the polarization type, and does not load CM or gate
metadata until all 5,760 queries finish. All 60 method winners agree with
independent 70-digit CVP recomputation to within `2.3e-16` in `ell^2`.

Read the full synthesis and claim boundary in
[passive-cliffords/docs/consolidated_results.md](passive-cliffords/docs/consolidated_results.md).

## What is and is not claimed

The repository distinguishes:

1. **Known benchmarks**, independently recovered to validate conventions.
2. **Exact constructions**, certified by rational, algebraic, or interval
   arithmetic.
3. **Bounded numerical evidence**, reported with finite search bounds,
   deterministic protocols, and explicit limitations.

The nonuniform records are not proven global maxima. The CM populations are
bounded computational populations, the generic controls do not define a
canonical distribution on moduli, and finite-budget searches do not exhaust a
noncompact moduli space.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
python -m pip install -e "./passive-cliffords[analysis,dev]"
```

## Reproduce

Certified systole results:

```bash
python scripts/reproduce_headline.py
python -m unittest discover -s tests -v
```

Passive-Clifford and numerical-extremality checks:

```bash
cd passive-cliffords
python scripts/generate_consolidated_results.py
for phase in 1 2 3 4 5 6 7 8 9 10; do
    python "scripts/check_phase${phase}.py"
done
python scripts/check_release.py
```

The public release stores large raw ledgers as deterministic `.json.gz` files.
Loaders and notebooks transparently accept compressed release data or plain
development JSON.

## Repository layout

```text
src/gkp_systole/       certified systole package
tests/                 core exact/numerical regression tests
scripts/               headline reproduction
notebooks/             certified construction notebooks
docs/                  conventions, methods, and exact certificates
data/                  certified result ledgers
passive-cliffords/     CM distance/symmetry experiments and blind searches
```

Bibliographic entries are in [references.bib](references.bib), and citation
metadata is in [CITATION.cff](CITATION.cff).

## License

Released under the MIT License. See [LICENSE](LICENSE).
