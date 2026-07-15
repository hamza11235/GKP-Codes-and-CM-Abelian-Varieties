# Reproducibility guide

## Supported environment

- Python 3.10 or newer
- NumPy 1.24 or newer
- mpmath 1.3 or newer

The exact core uses only the Python standard library. NumPy is used for numerical
moduli searches and mpmath for high-precision and interval calculations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
```

## Exact regression suite

```bash
python -m unittest discover -s tests -v
```

## Headline results

```bash
python scripts/reproduce_headline.py
```

This recomputes the exact `D4`, Klein-quartic, `E8`, type-`(1,5)`, and
type-`(1,1,2)` certificates.

## Curated notebooks

1. `01_quickstart_kernel.ipynb`: polarization and kernel enumeration.
2. `02_verified_benchmarks.ipynb`: `D4` and Klein period models.
3. `03_e8_benchmark.ipynb`: exact `E8` PPAV and uniform optimum.
4. `04_generic_real_controls.ipynb`: 10,000 generic-real control samples.
5. `05_g2_cm_survey.ipynb`: consolidated dimension-two survey.
6. `06_type15_exact.ipynb`: exact type-`(1,5)` reconstruction.
7. `07_g3_cm_survey.ipynb`: ternary-Hermitian survey and controls.
8. `08_type112_exact.ipynb`: exact type-`(1,1,2)` reconstruction.

The notebooks contain stored outputs but should also run from top to bottom from
the repository root. Their import preambles add the local `src` directory.

## Machine-readable data

The `data` directory contains JSON and CSV ledgers. Every entry records:

- polarization type and dimension;
- exact and decimal values;
- multiplicities;
- metric convention;
- arithmetic certification status;
- search scope and claim strength.

## Exact versus numerical results

An exact result must pass exact polarized-abelian validation and exhaustive exact
CVP/SVP enumeration. High-precision floating agreement is a diagnostic rather
than a proof. The only exception is an explicitly labelled interval certificate,
where outward rounding and exhaustive interval enumeration provide rigor.
