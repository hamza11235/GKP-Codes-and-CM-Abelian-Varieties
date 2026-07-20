# Data provenance

The release data separate four artifact classes:

- `*_protocol.json`: frozen experimental design, budgets, seeds, and claim boundary;
- `*_summary.json` / `*.csv`: compact result tables used by the consolidated report;
- `*_evaluations.json.gz` and control `*.json.gz`: complete raw ledgers;
- `*_audit.json`: independent high-precision recomputations.

Large raw ledgers are deterministically gzip-compressed in the public release.
The package loaders and notebooks transparently read either `name.json` or
`name.json.gz`. Generation scripts produce plain JSON for development.

The central machine-readable release summary is
`consolidated_results.json`. The radius-1.5 blind-search table is also stored
as `consolidated_headline_results.csv`.

## Metric convention

All nonuniform results use the `polarization_scaled_metric` convention. Raw
`ell` or `ell^2` values are compared only within a fixed polarization type.

## Exactness language

Integral Hermitian-form and finite-group computations are exact. Generic real
deformations use floating-point branch-and-bound CVP, with selected winners
independently audited at 60 or 70 decimal digits. The release does not label a
floating-point moduli endpoint as an arithmetically certified CM point.
