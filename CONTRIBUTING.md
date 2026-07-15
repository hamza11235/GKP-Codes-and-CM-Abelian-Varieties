# Contributing

Contributions that improve mathematical verification, reproducibility, or
coverage of polarized CM examples are welcome.

Before submitting a change:

1. run `python -m unittest discover -s tests -v`;
2. run `python scripts/reproduce_headline.py`;
3. label floating results separately from exact or interval-certified results;
4. document the polarization type and metric convention;
5. avoid global-optimality language unless accompanied by a proved upper bound.

New exact models should include tests for the Riemann conditions, polarization
type, determinant normalization, relative systole, both multiplicities, and any
claimed CM or isogeny certificate.
