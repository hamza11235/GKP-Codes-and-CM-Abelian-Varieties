# Relative-systole solver

The production entry point is

```python
compute_relative_systole(A, G, metric_convention=...)
```

where `A` is an integral nonsingular alternating matrix and `G` is a symmetric
positive-definite Gram matrix in the same lattice basis.

The required `metric_convention` keyword records whether `G` is a fixed
principal metric or was scaled together with the polarization. It is retained
in the returned normalization metadata; see `docs/conventions.md`.

The calculation is

\[
\ell^2=
\min_{0\ne[c]\in A^{-T}\mathbb Z^m/\mathbb Z^m}
\min_{n\in\mathbb Z^m}(c+n)^TG(c+n).
\]

## Exact metrics

When every entry of `G` is integral or rational, distances are evaluated using
`fractions.Fraction`. The solver computes an exact decomposition

\[
G=LDL^T
\]

and recursively enumerates every integer coordinate whose exact partial
quadratic contribution does not exceed the best complete candidate. No
floating-point pruning is used. Results from this backend are marked
`certified=True`.

## Floating metrics

Metrics containing floating-point entries use a Cholesky branch-and-bound
enumeration. The implementation recursively enumerates integer translates in
nearest-first order and prunes branches whose partial Cholesky norm already
exceeds the best complete candidate.

These results are marked `certified=False`: the algorithm is exhaustive up to
the configured floating tolerance, but it is not an interval-arithmetic proof.
Top survey candidates supplied numerically should eventually be recomputed with
algebraic or ball-arithmetic inputs.

## Multiplicities

The result records both:

- `class_multiplicity`: the number of kernel classes at distance `ell`;
- `lift_multiplicity`: the number of shortest vectors in
  `Lambda^perp \ Lambda`.

The second is the literal lattice multiplicity used immediately before Theorem
7.6 of Mayrand--Royer.

## Input normalization

The solver deliberately does not rescale `A` or `G`, and it does not yet infer
either matrix from a period matrix. This prevents hidden normalization changes.
When a candidate is generated from an abelian variety, a separate input layer
must place its polarization form and chosen noise metric in the same lattice
basis and document the normalization.
