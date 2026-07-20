# Phase 5: bounded CM population survey

Phase 5 removes the strongest selection bias in the earlier gate calculation:
it evaluates every retained CM candidate in the pre-existing bounded
Hermitian searches, rather than only the distance record holders.

For each candidate the exact pipeline records

$$
\ell^2,\quad N_{\rm class},\quad N_{\rm lift},\quad
|\mathrm{Aut}_0(X,L)|,\quad
|\mathrm{im}(\mathrm{Aut}_0(X,L)\to\mathrm{Sp}(K(L)))|.
$$

The binary population uses all imaginary-quadratic order discriminants
`3 <= |Delta| <= 160` and diagonal bound 16.  The ternary population uses
`3 <= |Delta| <= 40`, diagonal bound 5, and off-diagonal order-basis
coefficients in `[-1,1]`.

These are bounded candidate populations under the repository's documented
elementary reductions.  Binary reduction includes shears, mode exchange, and
units.  Ternary reduction includes mode permutations and diagonal units.  The
lists are not complete classifications under the full Hermitian isometry
groups, so isometric multiplicities may remain, especially in rank three.

## Population results

| type | candidates | enhanced passive image | fraction | top-quartile fraction | mean ell^2, enhanced | mean ell^2, minimal | corr(ell^2, log image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(1,3)` | 876 | 43 | 0.0491 | 0.0868 | 0.5340 | 0.3547 | 0.2218 |
| `(1,5)` | 915 | 38 | 0.0415 | 0.1135 | 0.3859 | 0.2495 | 0.1994 |
| `(1,1,2)` | 1,051 | 131 | 0.1246 | 0.2281 | 0.6562 | 0.4587 | 0.3270 |
| `(1,1,3)` | 1,070 | 106 | 0.0991 | 0.2276 | 0.5861 | 0.3610 | 0.4423 |
| `(1,2,2)` | 253 | 203 | 0.8024 | 0.9062 | 0.4019 | 0.3011 | 0.6632 |

"Enhanced" means larger than the logical image of the unavoidable generic
origin-fixing group `{+I,-I}`.  That baseline image has order one for
exponent-two kernels and order two for the odd-prime types in this table.

Three features are consistent across all five fixed-`D` populations:

1. the enhanced-image subset has larger mean `ell^2`;
2. enhanced images are enriched in the top distance quartile;
3. the correlation between `ell^2` and the logarithm of image order is
   positive.

This is the first dataset-wide evidence in the project that distance and
passive symmetry are positively associated within the bounded CM
construction.  It does not imply that every CM point is good, and it is not a
CM-versus-non-CM population average.

## Logical image distributions

| type | image-order histogram |
|---|---|
| `(1,3)` | `2:833, 4:16, 6:26, 24:1` |
| `(1,5)` | `2:877, 4:25, 6:12, 8:1` |
| `(1,1,2)` | `1:920, 2:84, 3:37, 6:10` |
| `(1,1,3)` | `2:964, 4:37, 6:61, 24:8` |
| `(1,2,2)` | `1:50, 2:144, 3:1, 4:4, 6:29, 8:10, 12:3, 18:10, 48:2` |

The type-`(1,3)` bounded distance record is also the unique candidate with
the complete order-24 logical image.  For type `(1,1,3)`, every bounded
distance maximizer in the retained list has the complete order-24 image.  The
two type-`(1,2,2)` distance maximizers have the maximum observed image order
48.

Type `(1,5)` exhibits a genuine bounded-population tradeoff: its distance
record has image order four, while lower-distance Pareto candidates attain
orders six and eight.  The later exact reconstructed type-`(1,5)` record lies
outside this bounded Hermitian population and has image order 24.

## Claim boundary

The survey supports the statement:

> Under the stated bounded CM candidate protocol, enhanced passive symmetry
> is positively associated with larger relative systole in every tested
> polarization type.

It does not establish:

- a canonical average over CM points;
- a global optimum in any nonuniform moduli space;
- that CM alone forces extra polarized automorphisms;
- a population-level comparison with certified non-CM algebraic points.

The exact row-level data are stored in `data/phase5_cm_population.json` and
`data/phase5_cm_population.csv`; aggregate results are in
`data/phase5_cm_population_summary.json`.
