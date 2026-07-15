# Common optimizer harness

The first stage of the adversarial search program is a single CM-blind
objective interface for all optimization algorithms. The initial target is

$$
g=2,\qquad D=(1,5),\qquad \ell^2_{\mathrm{record}}=\sqrt{2/5}.
$$

## Compatible coordinates

The harness uses the existing full six-dimensional Cartan chart. If $A$ is the
fixed type-$(1,5)$ polarization and $G_0$ is the chart center, then

$$
G(x)=\exp(H(x))^T G_0\exp(H(x)),
$$

where the six basis generators satisfy

$$
H^T A+AH=0,
\qquad
H^T G_0=G_0H.
$$

Consequently every generated metric has the fixed polarization covolume and a
compatible complex structure. The harness nevertheless checks positivity,
symmetry, determinant, and $J^2=-I$ numerically at every point.

## Optimizer contract

`OptimizerHarness.objective(x)` returns

$$
-\ell^2(G(x)),
$$

so ordinary minimizers can use it directly. Points outside the declared box
receive an infinite penalty. Detailed evaluations additionally retain:

- the active logical classes and all shortest lifts;
- class and lift multiplicities;
- compatibility residuals;
- the complete coordinate vector and objective value;
- whether the request was served from cache;
- optional high-precision CVP verification.

Every run freezes the chart, bounds, seed, optimizer label, screening budget,
precision, and checkpoint interval in `OptimizationRunConfig`.

## Caching, batching, and checkpoints

Repeated points are served from an exact floating-coordinate cache while every
optimizer request is retained in the in-memory trace. Batches can be evaluated
with a deterministic input order and multiple worker threads. A JSON-lines
checkpoint stores the frozen experiment metadata and every unique evaluation;
`OptimizerHarness.from_checkpoint` resumes the cache and remaining budget.

The checkpoint is intentionally algorithm-independent. CMA-ES, differential
evolution, SHGO, DIRECT, Sobol screening, and later Bayesian methods will all
use the same records and numerical objective.

## Fast and verification modes

- **Screen:** double-precision compatible metric plus exhaustive floating CVP.
- **Verify:** the same active-set metadata plus an independent 60--100 digit
  `mpmath` reconstruction and exhaustive CVP calculation.

High precision is a numerical consistency check. Exact reconstruction and CM
classification remain separate downstream stages and are never consulted by
the optimization objective.

## Reproduction

```bash
python scripts/reproduce_optimizer_harness.py
python -m unittest tests.test_optimizer_harness -v
```

The corresponding executable walkthrough is
[`notebooks/09_optimizer_harness.ipynb`](../notebooks/09_optimizer_harness.ipynb).

