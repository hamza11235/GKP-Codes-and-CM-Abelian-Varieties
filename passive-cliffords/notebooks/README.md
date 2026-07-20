# Executed notebooks

| notebook | purpose |
|---|---|
| `01_phase1_automorphism_action_engine.ipynb` | one-mode exact automorphism and kernel-action benchmarks |
| `02_phase2_d4_klein_benchmarks.ipynb` | `D4`/Bolza and Klein-quartic higher-dimensional checks |
| `03_phase3_nonuniform_cm_actions.ipynb` | first nonuniform CM passive-action records |
| `04_phase4_matched_noncm_controls.ipynb` | initial matched generic-real controls |
| `05_phase5_cm_population_survey.ipynb` | 4,165-candidate bounded CM population |
| `06_phase6_preregistered_generic_controls.ipynb` | population-wide preregistered controls |
| `07_phase7_equal_distance_controls.ipynb` | exactly equal intrinsic-distance controls |
| `08_phase8_adversarial_local_search.ipynb` | Sobol and Bayesian adversarial local searches |
| `09_phase9_gate_robustness.ipynb` | exact and approximate passive-gate retention |
| `10_phase10_blind_global_search.ipynb` | blind bounded Sobol/CMA-ES/Bayesian search |

Every notebook is stored with executed outputs and terminates in explicit
assertions. Notebooks 5–10 accept either plain development JSON or compressed
release `.json.gz` ledgers.

Run from `passive-cliffords/` so local paths resolve consistently:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/10_phase10_blind_global_search.ipynb
```
