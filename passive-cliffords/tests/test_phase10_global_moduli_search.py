from __future__ import annotations

import numpy as np

from gkp_passive_cliffords import (
    canonical_product_family,
    intrinsic_coordinate_radius,
    rms_affine_invariant_distance,
    run_phase10_blind_search,
    sobol_ball,
)


def test_canonical_family_and_radius_conversion() -> None:
    family = canonical_product_family((1, 1, 2))
    assert family.coordinate_dimension == 12
    radius = 0.4
    coordinate_radius = intrinsic_coordinate_radius(3, radius)
    coordinates = np.zeros(12)
    coordinates[0] = coordinate_radius
    metric = family.metric(coordinates)
    distance = rms_affine_invariant_distance(family.reference_metric, metric)
    assert abs(distance - radius) < 2e-12
    family.validate_coordinates(coordinates)


def test_sobol_ball_is_deterministic_and_bounded() -> None:
    left = sobol_ball(6, 16, 1234, 0.75)
    right = sobol_ball(6, 16, 1234, 0.75)
    assert np.array_equal(left, right)
    assert float(np.max(np.linalg.norm(left, axis=1))) <= 0.75


def test_three_optimizers_respect_equal_tiny_budget() -> None:
    summaries, evaluations = run_phase10_blind_search(
        (1, 3),
        0.25,
        method_budget=8,
        bo_initial=4,
        bo_steps=4,
        acquisition_pool_size=32,
        cma_restarts=2,
    )
    assert {item.method for item in summaries} == {
        "sobol",
        "cma_es",
        "bayesian_ucb",
    }
    assert all(item.budget == 8 for item in summaries)
    assert len(evaluations) == 24
    assert all(item.achieved_rms_distance <= 0.25 + 2e-10 for item in evaluations)
    assert all(np.isfinite(item.ell_squared) and item.ell_squared > 0 for item in evaluations)
