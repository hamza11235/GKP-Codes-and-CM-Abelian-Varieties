from __future__ import annotations

from pathlib import Path

import numpy as np

from gkp_passive_cliffords import (
    compatible_tangent_model,
    fixed_radius_deformation,
    load_json_artifact,
    phase8_champion_rows,
    sobol_sphere,
)
from gkp_passive_cliffords.preregistered_controls import form_from_population_row


PROJECT = Path(__file__).resolve().parents[1]


def _champions():
    rows = load_json_artifact(PROJECT / "data" / "phase5_cm_population.json")
    return phase8_champion_rows(rows)


def _physical_metric(form):
    return np.asarray(form.metric_core, dtype=float) / np.sqrt(form.order.radicand)


def test_tangent_dimensions_and_orthonormality():
    champions = _champions()
    for dimension in (2, 3):
        row = next(row for row in champions if int(row["dimension_g"]) == dimension)
        form = form_from_population_row(row)
        model = compatible_tangent_model(form.alternating, _physical_metric(form))
        assert model.tangent_dimension == dimension * (dimension + 1)
        assert np.max(np.abs(model.tangent_gram - np.eye(model.tangent_dimension))) < 2e-10
        for generator in model.generators:
            residual = generator.T @ model.alternating + model.alternating @ generator
            assert np.max(np.abs(residual)) < 2e-10


def test_fixed_radius_deformation_preserves_geometry():
    row = _champions()[0]
    form = form_from_population_row(row)
    model = compatible_tangent_model(form.alternating, _physical_metric(form))
    direction = sobol_sphere(model.tangent_dimension, 1, seed=12345)[0]
    result = fixed_radius_deformation(model, direction, 0.02)
    assert abs(result.achieved_distance - 0.02) < 2e-11
    assert result.polarization_residual < 2e-10
    assert result.log_volume_residual < 2e-10
    assert np.min(np.linalg.eigvalsh(result.metric)) > 0


def test_sobol_sphere_is_deterministic_and_normalized():
    first = sobol_sphere(12, 32, seed=77)
    second = sobol_sphere(12, 32, seed=77)
    assert np.array_equal(first, second)
    assert np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)) < 1e-14
