from __future__ import annotations

from pathlib import Path

import numpy as np

from gkp_passive_cliffords import (
    logical_gate_defect_model,
    load_json_artifact,
    measure_gate_robustness,
    phase9_champion_rows,
)
from gkp_passive_cliffords.preregistered_controls import form_from_population_row


PROJECT = Path(__file__).resolve().parents[1]


def test_cm_baselines_have_zero_defect_and_full_retention():
    population = load_json_artifact(PROJECT / "data/phase5_cm_population.json")
    for row in phase9_champion_rows(population):
        form = form_from_population_row(row)
        model = logical_gate_defect_model(form)
        result = measure_gate_robustness(model, model.baseline_metric)
        assert model.logical_image_order == int(row["logical_image_order"])
        assert max(result.action_defects) < 2e-10
        assert abs(result.retention_score - 1.0) < 2e-12
        assert result.epsilon_count(1e-8) == model.logical_image_order
        assert result.enhanced_epsilon_count(1e-8) == (
            model.logical_image_order - len(model.generic_action_indices)
        )


def test_identity_logical_action_has_zero_defect_after_deformation():
    population = load_json_artifact(PROJECT / "data/phase5_cm_population.json")
    row = phase9_champion_rows(population)[0]
    form = form_from_population_row(row)
    model = logical_gate_defect_model(form)
    deformed = np.array(model.baseline_metric, copy=True)
    deformed[0, 0] *= 1.0001
    result = measure_gate_robustness(model, deformed)
    assert result.action_defects[model.identity_action_index] < 2e-12
