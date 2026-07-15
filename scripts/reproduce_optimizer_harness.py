#!/usr/bin/env python3
"""Exercise the common type-(1,5) optimizer objective end to end."""

from __future__ import annotations

from dataclasses import asdict
import json
from math import sqrt
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gkp_systole import (  # noqa: E402
    EvaluationMode,
    OptimizationRunConfig,
    OptimizerHarness,
    type15_compatible_metric_family,
)


def main() -> None:
    family = type15_compatible_metric_family()
    config = OptimizationRunConfig.symmetric_box(
        experiment_id="type15-harness-reproduction",
        optimizer_name="deterministic-random-baseline",
        seed=20260715,
        coordinate_dimension=family.coordinate_dimension,
        radius=0.25,
        evaluation_budget=40,
        verification_decimal_places=60,
        checkpoint_interval=8,
    )

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "type15.jsonl"
        harness = OptimizerHarness(family, config, checkpoint_path=checkpoint)
        origin = harness.evaluate((0.0,) * family.coordinate_dimension)

        random = np.random.default_rng(config.seed)
        coordinates = random.uniform(-0.05, 0.05, size=(24, family.coordinate_dimension))
        samples = harness.evaluate_many(coordinates, workers=2)
        verified = harness.evaluate(origin.coordinates, mode=EvaluationMode.VERIFY)
        harness.save_checkpoint()
        resumed = OptimizerHarness.from_checkpoint(family, checkpoint)

        assert family.coordinate_dimension == 6
        assert family.polarization.type == (1, 5)
        assert abs(origin.squared_systole - sqrt(2.0 / 5.0)) < 1e-12
        assert origin.class_multiplicity == origin.lift_multiplicity == 24
        assert origin.diagnostics.valid
        assert abs(float(verified.high_precision_squared_systole) - origin.squared_systole) < 1e-12
        assert resumed.summary().unique_evaluation_count == harness.summary().unique_evaluation_count

        output = {
            "family": family.name,
            "polarization_type": family.polarization.type,
            "coordinate_dimension": family.coordinate_dimension,
            "origin": {
                "ell_squared": origin.squared_systole,
                "exact": family.reference_exact_ell_squared,
                "classes": origin.class_multiplicity,
                "lifts": origin.lift_multiplicity,
                "diagnostics": asdict(origin.diagnostics),
            },
            "random_batch": {
                "count": len(samples),
                "best_ell_squared": max(sample.squared_systole for sample in samples),
            },
            "verification": verified.high_precision_squared_systole,
            "summary": asdict(harness.summary()),
            "checkpoint_resume_unique_evaluations": resumed.summary().unique_evaluation_count,
        }
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

