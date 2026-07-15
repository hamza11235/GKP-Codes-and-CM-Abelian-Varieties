import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gkp_systole import (
    CoordinateOutOfBounds,
    EvaluationBudgetExceeded,
    EvaluationMode,
    OptimizationRunConfig,
    OptimizerHarness,
    type15_compatible_metric_family,
)


def _config(family, *, budget=20, radius=0.25, checkpoint_interval=100):
    return OptimizationRunConfig.symmetric_box(
        experiment_id="type15-harness-test",
        optimizer_name="test-driver",
        seed=20260715,
        coordinate_dimension=family.coordinate_dimension,
        radius=radius,
        evaluation_budget=budget,
        verification_decimal_places=40,
        checkpoint_interval=checkpoint_interval,
    )


class Type15HarnessTests(unittest.TestCase):
    def test_type15_chart_has_correct_dimension_type_and_origin(self):
        family = type15_compatible_metric_family()
        self.assertEqual(family.coordinate_dimension, 6)
        self.assertEqual(family.polarization.type, (1, 5))
        family.validate_coordinates((0.0,) * 6)
        self.assertAlmostEqual(
            family.evaluate((0.0,) * 6).squared_systole,
            math.sqrt(2.0 / 5.0),
            places=12,
        )

    def test_origin_records_active_classes_and_compatible_metric(self):
        family = type15_compatible_metric_family()
        harness = OptimizerHarness(family, _config(family))
        result = harness.evaluate((0.0,) * 6)
        self.assertAlmostEqual(result.squared_systole, math.sqrt(2.0 / 5.0), places=12)
        self.assertAlmostEqual(result.loss, -result.squared_systole)
        self.assertEqual(result.class_multiplicity, 24)
        self.assertEqual(result.lift_multiplicity, 24)
        self.assertEqual(len(result.active_classes), 24)
        self.assertEqual(len(result.active_lifts), 24)
        self.assertTrue(result.diagnostics.valid)
        self.assertGreater(result.diagnostics.minimum_eigenvalue, 0.0)

    def test_cache_and_trace_include_repeated_optimizer_requests(self):
        family = type15_compatible_metric_family()
        harness = OptimizerHarness(family, _config(family))
        first = harness.evaluate((0.0,) * 6)
        second = harness.evaluate((0.0,) * 6)
        self.assertIs(first, second)
        summary = harness.summary()
        self.assertEqual(summary.request_count, 2)
        self.assertEqual(summary.unique_evaluation_count, 1)
        self.assertEqual(summary.screening_evaluation_count, 1)
        self.assertEqual(summary.cache_hits, 1)
        self.assertFalse(harness.trace[0].cache_hit)
        self.assertTrue(harness.trace[1].cache_hit)

    def test_bounds_penalty_and_strict_evaluation(self):
        family = type15_compatible_metric_family()
        harness = OptimizerHarness(family, _config(family, radius=0.1))
        outside = (0.11, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertTrue(math.isinf(harness.objective(outside)))
        with self.assertRaises(CoordinateOutOfBounds):
            harness.evaluate(outside)

    def test_screening_budget_is_enforced(self):
        family = type15_compatible_metric_family()
        harness = OptimizerHarness(family, _config(family, budget=1))
        harness.evaluate((0.0,) * 6)
        with self.assertRaises(EvaluationBudgetExceeded):
            harness.evaluate((0.01, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_parallel_batch_preserves_input_order_and_unique_sequences(self):
        family = type15_compatible_metric_family()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parallel.jsonl"
            harness = OptimizerHarness(
                family,
                _config(family, checkpoint_interval=2),
                checkpoint_path=checkpoint,
            )
            rows = (
                (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, -0.01, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.01, 0.0, 0.0, 0.0),
            )
            results = harness.evaluate_many(rows, workers=2)
            harness.save_checkpoint()
            self.assertEqual(tuple(result.coordinates for result in results), rows)
            self.assertEqual(len({result.sequence for result in results}), len(rows))
            self.assertTrue(all(result.diagnostics.valid for result in results))
            self.assertEqual(
                OptimizerHarness.from_checkpoint(family, checkpoint)
                .summary()
                .unique_evaluation_count,
                len(rows),
            )

    def test_high_precision_verification_matches_screening(self):
        family = type15_compatible_metric_family()
        harness = OptimizerHarness(family, _config(family))
        coordinates = (0.01, -0.006, 0.004, 0.003, -0.002, 0.001)
        screen = harness.evaluate(coordinates)
        verified = harness.evaluate(coordinates, mode=EvaluationMode.VERIFY)
        self.assertIsNotNone(verified.high_precision_squared_systole)
        self.assertAlmostEqual(
            float(verified.high_precision_squared_systole),
            screen.squared_systole,
            places=10,
        )
        self.assertEqual(verified.active_classes, screen.active_classes)

    def test_checkpoint_roundtrip_restores_cache_and_configuration(self):
        family = type15_compatible_metric_family()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            harness = OptimizerHarness(
                family,
                _config(family, checkpoint_interval=2),
                checkpoint_path=path,
            )
            origin = harness.evaluate((0.0,) * 6)
            harness.evaluate((0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
            self.assertTrue(path.exists())

            resumed = OptimizerHarness.from_checkpoint(family, path)
            self.assertEqual(resumed.config, harness.config)
            self.assertEqual(resumed.summary().unique_evaluation_count, 2)
            self.assertEqual(resumed.summary().request_count, 2)
            cached = resumed.evaluate((0.0,) * 6)
            self.assertEqual(cached.squared_systole, origin.squared_systole)
            self.assertEqual(resumed.summary().cache_hits, 1)


class RunConfigTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            OptimizationRunConfig(
                experiment_id="bad",
                optimizer_name="test",
                seed=1,
                bounds=((1.0, -1.0),),
                evaluation_budget=1,
            )
        with self.assertRaises(ValueError):
            OptimizationRunConfig.symmetric_box(
                experiment_id="bad",
                optimizer_name="test",
                seed=1,
                coordinate_dimension=6,
                radius=0.0,
                evaluation_budget=1,
            )


if __name__ == "__main__":
    unittest.main()
