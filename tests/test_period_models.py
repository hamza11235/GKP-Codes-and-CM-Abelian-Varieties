import unittest
from fractions import Fraction
from math import sqrt

from gkp_systole import (
    D4_PERIOD_MODEL,
    KLEIN_QUARTIC_PERIOD_MODEL,
    MetricConvention,
    VERIFIED_PERIOD_MODELS,
    validate_d4_derivation,
)


class PeriodModelTests(unittest.TestCase):
    def test_all_models_pass_exact_riemann_checks(self):
        for model in VERIFIED_PERIOD_MODELS:
            with self.subTest(model=model.name):
                model.validate()

    def test_d4_period_and_metric(self):
        model = D4_PERIOD_MODEL
        validate_d4_derivation()
        self.assertEqual(model.scale_radicand, 2)
        self.assertEqual(
            model.metric_core,
            (
                (2, 0, 1, 1),
                (0, 2, 1, 1),
                (1, 1, 2, 1),
                (1, 1, 1, 2),
            ),
        )
        result = model.compute_qubit_systole()
        self.assertEqual(result.squared_systole_coefficient, Fraction(1, 2))
        self.assertEqual(result.squared_systole_expression, "(1/2)/sqrt(2)")
        self.assertAlmostEqual(result.squared_systole, 1 / (2 * sqrt(2)), places=12)
        self.assertEqual(result.class_multiplicity, 12)
        self.assertEqual(result.lift_multiplicity, 24)
        self.assertTrue(result.certified)
        self.assertEqual(
            result.core_result.metric_convention,
            MetricConvention.FIXED_PRINCIPAL,
        )

    def test_klein_period_and_metric(self):
        model = KLEIN_QUARTIC_PERIOD_MODEL
        self.assertEqual(model.scale_radicand, 28)
        result = model.compute_qubit_systole()
        self.assertEqual(result.squared_systole_coefficient, Fraction(2))
        self.assertEqual(result.squared_systole_expression, "1/sqrt(7)")
        self.assertAlmostEqual(result.squared_systole, 1 / sqrt(7), places=12)
        self.assertEqual(result.class_multiplicity, 21)
        self.assertEqual(result.lift_multiplicity, 42)
        self.assertTrue(result.certified)
        self.assertEqual(
            result.core_result.metric_convention,
            MetricConvention.FIXED_PRINCIPAL,
        )

    def test_numeric_period_matrices_are_symmetric_with_positive_imaginary_part(self):
        for model in VERIFIED_PERIOD_MODELS:
            with self.subTest(model=model.name):
                period = model.period_numeric
                self.assertEqual(period, tuple(zip(*period)))
                # Positive definiteness of Y_core and a positive scale imply the
                # same for Im(Omega); validate() checks Y_core exactly.
                self.assertGreater(model.metric_scale, 0)

    def test_uniform_svp_shortcut_matches_full_kernel_results(self):
        for model in VERIFIED_PERIOD_MODELS:
            with self.subTest(model=model.name):
                full = model.compute_qubit_systole()
                shortcut = model.compute_uniform_systole_via_svp(2)
                self.assertEqual(
                    shortcut.squared_systole_coefficient,
                    full.squared_systole_coefficient,
                )
                self.assertAlmostEqual(shortcut.squared_systole, full.squared_systole)
                self.assertEqual(shortcut.class_multiplicity, full.class_multiplicity)
                self.assertEqual(shortcut.lift_multiplicity, full.lift_multiplicity)
                self.assertTrue(shortcut.certified)


if __name__ == "__main__":
    unittest.main()
