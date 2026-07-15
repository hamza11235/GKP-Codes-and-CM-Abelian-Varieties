import unittest
from fractions import Fraction

from gkp_systole import (
    Metric,
    MetricConvention,
    canonical_alternating,
    compute_relative_systole,
    uniform_metric,
)


class MetricConventionTests(unittest.TestCase):
    def test_fixed_principal_metric_does_not_rescale(self):
        principal = ((2, 1), (1, 2))
        result = uniform_metric(
            principal,
            3,
            MetricConvention.FIXED_PRINCIPAL,
        )
        self.assertEqual(result.matrix, Metric(principal).matrix)
        self.assertEqual(result.determinant, Fraction(3))

    def test_polarization_scaled_metric_multiplies_by_level(self):
        principal = ((2, 1), (1, 2))
        result = uniform_metric(
            principal,
            3,
            MetricConvention.POLARIZATION_SCALED,
        )
        self.assertEqual(result.matrix, ((6, 3), (3, 6)))
        self.assertEqual(result.determinant, Fraction(27))

    def test_string_conventions_are_validated(self):
        result = uniform_metric(
            ((1, 0), (0, 1)),
            2,
            "fixed_principal_metric",
        )
        self.assertEqual(result.determinant, Fraction(1))
        with self.assertRaises(ValueError):
            uniform_metric(((1, 0), (0, 1)), 2, "ambiguous")

    def test_metric_float_determinant(self):
        self.assertAlmostEqual(Metric(((2.0, 0.5), (0.5, 1.0))).determinant, 1.75)

    def test_uniform_relative_systole_scaling_laws(self):
        principal = ((1, 0), (0, 1))
        for level in (2, 3, 4):
            alternating = canonical_alternating((level,))
            fixed = compute_relative_systole(
                alternating,
                uniform_metric(
                    principal,
                    level,
                    MetricConvention.FIXED_PRINCIPAL,
                ),
                metric_convention=MetricConvention.FIXED_PRINCIPAL,
            )
            scaled = compute_relative_systole(
                alternating,
                uniform_metric(
                    principal,
                    level,
                    MetricConvention.POLARIZATION_SCALED,
                ),
                metric_convention=MetricConvention.POLARIZATION_SCALED,
            )
            with self.subTest(level=level):
                self.assertEqual(fixed.squared_systole, Fraction(1, level * level))
                self.assertEqual(scaled.squared_systole, Fraction(1, level))


if __name__ == "__main__":
    unittest.main()
