import unittest
from fractions import Fraction
from itertools import product
from math import sqrt

from gkp_systole import (
    D4_PERIOD_MODEL,
    KLEIN_QUARTIC_PERIOD_MODEL,
    Metric,
    MetricConvention,
    canonical_alternating,
    compute_relative_systole,
    compute_uniform_relative_systole,
    initial_benchmarks,
    shortest_lattice_vectors,
)


class ShortestVectorTests(unittest.TestCase):
    def test_exact_square_minimal_vectors(self):
        result = shortest_lattice_vectors(((1, 0), (0, 1)))
        self.assertEqual(result.squared_length, Fraction(1))
        self.assertEqual(
            set(result.vectors),
            {(-1, 0), (0, -1), (0, 1), (1, 0)},
        )
        self.assertEqual(result.multiplicity, 4)
        self.assertTrue(result.certified)

    def test_exact_solver_matches_direct_enumeration(self):
        metric = Metric(((4, 1, 0, 0), (1, 3, 1, 0), (0, 1, 3, 1), (0, 0, 1, 2)))
        result = shortest_lattice_vectors(metric)
        direct_values = {
            vector: metric.quadratic(vector)
            for vector in product(range(-2, 3), repeat=4)
            if any(vector)
        }
        direct_minimum = min(direct_values.values())
        direct_vectors = {
            vector for vector, value in direct_values.items() if value == direct_minimum
        }
        self.assertEqual(result.squared_length, direct_minimum)
        self.assertEqual(set(result.vectors), direct_vectors)

    def test_float_hexagonal_minimal_vectors(self):
        metric = initial_benchmarks[1].metric
        result = shortest_lattice_vectors(metric)
        self.assertAlmostEqual(result.squared_length, 2 / sqrt(3), places=12)
        self.assertEqual(result.multiplicity, 6)
        self.assertFalse(result.certified)


class UniformRelativeSystoleTests(unittest.TestCase):
    def _assert_matches_full_kernel(self, principal_metric, level, convention):
        shortcut = compute_uniform_relative_systole(
            principal_metric,
            level,
            metric_convention=convention,
        )
        full = compute_relative_systole(
            canonical_alternating((level,) * shortcut.dimension),
            shortcut.metric,
            metric_convention=convention,
        )
        if shortcut.certified and full.certified:
            self.assertEqual(shortcut.squared_systole, full.squared_systole)
        else:
            self.assertAlmostEqual(
                float(shortcut.squared_systole),
                float(full.squared_systole),
                places=11,
            )
        self.assertEqual(shortcut.class_multiplicity, full.class_multiplicity)
        self.assertEqual(shortcut.lift_multiplicity, full.lift_multiplicity)
        return shortcut

    def test_square_fixed_and_scaled_conventions(self):
        square = ((1, 0), (0, 1))
        fixed = self._assert_matches_full_kernel(
            square,
            2,
            MetricConvention.FIXED_PRINCIPAL,
        )
        scaled = self._assert_matches_full_kernel(
            square,
            2,
            MetricConvention.POLARIZATION_SCALED,
        )
        self.assertEqual(fixed.lambda1_squared, Fraction(1))
        self.assertEqual(fixed.squared_systole, Fraction(1, 4))
        self.assertEqual(scaled.squared_systole, Fraction(1, 2))
        self.assertEqual(fixed.class_multiplicity, 2)
        self.assertEqual(fixed.lift_multiplicity, 4)

    def test_hexagonal_shortcut_matches_full_kernel(self):
        result = self._assert_matches_full_kernel(
            initial_benchmarks[1].metric,
            2,
            MetricConvention.FIXED_PRINCIPAL,
        )
        self.assertAlmostEqual(result.squared_systole, 1 / (2 * sqrt(3)), places=12)
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 6)

    def test_d4_period_core_shortcut_matches_full_kernel(self):
        shortcut = self._assert_matches_full_kernel(
            D4_PERIOD_MODEL.metric_core,
            2,
            MetricConvention.FIXED_PRINCIPAL,
        )
        full = D4_PERIOD_MODEL.compute_qubit_systole().core_result
        self.assertEqual(shortcut.lambda1_squared, Fraction(2))
        self.assertEqual(shortcut.squared_systole, full.squared_systole)
        self.assertEqual(shortcut.class_multiplicity, 12)
        self.assertEqual(shortcut.lift_multiplicity, 24)

    def test_klein_period_core_shortcut_matches_full_kernel(self):
        shortcut = self._assert_matches_full_kernel(
            KLEIN_QUARTIC_PERIOD_MODEL.metric_core,
            2,
            MetricConvention.FIXED_PRINCIPAL,
        )
        full = KLEIN_QUARTIC_PERIOD_MODEL.compute_qubit_systole().core_result
        self.assertEqual(shortcut.lambda1_squared, Fraction(8))
        self.assertEqual(shortcut.squared_systole, full.squared_systole)
        self.assertEqual(shortcut.class_multiplicity, 21)
        self.assertEqual(shortcut.lift_multiplicity, 42)

    def test_metadata_records_selected_physical_metric(self):
        result = compute_uniform_relative_systole(
            ((1, 0), (0, 1)),
            3,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        self.assertEqual(
            result.normalization_record(),
            {
                "metric_convention": "polarization_scaled_metric",
                "dimension_g": 1,
                "polarization_type": (3,),
                "metric_determinant": Fraction(9),
                "ell_squared": Fraction(1, 3),
                "lambda1_squared": Fraction(1),
            },
        )

    def test_rejects_invalid_level_and_odd_phase_space_dimension(self):
        with self.assertRaises(ValueError):
            compute_uniform_relative_systole(((1, 0), (0, 1)), 1)
        with self.assertRaises(ValueError):
            compute_uniform_relative_systole(((1,),), 2)


if __name__ == "__main__":
    unittest.main()
