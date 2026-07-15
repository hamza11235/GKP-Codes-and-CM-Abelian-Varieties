import unittest
from fractions import Fraction
from math import sqrt

from gkp_systole import (
    MetricConvention,
    Polarization,
    compute_relative_systole,
    initial_benchmarks,
)


def transpose(matrix):
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    )


def multiply(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


class RelativeSystoleTests(unittest.TestCase):
    def test_square_one_mode_qubit(self):
        square = initial_benchmarks[0]
        result = compute_relative_systole(
            square.alternating,
            square.metric,
            metric_convention=square.metric_convention,
        )
        self.assertEqual(result.squared_systole, Fraction(1, 4))
        self.assertEqual(result.class_multiplicity, 2)
        self.assertEqual(result.lift_multiplicity, 4)
        self.assertTrue(result.certified)

    def test_hexagonal_one_mode_qubit(self):
        hexagonal = initial_benchmarks[1]
        result = compute_relative_systole(
            hexagonal.alternating,
            hexagonal.metric,
            metric_convention=hexagonal.metric_convention,
        )
        self.assertAlmostEqual(
            float(result.squared_systole),
            1 / (2 * sqrt(3)),
            places=12,
        )
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 6)
        self.assertFalse(result.certified)

    def test_d4_lattice_metric(self):
        d4 = initial_benchmarks[3]
        result = compute_relative_systole(
            d4.alternating,
            d4.metric,
            metric_convention=d4.metric_convention,
        )
        self.assertEqual(result.squared_systole, Fraction(1, 2))
        self.assertEqual(result.class_multiplicity, 12)
        self.assertEqual(result.lift_multiplicity, 24)
        self.assertTrue(result.certified)

    def test_matches_reference_benchmarks(self):
        for benchmark in (initial_benchmarks[0], initial_benchmarks[1], initial_benchmarks[3]):
            with self.subTest(benchmark=benchmark.name):
                result = compute_relative_systole(
                    benchmark.alternating,
                    benchmark.metric,
                    metric_convention=benchmark.metric_convention,
                )
                self.assertAlmostEqual(
                    float(result.squared_systole),
                    benchmark.expected_relative_systole_squared,
                    places=12,
                )

    def test_full_result_is_invariant_under_integral_basis_change(self):
        square = initial_benchmarks[0]
        transform = ((1, 1), (0, 1))
        transformed_alternating = multiply(
            multiply(transpose(transform), square.alternating),
            transform,
        )
        transformed_metric = multiply(
            multiply(transpose(transform), square.metric),
            transform,
        )
        original = compute_relative_systole(
            square.alternating,
            square.metric,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        transformed = compute_relative_systole(
            transformed_alternating,
            transformed_metric,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        self.assertEqual(Polarization(transformed_alternating).type, (2,))
        self.assertEqual(transformed.squared_systole, original.squared_systole)
        self.assertEqual(transformed.class_multiplicity, original.class_multiplicity)
        self.assertEqual(transformed.lift_multiplicity, original.lift_multiplicity)

    def test_result_records_normalization_metadata(self):
        square = initial_benchmarks[0]
        result = compute_relative_systole(
            square.alternating,
            square.metric,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        self.assertEqual(result.metric_convention, MetricConvention.FIXED_PRINCIPAL)
        self.assertEqual(
            result.normalization_record(),
            {
                "metric_convention": "fixed_principal_metric",
                "dimension_g": 1,
                "polarization_type": (2,),
                "metric_determinant": Fraction(1),
                "ell_squared": Fraction(1, 4),
            },
        )

    def test_core_solver_requires_a_metric_convention(self):
        square = initial_benchmarks[0]
        with self.assertRaises(TypeError):
            compute_relative_systole(square.alternating, square.metric)


if __name__ == "__main__":
    unittest.main()
