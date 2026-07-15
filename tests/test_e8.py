import unittest
from fractions import Fraction
from math import sqrt

from gkp_systole import (
    D4_PERIOD_MODEL,
    E8_COMPLEX_STRUCTURE,
    E8_GRAM,
    E8_PPAV_MODEL,
    E8_PRINCIPAL_ALTERNATING,
    MetricConvention,
    Polarization,
    compute_relative_systole,
    repeated_period_model,
    validate_e8_derivation,
    validate_ppav_data,
)


def transpose(matrix):
    return tuple(zip(*matrix))


def multiply(left, right):
    return tuple(
        tuple(
            sum(
                Fraction(left[i][k]) * Fraction(right[k][j])
                for k in range(len(right))
            )
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


class E8StructuralTests(unittest.TestCase):
    def test_exact_gram_matrix_and_principal_certificate(self):
        expected_gram = (
            (2, 0, 0, 0, 0, 0, 0, 1),
            (0, 2, -1, 0, 0, 0, 0, 0),
            (0, -1, 2, -1, 0, 0, 0, 0),
            (0, 0, -1, 2, -1, 0, 0, 0),
            (0, 0, 0, -1, 2, -1, 0, 0),
            (0, 0, 0, 0, -1, 2, -1, -1),
            (0, 0, 0, 0, 0, -1, 2, 0),
            (1, 0, 0, 0, 0, -1, 0, 2),
        )
        self.assertEqual(E8_GRAM, expected_gram)

        validate_e8_derivation()
        certificate = E8_PPAV_MODEL.validation_certificate()
        self.assertEqual(certificate.dimension, 4)
        self.assertEqual(certificate.polarization_type, (1, 1, 1, 1))
        self.assertEqual(certificate.physical_metric_determinant, Fraction(1))
        self.assertEqual(certificate.polarization.determinant, 1)
        self.assertTrue(certificate.certified)

    def test_complex_structure_and_riemann_form_are_integral(self):
        self.assertTrue(
            all(isinstance(value, int) for row in E8_COMPLEX_STRUCTURE for value in row)
        )
        self.assertTrue(
            all(
                isinstance(value, int)
                for row in E8_PRINCIPAL_ALTERNATING
                for value in row
            )
        )
        self.assertEqual(
            multiply(E8_GRAM, E8_COMPLEX_STRUCTURE),
            tuple(tuple(Fraction(value) for value in row) for row in E8_PRINCIPAL_ALTERNATING),
        )

    def test_qubit_polarization_has_type_two_four_times(self):
        qubit = Polarization(E8_PPAV_MODEL.uniform_alternating(2))
        self.assertEqual(qubit.type, (2, 2, 2, 2))
        self.assertEqual(qubit.kernel_order, 2**8)

    def test_ppav_certificate_survives_unimodular_basis_change(self):
        size = 8
        change = [[int(i == j) for j in range(size)] for i in range(size)]
        change[0][1] = 1
        inverse = [[int(i == j) for j in range(size)] for i in range(size)]
        inverse[0][1] = -1
        change = tuple(tuple(row) for row in change)
        inverse = tuple(tuple(row) for row in inverse)

        metric = multiply(multiply(transpose(change), E8_GRAM), change)
        alternating = multiply(
            multiply(transpose(change), E8_PRINCIPAL_ALTERNATING), change
        )
        alternating = tuple(tuple(int(value) for value in row) for row in alternating)
        complex_structure = multiply(
            multiply(inverse, E8_COMPLEX_STRUCTURE), change
        )

        certificate = validate_ppav_data(metric, complex_structure, alternating)
        self.assertEqual(certificate.polarization_type, (1, 1, 1, 1))
        self.assertEqual(certificate.physical_metric_determinant, Fraction(1))


class E8SystoleTests(unittest.TestCase):
    def test_uniform_svp_recovers_e8_root_data(self):
        result = E8_PPAV_MODEL.compute_qubit_systole()
        self.assertEqual(result.lambda1_squared, Fraction(2))
        self.assertEqual(result.squared_systole, Fraction(1, 2))
        self.assertEqual(result.class_multiplicity, 120)
        self.assertEqual(result.lift_multiplicity, 240)
        self.assertEqual(len(result.minimal_vectors), 240)
        self.assertTrue(result.certified)

    def test_full_kernel_path_matches_svp_shortcut(self):
        shortcut = E8_PPAV_MODEL.compute_qubit_systole()
        full = E8_PPAV_MODEL.compute_full_uniform_systole(2)

        self.assertEqual(full.squared_systole, shortcut.squared_systole)
        self.assertEqual(full.class_multiplicity, shortcut.class_multiplicity)
        self.assertEqual(full.lift_multiplicity, shortcut.lift_multiplicity)
        self.assertTrue(full.certified)

    def test_noncanonical_uniform_form_also_works_through_public_solver(self):
        result = compute_relative_systole(
            E8_PPAV_MODEL.uniform_alternating(2),
            E8_GRAM,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        self.assertEqual(result.squared_systole, Fraction(1, 2))

    def test_e8_beats_decomposable_d4_squared_baseline(self):
        e8 = E8_PPAV_MODEL.compute_qubit_systole()
        d4_squared = repeated_period_model(D4_PERIOD_MODEL, 2).compute_uniform_systole(2)
        self.assertGreater(float(e8.squared_systole), d4_squared.squared_systole)
        self.assertAlmostEqual(d4_squared.squared_systole, 1 / (2 * sqrt(2)))


if __name__ == "__main__":
    unittest.main()
