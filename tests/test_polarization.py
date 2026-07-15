import unittest

from gkp_systole import Polarization, PolarizationError, canonical_alternating
from gkp_systole.polarization import determinant, smith_invariant_factors


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


class DeterminantTests(unittest.TestCase):
    def test_bareiss_determinant(self):
        self.assertEqual(determinant(((0, 2), (-2, 0))), 4)
        self.assertEqual(determinant(((1, 2), (3, 4))), -2)
        self.assertEqual(determinant(((0, 0), (0, 0))), 0)


class PolarizationTests(unittest.TestCase):
    def test_canonical_types(self):
        cases = {
            (2,): (4, (2, 2)),
            (1, 2): (4, (1, 1, 2, 2)),
            (2, 2): (16, (2, 2, 2, 2)),
            (2, 2, 2): (64, (2, 2, 2, 2, 2, 2)),
            (1, 2, 6): (144, (1, 1, 2, 2, 6, 6)),
        }
        for expected_type, (kernel_order, smith_factors) in cases.items():
            with self.subTest(expected_type=expected_type):
                polarization = Polarization(canonical_alternating(expected_type))
                self.assertEqual(polarization.type, expected_type)
                self.assertEqual(polarization.dimension, len(expected_type))
                self.assertEqual(polarization.smith_factors, smith_factors)
                self.assertEqual(polarization.kernel_order, kernel_order)
                self.assertTrue(polarization.verify_kernel_order())

    def test_type_is_invariant_under_unimodular_basis_change(self):
        alternating = canonical_alternating((1, 2))
        transform = (
            (1, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, -1, 1),
        )
        self.assertEqual(abs(determinant(transform)), 1)
        transformed = multiply(multiply(transpose(transform), alternating), transform)
        polarization = Polarization(transformed)
        self.assertEqual(polarization.type, (1, 2))
        self.assertEqual(polarization.kernel_order, 4)

    def test_rejects_non_square(self):
        with self.assertRaisesRegex(PolarizationError, "square"):
            Polarization(((0, 1, 2), (-1, 0, 3)))

    def test_rejects_odd_order(self):
        with self.assertRaisesRegex(PolarizationError, "even order"):
            Polarization(((0, 1, 0), (-1, 0, 1), (0, -1, 0)))

    def test_rejects_non_integral(self):
        with self.assertRaisesRegex(PolarizationError, "integer"):
            Polarization(((0, 0.5), (-0.5, 0)))

    def test_rejects_non_alternating(self):
        with self.assertRaisesRegex(PolarizationError, "alternating"):
            Polarization(((0, 1), (1, 0)))

    def test_rejects_singular(self):
        with self.assertRaisesRegex(PolarizationError, "nonsingular"):
            Polarization(((0, 0), (0, 0)))

    def test_smith_factors_for_noncanonical_matrix(self):
        matrix = ((0, 2), (-2, 0))
        self.assertEqual(smith_invariant_factors(matrix), (2, 2))


if __name__ == "__main__":
    unittest.main()
