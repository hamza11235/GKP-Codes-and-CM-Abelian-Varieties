import unittest
from fractions import Fraction

from gkp_systole import (
    KernelElement,
    KernelGroup,
    Polarization,
    canonical_alternating,
    initial_benchmarks,
)
from gkp_systole.kernel import invert_integer_matrix, matvec


class RationalLinearAlgebraTests(unittest.TestCase):
    def test_exact_inverse(self):
        matrix = ((0, 2), (-2, 0))
        inverse = invert_integer_matrix(matrix)
        self.assertEqual(
            inverse,
            ((Fraction(0), Fraction(-1, 2)), (Fraction(1, 2), Fraction(0))),
        )
        for column in range(2):
            product = matvec(matrix, tuple(inverse[row][column] for row in range(2)))
            self.assertEqual(
                product,
                tuple(Fraction(int(row == column)) for row in range(2)),
            )


class KernelElementTests(unittest.TestCase):
    def test_coordinates_are_reduced_modulo_integers(self):
        element = KernelElement((Fraction(-1, 2), Fraction(5, 2)))
        self.assertEqual(element.coordinates, (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(element.order, 2)

    def test_addition_is_quotient_addition(self):
        element = KernelElement((Fraction(1, 2), Fraction(0)))
        self.assertTrue((element + element).is_zero)


class KernelGroupTests(unittest.TestCase):
    def test_one_mode_qubit_classes(self):
        group = KernelGroup.from_polarization(
            Polarization(canonical_alternating((2,)))
        )
        expected = {
            (Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2)),
        }
        self.assertEqual({element.coordinates for element in group.elements}, expected)
        self.assertEqual(group.order, 4)
        self.assertEqual(group.exponent, 2)

    def test_initial_benchmark_kernel_orders_and_exponents(self):
        for benchmark in initial_benchmarks:
            with self.subTest(benchmark=benchmark.name):
                group = KernelGroup.from_polarization(
                    Polarization(benchmark.alternating)
                )
                self.assertEqual(group.order, benchmark.expected_kernel_order)
                self.assertEqual(
                    group.exponent,
                    benchmark.polarization_type[-1],
                )
                self.assertEqual(len(group.nonzero_elements), group.order - 1)

    def test_all_elements_are_closed_under_addition(self):
        group = KernelGroup.from_polarization(
            Polarization(canonical_alternating((1, 2)))
        )
        element_set = set(group.elements)
        for left in group.elements:
            for right in group.elements:
                self.assertIn(left + right, element_set)

    def test_noncanonical_basis_has_same_group_order_and_exponent(self):
        alternating = (
            (0, 0, 1, 1),
            (0, 0, 0, 2),
            (-1, 0, 0, 0),
            (-1, -2, 0, 0),
        )
        polarization = Polarization(alternating)
        group = KernelGroup.from_polarization(polarization)
        self.assertEqual(polarization.type, (1, 2))
        self.assertEqual(group.order, 4)
        self.assertEqual(group.exponent, 2)


if __name__ == "__main__":
    unittest.main()
