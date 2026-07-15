import unittest
from fractions import Fraction
from math import sqrt

from gkp_systole import (
    D4_PERIOD_MODEL,
    KLEIN_QUARTIC_PERIOD_MODEL,
    ProductPPAVModel,
    block_diagonal,
    hexagonal_product_model,
    repeated_period_model,
    square_product_model,
)


class BlockDiagonalTests(unittest.TestCase):
    def test_exact_block_diagonal(self):
        self.assertEqual(
            block_diagonal((((1, 2), (3, 4)), ((5,),))),
            (
                (Fraction(1), Fraction(2), Fraction(0)),
                (Fraction(3), Fraction(4), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(5)),
            ),
        )

    def test_rejects_empty_and_nonsquare_blocks(self):
        with self.assertRaises(ValueError):
            block_diagonal(())
        with self.assertRaises(ValueError):
            block_diagonal((((1, 2),),))


class HigherDimensionalProductTests(unittest.TestCase):
    def test_square_products_have_known_dimension_generic_formula(self):
        for dimension in (1, 2, 4, 8):
            with self.subTest(dimension=dimension):
                model = square_product_model(dimension)
                certificate = model.validation_certificate()
                result = model.compute_uniform_systole(2)

                self.assertEqual(model.dimension, dimension)
                self.assertEqual(certificate.polarization_type, (1,) * dimension)
                self.assertEqual(certificate.physical_metric_determinant, Fraction(1))
                self.assertEqual(result.lambda1_squared_coefficient, Fraction(1))
                self.assertEqual(result.squared_systole_coefficient, Fraction(1, 4))
                self.assertEqual(result.class_multiplicity, 2 * dimension)
                self.assertEqual(result.lift_multiplicity, 4 * dimension)
                self.assertEqual(result.full_kernel_class_count, 2 ** (2 * dimension) - 1)
                self.assertTrue(result.certified)

    def test_hexagonal_products_have_known_dimension_generic_formula(self):
        for dimension in (1, 2, 4, 8):
            with self.subTest(dimension=dimension):
                model = hexagonal_product_model(dimension)
                result = model.compute_uniform_systole(2)

                self.assertEqual(result.lambda1_squared_coefficient, Fraction(2))
                self.assertEqual(result.squared_systole_coefficient, Fraction(1, 2))
                self.assertAlmostEqual(result.squared_systole, 1 / (2 * sqrt(3)))
                self.assertEqual(result.class_multiplicity, 3 * dimension)
                self.assertEqual(result.lift_multiplicity, 6 * dimension)
                self.assertTrue(result.certified)

    def test_d4_squared_is_a_certified_g4_baseline(self):
        model = repeated_period_model(D4_PERIOD_MODEL, 2)
        result = model.compute_uniform_systole(2)

        self.assertEqual(model.dimension, 4)
        self.assertEqual(model.validation_certificate().polarization_type, (1, 1, 1, 1))
        self.assertEqual(result.squared_systole_coefficient, Fraction(1, 2))
        self.assertAlmostEqual(result.squared_systole, 1 / (2 * sqrt(2)))
        self.assertEqual(result.class_multiplicity, 24)
        self.assertEqual(result.lift_multiplicity, 48)
        self.assertEqual(result.full_kernel_class_count, 255)

    def test_klein_squared_is_a_certified_g6_baseline(self):
        model = repeated_period_model(KLEIN_QUARTIC_PERIOD_MODEL, 2)
        result = model.compute_uniform_systole(2)

        self.assertEqual(model.dimension, 6)
        self.assertEqual(result.squared_systole_coefficient, Fraction(2))
        self.assertAlmostEqual(result.squared_systole, 1 / sqrt(7))
        self.assertEqual(result.class_multiplicity, 42)
        self.assertEqual(result.lift_multiplicity, 84)
        self.assertEqual(result.full_kernel_class_count, 4095)

    def test_product_bottleneck_keeps_distance_and_adds_multiplicity(self):
        one = D4_PERIOD_MODEL.compute_uniform_systole_via_svp(2)
        two = repeated_period_model(D4_PERIOD_MODEL, 2).compute_uniform_systole(2)

        self.assertEqual(two.squared_systole_coefficient, one.squared_systole_coefficient)
        self.assertEqual(two.class_multiplicity, 2 * one.class_multiplicity)
        self.assertEqual(two.lift_multiplicity, 2 * one.lift_multiplicity)

    def test_rejects_products_with_different_algebraic_scales(self):
        square = square_product_model(1).factors[0]
        hexagonal = hexagonal_product_model(1).factors[0]
        with self.assertRaisesRegex(ValueError, "common scale radicand"):
            ProductPPAVModel("mixed", (square, hexagonal))


if __name__ == "__main__":
    unittest.main()
