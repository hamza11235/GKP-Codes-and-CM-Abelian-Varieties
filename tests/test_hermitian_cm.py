import unittest
from fractions import Fraction

from gkp_systole import (
    GaussianHermitianForm,
    MetricConvention,
    reduced_gaussian_hermitian_forms,
    survey_gaussian_cm_polarizations,
)


class GaussianHermitianTests(unittest.TestCase):
    def test_product_type_one_two(self):
        product = GaussianHermitianForm(1, 2)
        product.validate()
        result = product.compute_relative_systole()
        self.assertEqual(product.polarization_type, (1, 2))
        self.assertEqual(result.squared_systole, Fraction(1, 2))
        self.assertEqual(result.class_multiplicity, 2)
        self.assertEqual(result.lift_multiplicity, 4)
        self.assertEqual(
            result.metric_convention,
            MetricConvention.POLARIZATION_SCALED,
        )

    def test_coupled_type_one_two(self):
        coupled = GaussianHermitianForm(2, 2, 1, 1)
        coupled.validate()
        result = coupled.compute_relative_systole()
        self.assertEqual(coupled.determinant, 2)
        self.assertEqual(coupled.polarization_type, (1, 2))
        self.assertEqual(result.squared_systole, Fraction(1))
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 24)
        self.assertTrue(result.certified)
        self.assertEqual(
            result.metric_convention,
            MetricConvention.POLARIZATION_SCALED,
        )

    def test_reduced_determinant_two_domain(self):
        self.assertEqual(
            reduced_gaussian_hermitian_forms(2),
            (
                GaussianHermitianForm(1, 2, 0, 0),
                GaussianHermitianForm(2, 2, 1, 1),
            ),
        )

    def test_reduced_determinant_three_domain(self):
        self.assertEqual(
            reduced_gaussian_hermitian_forms(3),
            (
                GaussianHermitianForm(1, 3, 0, 0),
                GaussianHermitianForm(2, 2, 1, 0),
            ),
        )

    def test_coupled_type_one_three_improves_product(self):
        scan = survey_gaussian_cm_polarizations(3)
        self.assertEqual(scan[0].form, GaussianHermitianForm(2, 2, 1, 0))
        self.assertEqual(scan[0].form.polarization_type, (1, 3))
        self.assertEqual(scan[0].systole_result.squared_systole, Fraction(2, 3))
        self.assertEqual(scan[1].systole_result.squared_systole, Fraction(1, 3))


if __name__ == "__main__":
    unittest.main()
