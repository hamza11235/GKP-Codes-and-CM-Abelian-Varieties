import unittest
from fractions import Fraction

from gkp_systole import (
    EisensteinHermitianForm,
    MetricConvention,
    eisenstein_norm,
    reduced_eisenstein_hermitian_forms,
    survey_eisenstein_cm_polarizations,
)


class EisensteinHermitianTests(unittest.TestCase):
    def test_eisenstein_norm(self):
        self.assertEqual(eisenstein_norm(1, 0), 1)
        self.assertEqual(eisenstein_norm(0, 1), 1)
        self.assertEqual(eisenstein_norm(1, 1), 1)
        self.assertEqual(eisenstein_norm(1, -1), 3)

    def test_product_type_one_two(self):
        product = EisensteinHermitianForm(1, 2)
        product.validate()
        result = product.compute_core_relative_systole()
        self.assertEqual(product.polarization_type, (1, 2))
        self.assertEqual(result.squared_systole, Fraction(1))
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 6)
        self.assertTrue(result.certified)
        self.assertEqual(
            result.metric_convention,
            MetricConvention.POLARIZATION_SCALED,
        )

    def test_coupled_type_one_three(self):
        coupled = EisensteinHermitianForm(2, 2, 1, 1)
        coupled.validate()
        result = coupled.compute_core_relative_systole()
        self.assertEqual(coupled.determinant, 3)
        self.assertEqual(coupled.polarization_type, (1, 3))
        self.assertEqual(result.squared_systole, Fraction(4, 3))
        self.assertEqual(result.class_multiplicity, 6)
        self.assertEqual(result.lift_multiplicity, 18)

    def test_reduced_determinant_two_domain(self):
        self.assertEqual(
            reduced_eisenstein_hermitian_forms(2),
            (EisensteinHermitianForm(1, 2, 0, 0),),
        )

    def test_reduced_determinant_three_domain(self):
        self.assertEqual(
            reduced_eisenstein_hermitian_forms(3),
            (
                EisensteinHermitianForm(1, 3, 0, 0),
                EisensteinHermitianForm(2, 2, 1, 1),
            ),
        )

    def test_coupled_type_one_three_improves_product(self):
        scan = survey_eisenstein_cm_polarizations(3)
        self.assertEqual(scan[0].form, EisensteinHermitianForm(2, 2, 1, 1))
        self.assertEqual(scan[0].squared_systole_coefficient, Fraction(4, 3))
        self.assertEqual(scan[1].squared_systole_coefficient, Fraction(2, 3))

    def test_reduced_scans_are_certified(self):
        for target in (2, 3):
            forms = reduced_eisenstein_hermitian_forms(target)
            self.assertTrue(forms)
            for form in forms:
                form.validate()
            scan = survey_eisenstein_cm_polarizations(target)
            self.assertTrue(all(item.core_systole_result.certified for item in scan))


if __name__ == "__main__":
    unittest.main()
