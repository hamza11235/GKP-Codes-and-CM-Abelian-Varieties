import unittest
from fractions import Fraction
from math import sqrt

from gkp_systole import (
    GaussianHermitianForm,
    ImaginaryQuadraticOrder,
    QuadraticHermitianForm,
    bounded_quadratic_hermitian_forms,
    survey_quadratic_hermitian_polarizations,
)


class ImaginaryQuadraticOrderTests(unittest.TestCase):
    def test_norms_and_units(self):
        gaussian = ImaginaryQuadraticOrder(-4)
        eisenstein = ImaginaryQuadraticOrder(-3)
        self.assertEqual(gaussian.norm((1, 1)), 2)
        self.assertEqual(len(gaussian.units), 4)
        self.assertEqual(eisenstein.norm((1, -1)), 1)
        self.assertEqual(len(eisenstein.units), 6)

    def test_invalid_discriminant_is_rejected(self):
        with self.assertRaises(ValueError):
            ImaginaryQuadraticOrder(5)
        with self.assertRaises(ValueError):
            ImaginaryQuadraticOrder(-5)


class QuadraticHermitianFormTests(unittest.TestCase):
    def test_general_model_reproduces_gaussian_winner(self):
        general = QuadraticHermitianForm(ImaginaryQuadraticOrder(-4), 2, 2, 1, 1)
        # The general module uses the opposite off-diagonal convention from
        # the older specialized Gaussian class.
        specialized = GaussianHermitianForm(2, 2, 1, -1)
        general.validate()
        self.assertEqual(general.polarization_type, (1, 2))
        self.assertEqual(general.metric_core, tuple(tuple(2 * x for x in row) for row in specialized.metric))
        result = general.compute_core_relative_systole()
        self.assertEqual(result.squared_systole, Fraction(2))
        self.assertAlmostEqual(float(result.squared_systole) / 2.0, 1.0)

    def test_new_discriminant_minus_twenty_four_winner_is_exact(self):
        form = QuadraticHermitianForm(ImaginaryQuadraticOrder(-24), 6, 6, 3, -2)
        certificate = form.validation_certificate()
        result = form.compute_core_relative_systole()
        self.assertTrue(certificate.certified)
        self.assertEqual(form.polarization_type, (1, 3))
        self.assertEqual(result.squared_systole, Fraction(4))
        self.assertAlmostEqual(float(result.squared_systole) / sqrt(24), sqrt(2 / 3))

    def test_bounded_enumerator_reaches_nonuniform_and_nonprimitive_types(self):
        order = ImaginaryQuadraticOrder(-4)
        type_four = {
            form.polarization_type
            for form in bounded_quadratic_hermitian_forms(order, 4, maximum_diagonal=8)
        }
        type_eight = {
            form.polarization_type
            for form in bounded_quadratic_hermitian_forms(order, 8, maximum_diagonal=8)
        }
        self.assertIn((1, 4), type_four)
        self.assertIn((2, 2), type_four)
        self.assertIn((2, 4), type_eight)

    def test_small_survey_ranks_new_one_three_candidate_first(self):
        survey = survey_quadratic_hermitian_polarizations(
            (-3, -4, -24),
            3,
            maximum_diagonal=6,
        )
        self.assertEqual(survey[0].form.order.discriminant, -24)
        self.assertEqual(survey[0].squared_systole_coefficient, Fraction(4))
        self.assertTrue(all(item.core_systole_result.certified for item in survey))


if __name__ == "__main__":
    unittest.main()
