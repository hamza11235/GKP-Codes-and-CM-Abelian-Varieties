import unittest
from math import sqrt

from gkp_systole import (
    CMProductSurface,
    MetricConvention,
    ReducedQuadraticForm,
    survey_cm_product_surfaces,
)


HEXAGONAL = ReducedQuadraticForm(1, 1, 1)
SQUARE = ReducedQuadraticForm(1, 0, 1)


class CMProductSurfaceTests(unittest.TestCase):
    def test_product_models_pass_exact_riemann_checks(self):
        for polarization_type in ((1, 2), (1, 3), (2, 2), (2, 4)):
            with self.subTest(polarization_type=polarization_type):
                CMProductSurface(
                    HEXAGONAL,
                    HEXAGONAL,
                    polarization_type,
                ).validate()

    def test_generalized_period_blocks(self):
        surface = CMProductSurface(HEXAGONAL, HEXAGONAL, (1, 2))
        left, right = surface.generalized_period_numeric
        tau = complex(-0.5, sqrt(3) / 2)
        self.assertEqual(left, ((1 + 0j, 0j), (0j, 2 + 0j)))
        self.assertAlmostEqual(right[0][0], tau)
        self.assertAlmostEqual(right[1][1], 2 * tau)

    def test_type_one_two_depends_on_logical_factor(self):
        thin = ReducedQuadraticForm(1, 0, 5)
        balanced = ReducedQuadraticForm(2, 2, 3)
        balanced_logical = CMProductSurface(thin, balanced, (1, 2))
        thin_logical = CMProductSurface(balanced, thin, (1, 2))
        balanced_result = balanced_logical.compute_relative_systole()
        thin_result = thin_logical.compute_relative_systole()
        self.assertAlmostEqual(balanced_result.squared_systole, 1 / sqrt(5), places=12)
        self.assertAlmostEqual(thin_result.squared_systole, 1 / (2 * sqrt(5)), places=12)
        self.assertGreater(balanced_result.squared_systole, thin_result.squared_systole)

    def test_hexagonal_product_type_two_two(self):
        result = CMProductSurface(
            HEXAGONAL,
            HEXAGONAL,
            (2, 2),
        ).compute_relative_systole()
        self.assertAlmostEqual(result.squared_systole, 1 / sqrt(3), places=12)
        self.assertEqual(result.class_multiplicity, 6)
        self.assertEqual(result.lift_multiplicity, 12)
        self.assertTrue(result.certified)
        self.assertEqual(
            result.core_result.metric_convention,
            MetricConvention.POLARIZATION_SCALED,
        )

    def test_surveys_rank_a_hexagonal_logical_factor_first(self):
        for polarization_type in ((1, 2), (1, 3), (2, 2)):
            with self.subTest(polarization_type=polarization_type):
                survey = survey_cm_product_surfaces(60, polarization_type)
                self.assertGreater(len(survey), 10)
                first = survey[0].surface
                self.assertEqual(first.second_form, HEXAGONAL)
                if polarization_type[0] > 1:
                    self.assertEqual(first.first_form, HEXAGONAL)
                self.assertTrue(
                    all(item.systole_result.certified for item in survey)
                )


if __name__ == "__main__":
    unittest.main()
