import unittest

from gkp_systole import (
    CyclotomicFivePolarization,
    high_precision_cyclotomic_five_systole,
    survey_cyclotomic_five_polarizations,
)


class CyclotomicFiveTests(unittest.TestCase):
    def test_principal_trace_pairing(self):
        polarization = CyclotomicFivePolarization(1, 0)
        polarization.validate()
        self.assertEqual(polarization.field, "Q(zeta_5)")
        self.assertEqual(polarization.cm_type, (1, 2))
        self.assertTrue(polarization.simple_cm)
        self.assertEqual(polarization.polarization_type, (1, 1))
        self.assertLess(max(polarization.validation_residuals().values()), 1e-12)

    def test_type_one_five_example(self):
        polarization = CyclotomicFivePolarization(2, -1)
        result = polarization.compute_relative_systole()
        self.assertEqual(polarization.real_norm, 5)
        self.assertEqual(polarization.polarization_type, (1, 5))
        self.assertAlmostEqual(float(result.squared_systole), 0.5505527681884692)
        self.assertFalse(result.certified)

    def test_high_precision_recheck(self):
        polarization = CyclotomicFivePolarization(2, -1)
        screened = float(polarization.compute_relative_systole().squared_systole)
        checked = float(
            high_precision_cyclotomic_five_systole(
                polarization,
                decimal_places=45,
            )
        )
        self.assertAlmostEqual(checked, screened, places=12)

    def test_bounded_survey_contains_multiple_types(self):
        survey = survey_cyclotomic_five_polarizations(6)
        types = {item.polarization.polarization_type for item in survey}
        self.assertIn((1, 5), types)
        self.assertIn((1, 11), types)
        self.assertIn((1, 19), types)
        self.assertIn((2, 10), types)

    def test_nonpositive_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            CyclotomicFivePolarization(0, 1)


if __name__ == "__main__":
    unittest.main()
