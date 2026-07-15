import unittest
from fractions import Fraction

from gkp_systole import (
    G3_TYPE_112_GAUSSIAN_FORM,
    G3_TYPE_113_EISENSTEIN_FORM,
    G3_TYPE_122_GAUSSIAN_FORM,
    ImaginaryQuadraticOrder,
    TernaryQuadraticHermitianForm,
    bounded_ternary_hermitian_forms,
    ternary_hermitian_moduli_family,
)


class TernaryHermitianTests(unittest.TestCase):
    def test_phase8_exact_cm_benchmarks(self):
        expected = {
            G3_TYPE_112_GAUSSIAN_FORM: (Fraction(2), 1.0, 3, 24),
            G3_TYPE_113_EISENSTEIN_FORM: (Fraction(2), 2.0 / (3.0**0.5), 8, 72),
            G3_TYPE_122_GAUSSIAN_FORM: (Fraction(2), 1.0, 15, 60),
        }
        for form, (core_squared, physical_squared, classes, lifts) in expected.items():
            result = form.compute_core_relative_systole()
            self.assertEqual(result.squared_systole, core_squared)
            self.assertAlmostEqual(
                float(core_squared) / form.order.radicand**0.5,
                physical_squared,
            )
            self.assertEqual(result.class_multiplicity, classes)
            self.assertEqual(result.lift_multiplicity, lifts)

    def test_gaussian_product_benchmarks(self):
        order = ImaginaryQuadraticOrder(-4)
        expected = {
            (1, 1, 2): (Fraction(1), 0.5),
            (1, 1, 3): (Fraction(2, 3), 1.0 / 3.0),
            (1, 2, 2): (Fraction(1), 0.5),
        }
        for diagonal, (core_squared, physical_squared) in expected.items():
            form = TernaryQuadraticHermitianForm(order, *diagonal)
            form.validate()
            self.assertEqual(form.polarization_type, diagonal)
            result = form.compute_core_relative_systole()
            self.assertEqual(result.squared_systole, core_squared)
            self.assertAlmostEqual(float(core_squared) / 2.0, physical_squared)

    def test_coupled_form_has_consistent_exact_invariants(self):
        form = TernaryQuadraticHermitianForm(
            ImaginaryQuadraticOrder(-4),
            1,
            2,
            3,
            0,
            0,
            -1,
            0,
            0,
            0,
        )
        self.assertEqual(form.determinant, 4)
        self.assertEqual(form.polarization_type, (1, 2, 2))
        form.validate()
        self.assertTrue(form.compute_core_relative_systole().certified)

    def test_bounded_enumerator_contains_product_and_coupled_forms(self):
        forms = bounded_ternary_hermitian_forms(
            ImaginaryQuadraticOrder(-4),
            2,
            maximum_diagonal=4,
            off_diagonal_bound=1,
            requested_types=((1, 1, 2),),
        )
        self.assertIn(
            TernaryQuadraticHermitianForm(ImaginaryQuadraticOrder(-4), 1, 1, 2),
            forms,
        )
        self.assertTrue(any(form.is_coupled for form in forms))

    def test_moduli_family_is_twelve_dimensional(self):
        form = TernaryQuadraticHermitianForm(ImaginaryQuadraticOrder(-4), 1, 1, 2)
        family = ternary_hermitian_moduli_family(form)
        self.assertEqual(family.coordinate_dimension, 12)
        self.assertAlmostEqual(family.reference_ell_squared, 0.5)
        self.assertAlmostEqual(family.evaluate((0.0,) * 12).squared_systole, 0.5)

    def test_invalid_or_nonpositive_forms_are_rejected(self):
        with self.assertRaises(ValueError):
            TernaryQuadraticHermitianForm(ImaginaryQuadraticOrder(-4), 1, 1, 1, 1, 0)
        with self.assertRaises(ValueError):
            bounded_ternary_hermitian_forms(
                ImaginaryQuadraticOrder(-4),
                2,
                off_diagonal_bound=-1,
            )


if __name__ == "__main__":
    unittest.main()
