import unittest
from fractions import Fraction

from gkp_systole import (
    D4_PERIOD_MODEL,
    KLEIN_QUARTIC_PERIOD_MODEL,
    GaussianHermitianForm,
    EisensteinHermitianForm,
    PPAVValidationError,
    canonical_alternating,
    validate_polarized_abelian_data,
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


class PPAVValidationTests(unittest.TestCase):
    def test_square_principal_certificate(self):
        metric = ((1, 0), (0, 1))
        complex_structure = ((0, 1), (-1, 0))
        result = validate_ppav_data(metric, complex_structure)

        self.assertTrue(result.certified)
        self.assertTrue(result.principal)
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.polarization_type, (1,))
        self.assertEqual(result.polarization.matrix, complex_structure)
        self.assertEqual(result.physical_metric_determinant, Fraction(1))
        self.assertEqual(len(result.checks), 6)

    def test_scaled_d4_and_klein_certificates(self):
        for model in (D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL):
            with self.subTest(model=model.name):
                result = model.validation_certificate()
                self.assertTrue(result.principal)
                self.assertEqual(result.polarization_type, (1,) * model.dimension)
                self.assertEqual(result.scale_radicand, model.scale_radicand)
                self.assertEqual(result.physical_metric_determinant, Fraction(1))

    def test_general_validator_accepts_nonprincipal_cm_polarizations(self):
        gaussian = GaussianHermitianForm(1, 2)
        eisenstein = EisensteinHermitianForm(1, 3)

        gaussian_result = gaussian.validation_certificate()
        eisenstein_result = eisenstein.validation_certificate()

        self.assertEqual(gaussian_result.polarization_type, (1, 2))
        self.assertEqual(gaussian_result.physical_metric_determinant, Fraction(4))
        self.assertEqual(eisenstein_result.polarization_type, (1, 3))
        self.assertEqual(eisenstein_result.physical_metric_determinant, Fraction(9))

    def test_principal_wrapper_rejects_nonprincipal_data(self):
        form = GaussianHermitianForm(1, 2)
        with self.assertRaisesRegex(PPAVValidationError, "principal polarization"):
            validate_ppav_data(
                form.metric,
                (
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                    (-1, 0, 0, 0),
                    (0, -1, 0, 0),
                ),
                form.alternating,
            )

    def test_rejects_invalid_complex_structure(self):
        with self.assertRaisesRegex(PPAVValidationError, r"J_num\^2"):
            validate_ppav_data(((1, 0), (0, 1)), ((0, 0), (0, 0)))

    def test_rejects_metric_incompatible_complex_structure(self):
        with self.assertRaisesRegex(PPAVValidationError, "not an isometry"):
            validate_ppav_data(((1, 0), (0, 2)), ((0, 1), (-1, 0)))

    def test_rejects_nonintegral_riemann_form(self):
        with self.assertRaisesRegex(PPAVValidationError, "not integral"):
            validate_polarized_abelian_data(
                ((Fraction(1, 2), 0), (0, Fraction(1, 2))),
                ((0, 1), (-1, 0)),
            )

    def test_rejects_claimed_polarization_mismatch(self):
        with self.assertRaisesRegex(PPAVValidationError, "does not equal"):
            validate_ppav_data(
                ((1, 0), (0, 1)),
                ((0, 1), (-1, 0)),
                ((0, -1), (1, 0)),
            )

    def test_rejects_float_inputs_for_exact_certification(self):
        with self.assertRaisesRegex(PPAVValidationError, "floats are not certified"):
            validate_ppav_data(((1.0, 0.0), (0.0, 1.0)), ((0, 1), (-1, 0)))
        with self.assertRaisesRegex(PPAVValidationError, "floats are not certified"):
            validate_ppav_data(((1, 0), (0, 1)), ((0.0, 1.0), (-1.0, 0.0)))

    def test_certificate_is_invariant_under_unimodular_basis_change(self):
        metric = ((1, 0), (0, 1))
        alternating = canonical_alternating((1,))
        complex_structure = alternating
        change = ((1, 1), (0, 1))
        inverse = ((1, -1), (0, 1))

        transformed_metric = multiply(multiply(transpose(change), metric), change)
        transformed_alternating = multiply(
            multiply(transpose(change), alternating), change
        )
        transformed_alternating = tuple(
            tuple(int(value) for value in row) for row in transformed_alternating
        )
        transformed_complex = multiply(
            multiply(inverse, complex_structure), change
        )

        result = validate_ppav_data(
            transformed_metric,
            transformed_complex,
            transformed_alternating,
        )
        self.assertTrue(result.principal)
        self.assertEqual(result.physical_metric_determinant, Fraction(1))


if __name__ == "__main__":
    unittest.main()
