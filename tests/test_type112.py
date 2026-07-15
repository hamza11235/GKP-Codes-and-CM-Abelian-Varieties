import unittest
from fractions import Fraction

from gkp_systole import (
    TYPE_112_EXACT_MODEL,
    TYPE_112_METRIC_CORE,
    reconstruct_type112_metric_core,
)


class Type112ExactReconstructionTests(unittest.TestCase):
    def test_exact_ppav_and_relative_systole(self):
        model = TYPE_112_EXACT_MODEL
        certificate = model.validation_certificate()
        result = model.core_relative_systole()
        self.assertTrue(certificate.certified)
        self.assertEqual(model.polarization.type, (1, 1, 2))
        self.assertEqual(result.squared_systole, Fraction(2))
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 36)
        self.assertTrue(result.certified)

    def test_numerical_metric_reconstructs_exact_core(self):
        model = TYPE_112_EXACT_MODEL
        perturbed = tuple(
            tuple(value + (1e-7 if row == column else -1e-7) for column, value in enumerate(values))
            for row, values in enumerate(model.metric_numeric)
        )
        reconstructed = reconstruct_type112_metric_core(perturbed)
        expected = tuple(tuple(Fraction(value) for value in row) for row in TYPE_112_METRIC_CORE)
        self.assertEqual(reconstructed, expected)

    def test_cm_isogeny_certificate(self):
        certificate = TYPE_112_EXACT_MODEL.cm_certificate()
        self.assertTrue(certificate.is_cm)
        self.assertEqual(certificate.field, "Q(sqrt(-3))")
        self.assertEqual(certificate.order_discriminant, -12)
        self.assertEqual(certificate.commutant_dimension, 18)
        self.assertEqual(certificate.rational_isogeny_degree, 72)

    def test_reconstruction_rejects_wrong_shape_and_distant_metric(self):
        with self.assertRaises(ValueError):
            reconstruct_type112_metric_core(((1.0,),))
        distant = tuple(
            tuple(value + (0.1 if row == column else 0.0) for column, value in enumerate(values))
            for row, values in enumerate(TYPE_112_EXACT_MODEL.metric_numeric)
        )
        with self.assertRaises(ArithmeticError):
            reconstruct_type112_metric_core(distant)


if __name__ == "__main__":
    unittest.main()
