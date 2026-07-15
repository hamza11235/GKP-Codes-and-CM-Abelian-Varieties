import unittest
from fractions import Fraction

from gkp_systole import (
    D4_ROOT_GRAM,
    TYPE_15_EXACT_MODEL,
    TYPE_15_METRIC_CORE,
    rational_commutant_dimension,
    reconstruct_equal_distance_metric,
)


class Type15ExactReconstructionTests(unittest.TestCase):
    def test_exact_ppav_and_relative_systole(self):
        model = TYPE_15_EXACT_MODEL
        certificate = model.validation_certificate()
        result = model.core_relative_systole()
        self.assertTrue(certificate.certified)
        self.assertEqual(model.polarization.type, (1, 5))
        self.assertEqual(result.squared_systole, Fraction(2))
        self.assertEqual(result.class_multiplicity, 24)
        self.assertEqual(result.lift_multiplicity, 24)
        self.assertTrue(result.certified)

    def test_active_lifts_reconstruct_metric_shape(self):
        result = TYPE_15_EXACT_MODEL.core_relative_systole()
        lifts = tuple(class_result.lifts[0] for class_result in result.class_results)
        reconstructed = reconstruct_equal_distance_metric(lifts)
        expected = tuple(
            tuple(Fraction(value, 2) for value in row)
            for row in TYPE_15_METRIC_CORE
        )
        self.assertEqual(reconstructed, expected)

    def test_dual_lattice_is_scaled_d4(self):
        self.assertTrue(TYPE_15_EXACT_MODEL.dual_d4_certificate())
        self.assertEqual(len(D4_ROOT_GRAM), 4)

    def test_cm_certificate(self):
        certificate = TYPE_15_EXACT_MODEL.cm_certificate()
        self.assertTrue(certificate.is_cm)
        self.assertEqual(certificate.field, "Q(sqrt(-10))")
        self.assertEqual(certificate.order_discriminant, -40)
        self.assertEqual(certificate.commutant_dimension, 8)
        self.assertEqual(certificate.rational_isogeny_degree, 47)

    def test_commutant_dimension(self):
        self.assertEqual(
            rational_commutant_dimension(TYPE_15_EXACT_MODEL.complex_structure_numerator),
            8,
        )


if __name__ == "__main__":
    unittest.main()
