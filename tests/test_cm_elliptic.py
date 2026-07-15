import unittest
from math import sqrt

from gkp_systole import (
    ReducedQuadraticForm,
    cm_elliptic_period_model,
    reduced_primitive_forms,
    scaled_squared_systole_greater,
    survey_cm_elliptic_curves,
)


class ReducedFormTests(unittest.TestCase):
    def test_class_number_one_forms(self):
        self.assertEqual(
            reduced_primitive_forms(-3),
            (ReducedQuadraticForm(1, 1, 1),),
        )
        self.assertEqual(
            reduced_primitive_forms(-4),
            (ReducedQuadraticForm(1, 0, 1),),
        )

    def test_discriminant_minus_twenty_has_two_classes(self):
        self.assertEqual(
            reduced_primitive_forms(-20),
            (
                ReducedQuadraticForm(1, 0, 5),
                ReducedQuadraticForm(2, 2, 3),
            ),
        )

    def test_invalid_discriminant(self):
        with self.assertRaises(ValueError):
            reduced_primitive_forms(-6)


class CMPeriodModelTests(unittest.TestCase):
    def test_hexagonal_period_and_systole(self):
        form = ReducedQuadraticForm(1, 1, 1)
        model = cm_elliptic_period_model(form)
        model.validate()
        result = model.compute_qubit_systole()
        self.assertAlmostEqual(result.squared_systole, 1 / (2 * sqrt(3)), places=12)
        self.assertEqual(result.class_multiplicity, 3)
        self.assertEqual(result.lift_multiplicity, 6)
        self.assertTrue(result.certified)

    def test_square_period_and_systole(self):
        form = ReducedQuadraticForm(1, 0, 1)
        result = cm_elliptic_period_model(form).compute_qubit_systole()
        self.assertAlmostEqual(result.squared_systole, 1 / 4, places=12)
        self.assertEqual(result.class_multiplicity, 2)
        self.assertEqual(result.lift_multiplicity, 4)

    def test_hexagonal_beats_square_exactly(self):
        hexagonal = cm_elliptic_period_model(
            ReducedQuadraticForm(1, 1, 1)
        ).compute_qubit_systole()
        square = cm_elliptic_period_model(
            ReducedQuadraticForm(1, 0, 1)
        ).compute_qubit_systole()
        self.assertTrue(scaled_squared_systole_greater(hexagonal, square))

    def test_survey_is_ranked_with_hexagonal_first(self):
        survey = survey_cm_elliptic_curves(100, level=2)
        self.assertGreater(len(survey), 10)
        self.assertEqual(survey[0].discriminant, -3)
        self.assertEqual(survey[0].form, ReducedQuadraticForm(1, 1, 1))
        self.assertTrue(all(item.systole_result.certified for item in survey))

    def test_uniform_level_only_rescales_ranking(self):
        qubits = survey_cm_elliptic_curves(40, level=2)
        qutrits = survey_cm_elliptic_curves(40, level=3)
        self.assertEqual(
            [(item.discriminant, item.form) for item in qubits],
            [(item.discriminant, item.form) for item in qutrits],
        )


if __name__ == "__main__":
    unittest.main()
