import unittest

import numpy as np

from gkp_systole import (
    CompatibleMetricFamily,
    canonical_alternating,
    eisenstein_type_13_family,
    gaussian_type_12_family,
    high_precision_coordinate_systole,
    refine_compatible_moduli_sample,
    refine_compatible_moduli_simplex,
    scan_compatible_moduli,
)


class CompatibleMetricFamilyTests(unittest.TestCase):
    def test_reference_families_have_six_coordinates_and_expected_types(self):
        gaussian = gaussian_type_12_family()
        eisenstein = eisenstein_type_13_family()

        self.assertEqual(gaussian.coordinate_dimension, 6)
        self.assertEqual(eisenstein.coordinate_dimension, 6)
        self.assertEqual(gaussian.polarization.type, (1, 2))
        self.assertEqual(eisenstein.polarization.type, (1, 3))
        self.assertAlmostEqual(gaussian.evaluate((0.0,) * 6).squared_systole, 1.0)
        self.assertAlmostEqual(
            eisenstein.evaluate((0.0,) * 6).squared_systole,
            4.0 / (3.0 * np.sqrt(3.0)),
        )

    def test_tangent_basis_satisfies_cartan_constraints(self):
        for family in (gaussian_type_12_family(), eisenstein_type_13_family()):
            alternating = np.asarray(family.alternating, dtype=float)
            metric = np.asarray(family.reference_metric, dtype=float)
            for tangent in family.tangent_basis:
                symplectic = tangent.T @ alternating + alternating @ tangent
                self_adjoint = tangent.T @ metric - metric @ tangent
                self.assertLess(np.max(np.abs(symplectic)), 2e-12)
                self.assertLess(np.max(np.abs(self_adjoint)), 2e-12)

    def test_generic_coordinates_remain_compatible(self):
        coordinates = (0.17, -0.09, 0.04, 0.13, -0.08, 0.11)
        for family in (gaussian_type_12_family(), eisenstein_type_13_family()):
            family.validate_coordinates(coordinates)
            metric = np.asarray(family.metric(coordinates))
            alternating = np.asarray(family.alternating, dtype=float)
            self.assertAlmostEqual(
                float(np.linalg.det(metric)),
                abs(float(np.linalg.det(alternating))),
                places=9,
            )

    def test_wrong_coordinate_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_type_12_family().metric((0.0,) * 5)

    def test_threefold_family_has_twelve_coordinates(self):
        alternating = canonical_alternating((1, 1, 2))
        metric = tuple(
            tuple(
                (1, 1, 2, 1, 1, 2)[row] if row == column else 0
                for column in range(6)
            )
            for row in range(6)
        )
        family = CompatibleMetricFamily.from_reference(
            name="threefold product benchmark",
            alternating=alternating,
            reference_metric=metric,
            reference_exact_ell_squared="1/2",
            reference_ell_squared=0.5,
            reference_cm="product benchmark",
        )
        self.assertEqual(family.coordinate_dimension, 12)
        family.validate_coordinates((0.0,) * 12)
        self.assertAlmostEqual(family.evaluate((0.0,) * 12).squared_systole, 0.5)


class CompatibleModuliSearchTests(unittest.TestCase):
    def test_scan_is_deterministic(self):
        family = gaussian_type_12_family()
        arguments = dict(
            sample_count=18,
            seed=20260713,
            radius=0.5,
            local_fraction=0.5,
            local_radius=0.08,
            refinement_starts=2,
            refinement_initial_step=0.04,
            refinement_minimum_step=0.01,
            refinement_rounds=8,
        )
        first = scan_compatible_moduli(family, **arguments)
        second = scan_compatible_moduli(family, **arguments)
        self.assertEqual(
            tuple(sample.coordinates for sample in first.samples),
            tuple(sample.coordinates for sample in second.samples),
        )
        self.assertEqual(
            tuple(sample.squared_systole for sample in first.samples),
            tuple(sample.squared_systole for sample in second.samples),
        )

    def test_local_refinement_improves_the_screening_point(self):
        family = eisenstein_type_13_family()
        result = scan_compatible_moduli(
            family,
            sample_count=36,
            seed=91,
            radius=0.6,
            local_fraction=0.5,
            local_radius=0.1,
            refinement_starts=3,
            refinement_initial_step=0.05,
            refinement_minimum_step=0.003,
            refinement_rounds=16,
        )
        self.assertGreaterEqual(
            result.best_sample.squared_systole,
            result.best_screen_sample.squared_systole,
        )
        self.assertEqual(result.number_beating_reference, 0)

    def test_high_precision_recheck_matches_double_precision(self):
        family = gaussian_type_12_family()
        coordinates = (0.031, -0.017, 0.009, 0.023, -0.012, 0.006)
        screened = family.evaluate(coordinates).squared_systole
        checked = float(
            high_precision_coordinate_systole(
                family,
                coordinates,
                decimal_places=45,
            )
        )
        self.assertAlmostEqual(checked, screened, places=11)

    def test_invalid_scan_arguments_are_rejected(self):
        family = gaussian_type_12_family()
        with self.assertRaises(ValueError):
            scan_compatible_moduli(family, sample_count=0, seed=1)
        with self.assertRaises(ValueError):
            scan_compatible_moduli(
                family,
                sample_count=1,
                seed=1,
                local_fraction=1.1,
            )
        with self.assertRaises(ValueError):
            scan_compatible_moduli(
                family,
                sample_count=1,
                seed=1,
                refinement_initial_step=0,
            )

    def test_random_direction_refinement_is_deterministic_and_non_decreasing(self):
        family = gaussian_type_12_family()
        start = family.evaluate((0.03, -0.02, 0.01, 0.02, -0.01, 0.005))
        arguments = dict(
            seed=17,
            direction_count=4,
            initial_step=0.01,
            minimum_step=0.005,
            maximum_rounds=3,
        )
        first = refine_compatible_moduli_sample(family, start, **arguments)
        second = refine_compatible_moduli_sample(family, start, **arguments)
        self.assertEqual(first.coordinates, second.coordinates)
        self.assertGreaterEqual(first.squared_systole, start.squared_systole)

    def test_simplex_refinement_is_non_decreasing(self):
        family = gaussian_type_12_family()
        start = family.evaluate((0.03, -0.02, 0.01, 0.02, -0.01, 0.005))
        refined = refine_compatible_moduli_simplex(
            family,
            start,
            initial_step=0.005,
            coordinate_tolerance=0.001,
            value_tolerance=1e-8,
            maximum_iterations=12,
        )
        self.assertGreaterEqual(refined.squared_systole, start.squared_systole)


if __name__ == "__main__":
    unittest.main()
