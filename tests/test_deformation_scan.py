import unittest
from fractions import Fraction

from gkp_systole import (
    GaussianHermitianForm,
    MetricConvention,
    deform_metric,
    high_precision_pi_systole,
    scan_pi_symplectic_deformations,
    scan_symplectic_deformations,
    symplectic_transvection,
)


def multiply(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0])))
        for i in range(len(left))
    )


def transpose(matrix):
    return tuple(tuple(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0])))


class DeformationScanTests(unittest.TestCase):
    def setUp(self):
        self.form = GaussianHermitianForm(2, 2, 1, 1)

    def test_transvection_preserves_polarization(self):
        a = self.form.alternating
        s = symplectic_transvection(a, (1, -1, 2, 0), Fraction(1, 7))
        self.assertEqual(multiply(multiply(transpose(s), a), s), a)

    def test_deformed_metric_remains_exact_positive(self):
        s = symplectic_transvection(
            self.form.alternating,
            (1, 0, 1, -1),
            Fraction(1, 11),
        )
        metric = deform_metric(self.form.metric, s)
        self.assertTrue(all(isinstance(value, Fraction) for row in metric for value in row))

    def test_deterministic_certified_scan(self):
        scan = scan_symplectic_deformations(
            self.form.alternating,
            self.form.metric,
            sample_count=8,
            seed=19,
            steps=3,
            numerator_bound=2,
            denominator=50,
        )
        repeated = scan_symplectic_deformations(
            self.form.alternating,
            self.form.metric,
            sample_count=8,
            seed=19,
            steps=3,
            numerator_bound=2,
            denominator=50,
        )
        self.assertEqual(
            tuple(sample.squared_systole for sample in scan.samples),
            tuple(sample.squared_systole for sample in repeated.samples),
        )
        self.assertTrue(all(sample.systole_result.certified for sample in scan.samples))
        self.assertEqual(
            scan.baseline_result.metric_convention,
            MetricConvention.POLARIZATION_SCALED,
        )

    def test_pi_scan_leaves_rational_transformation_model(self):
        scan = scan_pi_symplectic_deformations(
            self.form.alternating,
            self.form.metric,
            sample_count=3,
            seed=41,
            amplitude=0.02,
            steps=2,
        )
        self.assertTrue(all(not sample.systole_result.certified for sample in scan.samples))
        self.assertTrue(all(sample.parameters for sample in scan.samples))

    def test_high_precision_recheck_matches_screen(self):
        scan = scan_pi_symplectic_deformations(
            self.form.alternating,
            self.form.metric,
            sample_count=2,
            seed=43,
            amplitude=0.01,
            steps=2,
        )
        sample = scan.best_sample
        checked = float(
            high_precision_pi_systole(
                self.form.alternating,
                self.form.metric,
                sample.parameters,
                decimal_places=40,
            )
        )
        self.assertAlmostEqual(checked, sample.squared_systole, places=11)


if __name__ == "__main__":
    unittest.main()
