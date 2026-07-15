import unittest
from fractions import Fraction
from itertools import product

from gkp_systole import KernelElement, Metric, MetricError
from gkp_systole.cvp import closest_integer_translate


class MetricTests(unittest.TestCase):
    def test_exact_metric(self):
        metric = Metric(((2, 1), (1, 2)))
        self.assertTrue(metric.is_exact)
        self.assertEqual(metric.quadratic((1, -1)), Fraction(2))

    def test_rejects_nonsymmetric_metric(self):
        with self.assertRaisesRegex(MetricError, "symmetric"):
            Metric(((1, 2), (0, 1)))

    def test_rejects_nonpositive_metric(self):
        with self.assertRaisesRegex(MetricError, "positive definite"):
            Metric(((1, 2), (2, 1)))


class ClosestVectorTests(unittest.TestCase):
    def test_square_corner_has_four_shortest_lifts(self):
        element = KernelElement((Fraction(1, 2), Fraction(1, 2)))
        result = closest_integer_translate(element, Metric(((1, 0), (0, 1))))
        self.assertTrue(result.certified)
        self.assertEqual(result.squared_distance, Fraction(1, 2))
        self.assertEqual(len(result.lifts), 4)

    def test_off_diagonal_metric_changes_closest_translate(self):
        element = KernelElement((Fraction(1, 2), Fraction(1, 2)))
        metric = Metric(((2, 1), (1, 2)))
        result = closest_integer_translate(element, metric)
        self.assertEqual(result.squared_distance, Fraction(1, 2))
        self.assertEqual(
            set(result.lifts),
            {
                (Fraction(-1, 2), Fraction(1, 2)),
                (Fraction(1, 2), Fraction(-1, 2)),
            },
        )

    def test_branch_and_bound_matches_direct_enumeration(self):
        cases = (
            (
                KernelElement((Fraction(1, 3), Fraction(2, 3))),
                Metric(((1.2, 0.35), (0.35, 0.9))),
            ),
            (
                KernelElement((Fraction(1, 2), Fraction(1, 4))),
                Metric(((2.0, -0.7), (-0.7, 1.4))),
            ),
        )
        for element, metric in cases:
            with self.subTest(element=element.coordinates):
                result = closest_integer_translate(element, metric)
                direct = min(
                    float(
                        metric.quadratic(
                            tuple(
                                coordinate + shift
                                for coordinate, shift in zip(element.coordinates, shifts)
                            )
                        )
                    )
                    for shifts in product(range(-3, 4), repeat=2)
                )
                self.assertAlmostEqual(float(result.squared_distance), direct, places=12)


if __name__ == "__main__":
    unittest.main()
