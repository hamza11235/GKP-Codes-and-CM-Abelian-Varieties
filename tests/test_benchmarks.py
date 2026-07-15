import unittest

from gkp_systole import (
    Polarization,
    initial_benchmarks,
    reference_uniform_relative_systole_squared,
)


class InitialBenchmarkTests(unittest.TestCase):
    def test_all_benchmark_polarization_types(self):
        for benchmark in initial_benchmarks:
            with self.subTest(benchmark=benchmark.name):
                polarization = Polarization(benchmark.alternating)
                self.assertEqual(polarization.type, benchmark.polarization_type)
                self.assertEqual(
                    polarization.kernel_order,
                    benchmark.expected_kernel_order,
                )

    def test_square_and_hexagonal_metric_determinants_are_one(self):
        for benchmark in initial_benchmarks[:2]:
            metric = benchmark.metric
            determinant = metric[0][0] * metric[1][1] - metric[0][1] * metric[1][0]
            with self.subTest(benchmark=benchmark.name):
                self.assertAlmostEqual(determinant, 1.0, places=12)

    def test_hexagonal_expected_distance_beats_square(self):
        square, hexagonal = initial_benchmarks[:2]
        self.assertGreater(
            hexagonal.expected_relative_systole_squared,
            square.expected_relative_systole_squared,
        )

    def test_reference_search_recovers_square_and_hexagonal_values(self):
        for benchmark in initial_benchmarks[:2]:
            with self.subTest(benchmark=benchmark.name):
                computed = reference_uniform_relative_systole_squared(
                    benchmark.metric,
                    d=2,
                )
                self.assertAlmostEqual(
                    computed,
                    benchmark.expected_relative_systole_squared,
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
