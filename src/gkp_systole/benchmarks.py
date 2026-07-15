"""Small, transparent benchmark inputs for the initial implementation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Sequence

from .conventions import MetricConvention


def canonical_alternating(polarization_type: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return J_D = [[0,D],[-D,0]] for a polarization type D."""

    values = tuple(int(value) for value in polarization_type)
    if not values or any(value <= 0 for value in values):
        raise ValueError("polarization entries must be positive")
    if any(right % left for left, right in zip(values, values[1:])):
        raise ValueError("polarization entries must form a divisibility chain")

    dimension = len(values)
    size = 2 * dimension
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for index, value in enumerate(values):
        matrix[index][dimension + index] = value
        matrix[dimension + index][index] = -value
    return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True)
class Benchmark:
    name: str
    polarization_type: tuple[int, ...]
    expected_kernel_order: int
    metric: tuple[tuple[float, ...], ...] | None = None
    expected_relative_systole_squared: float | None = None
    metric_convention: MetricConvention | None = None

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        return canonical_alternating(self.polarization_type)


initial_benchmarks = (
    Benchmark(
        name="square_one_mode_qubit",
        polarization_type=(2,),
        expected_kernel_order=4,
        metric=((1, 0), (0, 1)),
        expected_relative_systole_squared=0.25,
        metric_convention=MetricConvention.FIXED_PRINCIPAL,
    ),
    Benchmark(
        name="hexagonal_one_mode_qubit",
        polarization_type=(2,),
        expected_kernel_order=4,
        metric=(
            (2.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
            (1.0 / sqrt(3.0), 2.0 / sqrt(3.0)),
        ),
        expected_relative_systole_squared=1.0 / (2.0 * sqrt(3.0)),
        metric_convention=MetricConvention.FIXED_PRINCIPAL,
    ),
    Benchmark(
        name="one_qubit_two_modes",
        polarization_type=(1, 2),
        expected_kernel_order=4,
    ),
    Benchmark(
        name="d4_type_two_mode_qubits",
        polarization_type=(2, 2),
        expected_kernel_order=16,
        metric=(
            (2, -1, 0, 0),
            (-1, 2, -1, -1),
            (0, -1, 2, 0),
            (0, -1, 0, 2),
        ),
        expected_relative_systole_squared=0.5,
        metric_convention=MetricConvention.POLARIZATION_SCALED,
    ),
    Benchmark(
        name="klein_type_three_mode_qubits",
        polarization_type=(2, 2, 2),
        expected_kernel_order=64,
    ),
)


def reference_uniform_relative_systole_squared(
    metric: Sequence[Sequence[float]],
    d: int,
    *,
    search_radius: int = 2,
) -> float:
    """Brute-force reference value for a uniform type ``(d,...,d)``.

    In this case ``Lambda^perp = (1/d) Lambda``. In lattice coordinates, the
    routine enumerates short integer numerators ``u`` for ``u/d`` and excludes
    vectors already in ``Lambda``. This is intentionally a transparent
    benchmark helper, not the production closest-vector solver.
    """

    if d <= 1:
        raise ValueError("d must be greater than one")
    if search_radius < 1:
        raise ValueError("search_radius must be positive")
    rows = tuple(tuple(float(value) for value in row) for row in metric)
    dimension = len(rows)
    if not rows or any(len(row) != dimension for row in rows):
        raise ValueError("metric must be a nonempty square matrix")

    best = float("inf")
    coordinate_range = range(-search_radius, search_radius + 1)
    for numerator in product(coordinate_range, repeat=dimension):
        if all(value % d == 0 for value in numerator):
            continue
        vector = tuple(value / d for value in numerator)
        squared_length = sum(
            vector[row] * rows[row][column] * vector[column]
            for row in range(dimension)
            for column in range(dimension)
        )
        best = min(best, squared_length)

    if best == float("inf"):
        raise RuntimeError("reference search found no nontrivial dual vector")
    return best
