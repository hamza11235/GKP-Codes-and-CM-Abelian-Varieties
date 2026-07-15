"""Closest integer translates for rational torus points."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, sqrt
from typing import Sequence

from .kernel import KernelElement
from .metric import Metric, Scalar


IntegerVector = tuple[int, ...]
LiftVector = tuple[Fraction, ...]


def _nearest_integer(value: Fraction) -> int:
    return floor(float(value) + 0.5)


def _lift(coordinates: Sequence[Fraction], shift: Sequence[int]) -> LiftVector:
    return tuple(value + integer for value, integer in zip(coordinates, shift))


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


@dataclass(frozen=True)
class ClosestVectorResult:
    """All shortest representatives of one torus class."""

    element: KernelElement
    squared_distance: Scalar
    shifts: tuple[IntegerVector, ...]
    lifts: tuple[LiftVector, ...]
    certified: bool
    method: str

    @property
    def distance(self) -> float:
        return sqrt(float(self.squared_distance))


def _initial_shift(element: KernelElement) -> IntegerVector:
    return tuple(-_nearest_integer(value) for value in element.coordinates)


def _exact_ldlt(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    """Return exact unit-lower L and diagonal D with G = L D L^T."""

    size = len(matrix)
    lower = [[Fraction(int(row == column)) for column in range(size)] for row in range(size)]
    diagonal = [Fraction(0) for _ in range(size)]
    for column in range(size):
        diagonal[column] = Fraction(matrix[column][column]) - sum(
            (
                lower[column][inner]
                * lower[column][inner]
                * diagonal[inner]
                for inner in range(column)
            ),
            Fraction(0),
        )
        if diagonal[column] <= 0:
            raise ValueError("exact metric must be positive definite")
        for row in range(column + 1, size):
            numerator = Fraction(matrix[row][column]) - sum(
                (
                    lower[row][inner]
                    * lower[column][inner]
                    * diagonal[inner]
                    for inner in range(column)
                ),
                Fraction(0),
            )
            lower[row][column] = numerator / diagonal[column]
    return tuple(tuple(row) for row in lower), tuple(diagonal)


def _integers_within_quadratic_bound(
    center: Fraction,
    weight: Fraction,
    bound: Fraction,
) -> tuple[int, ...]:
    """Enumerate all integers n with weight*(n-center)^2 <= bound."""

    if bound < 0:
        return ()
    base = _floor_fraction(center)
    candidates: list[int] = []

    integer = base
    while weight * (Fraction(integer) - center) ** 2 <= bound:
        candidates.append(integer)
        integer -= 1

    integer = base + 1
    while weight * (Fraction(integer) - center) ** 2 <= bound:
        candidates.append(integer)
        integer += 1

    return tuple(sorted(candidates, key=lambda value: abs(Fraction(value) - center)))


def _exact_branch_and_bound_solver(
    element: KernelElement,
    metric: Metric,
) -> ClosestVectorResult:
    """Certified exact LDL^T branch-and-bound for a rational metric."""

    lower, diagonal = _exact_ldlt(metric.matrix)
    size = metric.dimension
    current = [0 for _ in range(size)]
    initial = _initial_shift(element)
    best = Fraction(metric.quadratic(_lift(element.coordinates, initial)))
    best_shifts: list[IntegerVector] = [initial]

    def recurse(index: int, partial: Fraction) -> None:
        nonlocal best, best_shifts
        if index < 0:
            shift = tuple(current)
            if partial < best:
                best = partial
                best_shifts = [shift]
            elif partial == best:
                best_shifts.append(shift)
            return

        # With U=L^T, the index-th contribution is
        # D_i * (c_i+n_i + sum_{j>i} U_ij(c_j+n_j))^2.
        tail = sum(
            (
                lower[column][index]
                * (element.coordinates[column] + current[column])
                for column in range(index + 1, size)
            ),
            Fraction(0),
        )
        center = -element.coordinates[index] - tail
        remaining = best - partial
        for integer in _integers_within_quadratic_bound(
            center,
            diagonal[index],
            remaining,
        ):
            current[index] = integer
            contribution = diagonal[index] * (Fraction(integer) - center) ** 2
            recurse(index - 1, partial + contribution)

    recurse(size - 1, Fraction(0))
    shifts = tuple(sorted(set(best_shifts)))
    return ClosestVectorResult(
        element=element,
        squared_distance=best,
        shifts=shifts,
        lifts=tuple(_lift(element.coordinates, shift) for shift in shifts),
        certified=True,
        method="exact_ldlt_branch_and_bound",
    )


def _branch_and_bound_solver(
    element: KernelElement,
    metric: Metric,
    tolerance: float,
) -> ClosestVectorResult:
    """Cholesky branch-and-bound for a floating-point metric."""

    upper_factor = metric.cholesky_upper
    size = metric.dimension
    current = [0 for _ in range(size)]
    initial = _initial_shift(element)
    best_value = float(metric.quadratic(_lift(element.coordinates, initial)))
    best_shifts: list[IntegerVector] = [initial]

    def recurse(index: int, partial: float) -> None:
        nonlocal best_value, best_shifts
        if index < 0:
            shift = tuple(current)
            squared = float(metric.quadratic(_lift(element.coordinates, shift)))
            scale = max(1.0, abs(best_value), abs(squared))
            if squared < best_value - tolerance * scale:
                best_value = squared
                best_shifts = [shift]
            elif abs(squared - best_value) <= tolerance * scale:
                best_shifts.append(shift)
            return

        remaining = max(0.0, best_value - partial)
        tail = sum(
            upper_factor[index][column]
            * (float(element.coordinates[column]) + current[column])
            for column in range(index + 1, size)
        )
        diagonal = upper_factor[index][index]
        center = -float(element.coordinates[index]) - tail / diagonal
        radius = sqrt(remaining) / abs(diagonal)
        lower = ceil(center - radius - 10 * tolerance)
        upper = floor(center + radius + 10 * tolerance)
        candidates = sorted(range(lower, upper + 1), key=lambda value: abs(value - center))
        for integer in candidates:
            current[index] = integer
            row_value = diagonal * (float(element.coordinates[index]) + integer) + tail
            new_partial = partial + row_value * row_value
            if new_partial <= best_value + tolerance * max(1.0, best_value):
                recurse(index - 1, new_partial)

    recurse(size - 1, 0.0)
    shifts = tuple(sorted(set(best_shifts)))
    return ClosestVectorResult(
        element=element,
        squared_distance=best_value,
        shifts=shifts,
        lifts=tuple(_lift(element.coordinates, shift) for shift in shifts),
        certified=False,
        method="branch_and_bound",
    )


def closest_integer_translate(
    element: KernelElement,
    metric: Metric | Sequence[Sequence[float | int | Fraction]],
    *,
    tolerance: float = 1e-12,
) -> ClosestVectorResult:
    """Minimize ``(c+n)^T G (c+n)`` over integral ``n``."""

    metric_object = metric if isinstance(metric, Metric) else Metric(metric)
    if len(element.coordinates) != metric_object.dimension:
        raise ValueError("kernel element and metric dimensions do not match")
    if metric_object.is_exact:
        return _exact_branch_and_bound_solver(element, metric_object)
    return _branch_and_bound_solver(element, metric_object, tolerance)
