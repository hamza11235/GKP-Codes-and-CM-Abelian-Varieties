"""Exact polarized lattice-automorphism enumeration in small dimension."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Sequence

from gkp_systole.kernel import invert_rational_matrix
from gkp_systole.polarization import Polarization, determinant

from .exact import (
    IntegerMatrix,
    RationalMatrix,
    RationalVector,
    bilinear,
    columns_to_matrix,
    commute,
    congruence,
    is_positive_definite,
    rational_matrix,
)


def _ceil_sqrt_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("square-root bound must be nonnegative")
    ceiling = (value.numerator + value.denominator - 1) // value.denominator
    result = isqrt(ceiling)
    return result if result * result >= ceiling else result + 1


def _coordinate_bound(metric: RationalMatrix, target_norm: Fraction) -> int:
    """Return a rigorous coordinate bound for vectors of a fixed norm.

    For positive-definite G,

        x^T G x >= ||x||_2^2 / ||G^{-1}||_2
                  >= ||x||_2^2 / sum_ij |(G^{-1})_ij|.

    The resulting bound is intentionally conservative but fully exact.
    """

    inverse = invert_rational_matrix(metric)
    inverse_entry_bound = sum((abs(value) for row in inverse for value in row), Fraction(0))
    return _ceil_sqrt_fraction(target_norm * inverse_entry_bound)


def integer_vectors_of_norm(
    metric: Sequence[Sequence[int | Fraction]], target_norm: int | Fraction
) -> tuple[tuple[int, ...], ...]:
    """Enumerate exactly all integral vectors with the requested squared norm.

    An exact ``LDL^T`` decomposition turns the ellipsoid into nested
    one-dimensional quadratic bounds.  This avoids the exponentially larger
    coordinate box used by the first prototype and is fast enough for the
    six-dimensional Klein-quartic benchmark.
    """

    exact_metric = rational_matrix(metric)
    target = Fraction(target_norm)
    if target <= 0:
        raise ValueError("target norm must be positive")

    size = len(exact_metric)
    lower = [[Fraction(int(row == column)) for column in range(size)] for row in range(size)]
    diagonal = [Fraction(0) for _ in range(size)]
    for column in range(size):
        diagonal[column] = exact_metric[column][column] - sum(
            (
                lower[column][inner] ** 2 * diagonal[inner]
                for inner in range(column)
            ),
            Fraction(0),
        )
        if diagonal[column] <= 0:
            raise ValueError("metric must be positive definite")
        for row in range(column + 1, size):
            numerator = exact_metric[row][column] - sum(
                (
                    lower[row][inner] * lower[column][inner] * diagonal[inner]
                    for inner in range(column)
                ),
                Fraction(0),
            )
            lower[row][column] = numerator / diagonal[column]

    def integer_candidates(center: Fraction, weight: Fraction, bound: Fraction) -> tuple[int, ...]:
        if bound < 0:
            return ()
        base = center.numerator // center.denominator
        values: list[int] = []
        integer = base
        while weight * (Fraction(integer) - center) ** 2 <= bound:
            values.append(integer)
            integer -= 1
        integer = base + 1
        while weight * (Fraction(integer) - center) ** 2 <= bound:
            values.append(integer)
            integer += 1
        return tuple(values)

    current = [0 for _ in range(size)]
    vectors: list[tuple[int, ...]] = []

    def recurse(index: int, partial: Fraction) -> None:
        if index < 0:
            if partial == target:
                vectors.append(tuple(current))
            return
        tail = sum(
            (lower[column][index] * current[column] for column in range(index + 1, size)),
            Fraction(0),
        )
        center = -tail
        remaining = target - partial
        for integer in integer_candidates(center, diagonal[index], remaining):
            current[index] = integer
            contribution = diagonal[index] * (Fraction(integer) - center) ** 2
            recurse(index - 1, partial + contribution)

    recurse(size - 1, Fraction(0))
    return tuple(sorted(set(vectors)))


@dataclass(frozen=True)
class PolarizedAutomorphismProblem:
    """Exact data defining the origin-fixing polarized automorphism search."""

    polarization: Polarization
    metric: Sequence[Sequence[int | Fraction]]
    complex_structure: Sequence[Sequence[int | Fraction]] | None = None

    def __post_init__(self) -> None:
        metric = rational_matrix(self.metric)
        if len(metric) != 2 * self.polarization.dimension:
            raise ValueError("metric and polarization dimensions do not agree")
        if metric != tuple(tuple(metric[column][row] for column in range(len(metric))) for row in range(len(metric))):
            raise ValueError("metric must be symmetric")
        if not is_positive_definite(metric):
            raise ValueError("metric must be positive definite")
        structure = None if self.complex_structure is None else rational_matrix(self.complex_structure)
        if structure is not None and len(structure) != len(metric):
            raise ValueError("complex structure and metric dimensions do not agree")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "complex_structure", structure)


@dataclass(frozen=True)
class PolarizedAutomorphismGroup:
    """The fully enumerated finite group of exact polarized automorphisms."""

    problem: PolarizedAutomorphismProblem
    elements: tuple[IntegerMatrix, ...]

    @property
    def order(self) -> int:
        return len(self.elements)


def enumerate_polarized_automorphisms(
    problem: PolarizedAutomorphismProblem,
) -> PolarizedAutomorphismGroup:
    """Enumerate every integral automorphism preserving the supplied data.

    Columns are constructed recursively from their exact Gram constraints.
    This Phase 1 backend is intended for dimensions 2, 4, and 6 with modest
    exact Gram forms. Later phases can add a faster lattice-isometry backend
    without changing the returned representation.
    """

    metric = problem.metric
    size = len(metric)
    candidates_by_norm = {
        metric[index][index]: integer_vectors_of_norm(metric, metric[index][index])
        for index in range(size)
    }
    integral_metric = all(value.denominator == 1 for row in metric for value in row)
    if integral_metric:
        metric_integer = tuple(tuple(value.numerator for value in row) for row in metric)
        candidate_images = {
            candidate: tuple(
                sum(metric_integer[row][column] * candidate[column] for column in range(size))
                for row in range(size)
            )
            for candidates in candidates_by_norm.values()
            for candidate in candidates
        }

        def pairing(left: tuple[int, ...], right: tuple[int, ...]) -> int:
            image = candidate_images[right]
            return sum(left[row] * image[row] for row in range(size))

    else:
        candidate_images = {
            candidate: tuple(
                sum(
                    (metric[row][column] * candidate[column] for column in range(size)),
                    Fraction(0),
                )
                for row in range(size)
            )
            for candidates in candidates_by_norm.values()
            for candidate in candidates
        }

        def pairing(left: tuple[int, ...], right: tuple[int, ...]) -> Fraction:
            image = candidate_images[right]
            return sum(
                (Fraction(left[row]) * image[row] for row in range(size)),
                Fraction(0),
            )

    columns: list[tuple[int, ...]] = []
    automorphisms: list[IntegerMatrix] = []

    def extend(column_index: int) -> None:
        if column_index == size:
            matrix = columns_to_matrix(columns)
            if abs(determinant(matrix)) != 1:
                return
            if congruence(matrix, metric) != metric:
                return
            if congruence(matrix, problem.polarization.matrix) != rational_matrix(problem.polarization.matrix):
                return
            if problem.complex_structure is not None and not commute(matrix, problem.complex_structure):
                return
            automorphisms.append(matrix)
            return

        for candidate in candidates_by_norm[metric[column_index][column_index]]:
            if all(
                pairing(columns[previous], candidate) == metric[previous][column_index]
                for previous in range(column_index)
            ):
                columns.append(candidate)
                extend(column_index + 1)
                columns.pop()

    extend(0)
    unique = tuple(sorted(set(automorphisms)))
    if not unique:
        raise ArithmeticError("no polarized automorphisms were found; identity should always survive")
    return PolarizedAutomorphismGroup(problem=problem, elements=unique)
