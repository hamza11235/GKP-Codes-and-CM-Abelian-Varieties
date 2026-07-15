"""Shortest-vector shortcut for uniform polarization types.

For a type ``(d,...,d)`` lattice, ``Lambda^perp=(1/d)Lambda``.  Hence the
relative systole is obtained from one shortest-vector problem on the
principal lattice instead of a closest-vector problem for every kernel class.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, sqrt
from typing import Sequence

from .benchmarks import canonical_alternating
from .conventions import (
    MetricConvention,
    NormalizationMetadata,
    coerce_metric_convention,
    uniform_metric,
)
from .metric import Metric, Scalar
from .polarization import Polarization


IntegerVector = tuple[int, ...]


def _exact_ldlt(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    size = len(matrix)
    lower = [
        [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]
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


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _exact_integer_candidates(
    center: Fraction,
    weight: Fraction,
    bound: Fraction,
) -> tuple[int, ...]:
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


@dataclass(frozen=True)
class ShortestVectorResult:
    """All nonzero shortest vectors of a lattice metric."""

    metric: Metric
    squared_length: Scalar
    vectors: tuple[IntegerVector, ...]
    certified: bool
    method: str

    @property
    def length(self) -> float:
        return sqrt(float(self.squared_length))

    @property
    def multiplicity(self) -> int:
        return len(self.vectors)


def _exact_shortest_vectors(metric: Metric) -> ShortestVectorResult:
    lower, diagonal = _exact_ldlt(metric.matrix)
    size = metric.dimension
    current = [0 for _ in range(size)]
    diagonal_entries = tuple(Fraction(metric.matrix[i][i]) for i in range(size))
    best = min(diagonal_entries)
    best_vectors: list[IntegerVector] = []

    def recurse(index: int, partial: Fraction) -> None:
        nonlocal best, best_vectors
        if index < 0:
            vector = tuple(current)
            if not any(vector):
                return
            if partial < best:
                best = partial
                best_vectors = [vector]
            elif partial == best:
                best_vectors.append(vector)
            return

        tail = sum(
            (
                lower[column][index] * current[column]
                for column in range(index + 1, size)
            ),
            Fraction(0),
        )
        center = -tail
        remaining = best - partial
        for integer in _exact_integer_candidates(center, diagonal[index], remaining):
            current[index] = integer
            contribution = diagonal[index] * (Fraction(integer) - center) ** 2
            recurse(index - 1, partial + contribution)

    recurse(size - 1, Fraction(0))
    vectors = tuple(sorted(set(best_vectors)))
    if not vectors:
        raise ArithmeticError("SVP enumeration found no nonzero lattice vector")
    return ShortestVectorResult(
        metric=metric,
        squared_length=best,
        vectors=vectors,
        certified=True,
        method="exact_ldlt_svp_branch_and_bound",
    )


def _float_shortest_vectors(metric: Metric, tolerance: float) -> ShortestVectorResult:
    upper_factor = metric.cholesky_upper
    size = metric.dimension
    current = [0 for _ in range(size)]
    best = min(float(metric.matrix[i][i]) for i in range(size))
    best_vectors: list[IntegerVector] = []

    def recurse(index: int, partial: float) -> None:
        nonlocal best, best_vectors
        if index < 0:
            vector = tuple(current)
            if not any(vector):
                return
            squared = float(metric.quadratic(vector))
            scale = max(1.0, abs(best), abs(squared))
            if squared < best - tolerance * scale:
                best = squared
                best_vectors = [vector]
            elif abs(squared - best) <= tolerance * scale:
                best_vectors.append(vector)
            return

        remaining = max(0.0, best - partial)
        tail = sum(
            upper_factor[index][column] * current[column]
            for column in range(index + 1, size)
        )
        diagonal = upper_factor[index][index]
        center = -tail / diagonal
        radius = sqrt(remaining) / abs(diagonal)
        lower_integer = ceil(center - radius - 10 * tolerance)
        upper_integer = floor(center + radius + 10 * tolerance)
        candidates = sorted(
            range(lower_integer, upper_integer + 1),
            key=lambda value: abs(value - center),
        )
        for integer in candidates:
            current[index] = integer
            row_value = diagonal * integer + tail
            new_partial = partial + row_value * row_value
            if new_partial <= best + tolerance * max(1.0, best):
                recurse(index - 1, new_partial)

    recurse(size - 1, 0.0)
    vectors = tuple(sorted(set(best_vectors)))
    if not vectors:
        raise ArithmeticError("SVP enumeration found no nonzero lattice vector")
    return ShortestVectorResult(
        metric=metric,
        squared_length=best,
        vectors=vectors,
        certified=False,
        method="cholesky_svp_branch_and_bound",
    )


def shortest_lattice_vectors(
    metric: Metric | Sequence[Sequence[int | float | Fraction]],
    *,
    tolerance: float = 1e-12,
) -> ShortestVectorResult:
    """Compute all nonzero shortest integer-coordinate lattice vectors."""

    metric_object = metric if isinstance(metric, Metric) else Metric(metric)
    if metric_object.is_exact:
        return _exact_shortest_vectors(metric_object)
    return _float_shortest_vectors(metric_object, tolerance)


@dataclass(frozen=True)
class UniformRelativeSystoleResult:
    """Relative systole obtained from the uniform-type SVP identity."""

    level: int
    principal_metric: Metric
    metric: Metric
    metric_convention: MetricConvention
    shortest_vector_result: ShortestVectorResult

    @property
    def dimension(self) -> int:
        return self.metric.dimension // 2

    @property
    def polarization(self) -> Polarization:
        return Polarization(canonical_alternating((self.level,) * self.dimension))

    @property
    def lambda1_squared(self) -> Scalar:
        """The principal metric's shortest-vector length squared."""

        return self.shortest_vector_result.squared_length

    @property
    def squared_systole(self) -> Scalar:
        denominator = (
            self.level * self.level
            if self.metric_convention is MetricConvention.FIXED_PRINCIPAL
            else self.level
        )
        return self.lambda1_squared / denominator

    @property
    def systole(self) -> float:
        return sqrt(float(self.squared_systole))

    @property
    def minimal_vectors(self) -> tuple[IntegerVector, ...]:
        return self.shortest_vector_result.vectors

    @property
    def lift_multiplicity(self) -> int:
        return self.shortest_vector_result.multiplicity

    @property
    def class_multiplicity(self) -> int:
        classes = {
            tuple(coordinate % self.level for coordinate in vector)
            for vector in self.minimal_vectors
        }
        classes.discard((0,) * self.metric.dimension)
        return len(classes)

    @property
    def certified(self) -> bool:
        return self.shortest_vector_result.certified

    @property
    def method(self) -> str:
        return self.shortest_vector_result.method

    @property
    def normalization(self) -> NormalizationMetadata:
        return NormalizationMetadata(
            metric_convention=self.metric_convention,
            dimension_g=self.dimension,
            polarization_type=(self.level,) * self.dimension,
            metric_determinant=self.metric.determinant,
        )

    def normalization_record(self) -> dict[str, object]:
        return {
            **self.normalization.as_dict(),
            "ell_squared": self.squared_systole,
            "lambda1_squared": self.lambda1_squared,
        }


def compute_uniform_relative_systole(
    principal_metric: Metric | Sequence[Sequence[int | float | Fraction]],
    level: int,
    *,
    metric_convention: MetricConvention | str = MetricConvention.FIXED_PRINCIPAL,
    tolerance: float = 1e-12,
) -> UniformRelativeSystoleResult:
    """Compute a uniform type relative systole using one principal SVP.

    The input metric is always the principal metric. The returned ``metric``
    is the physical metric selected by ``metric_convention``.
    """

    if level <= 1:
        raise ValueError("uniform polarization level must be greater than one")
    principal = principal_metric if isinstance(principal_metric, Metric) else Metric(principal_metric)
    if principal.dimension % 2:
        raise ValueError("a polarized phase-space metric must have even dimension")
    convention = coerce_metric_convention(metric_convention)
    selected_metric = uniform_metric(principal, level, convention)
    shortest = shortest_lattice_vectors(principal, tolerance=tolerance)
    return UniformRelativeSystoleResult(
        level=level,
        principal_metric=principal,
        metric=selected_metric,
        metric_convention=convention,
        shortest_vector_result=shortest,
    )
