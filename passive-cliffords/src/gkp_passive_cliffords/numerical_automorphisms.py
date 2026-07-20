"""Numerical integral-automorphism enumeration for generic real controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import Sequence

import numpy as np

from gkp_systole.polarization import Polarization, determinant

from .exact import IntegerMatrix, columns_to_matrix


FloatMatrix = tuple[tuple[float, ...], ...]


def _float_matrix(matrix: Sequence[Sequence[int | float]]) -> FloatMatrix:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be nonempty and square")
    return tuple(tuple(float(value) for value in row) for row in array)


def numerical_integer_vectors_of_norm(
    metric: Sequence[Sequence[int | float]],
    target_norm: float,
    *,
    tolerance: float = 1e-9,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate integral vectors at one floating-point squared norm."""

    g = np.asarray(metric, dtype=float)
    if target_norm <= 0 or tolerance <= 0:
        raise ValueError("target norm and tolerance must be positive")
    upper = np.linalg.cholesky(g).T
    size = g.shape[0]
    current = [0 for _ in range(size)]
    vectors: list[tuple[int, ...]] = []
    slack = tolerance * max(1.0, abs(target_norm), float(np.max(np.abs(g)))) * 20.0

    def recurse(index: int, partial: float) -> None:
        if index < 0:
            vector = tuple(current)
            value = float(np.asarray(vector) @ g @ np.asarray(vector))
            if abs(value - target_norm) <= slack:
                vectors.append(vector)
            return
        tail = sum(upper[index, column] * current[column] for column in range(index + 1, size))
        center = -tail / upper[index, index]
        remaining = target_norm + slack - partial
        if remaining < 0:
            return
        radius = sqrt(remaining) / abs(upper[index, index])
        for integer in range(ceil(center - radius), floor(center + radius) + 1):
            current[index] = integer
            row_value = upper[index, index] * integer + tail
            recurse(index - 1, partial + row_value * row_value)

    recurse(size - 1, 0.0)
    return tuple(sorted(set(vectors)))


@dataclass(frozen=True)
class NumericalPolarizedAutomorphismProblem:
    polarization: Polarization
    metric: Sequence[Sequence[int | float]]
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        metric = _float_matrix(self.metric)
        if len(metric) != 2 * self.polarization.dimension:
            raise ValueError("metric and polarization dimensions do not agree")
        array = np.asarray(metric)
        if np.max(np.abs(array - array.T)) > self.tolerance:
            raise ValueError("metric must be symmetric")
        if float(np.min(np.linalg.eigvalsh(array))) <= 0:
            raise ValueError("metric must be positive definite")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(self, "metric", metric)


@dataclass(frozen=True)
class NumericalPolarizedAutomorphismGroup:
    problem: NumericalPolarizedAutomorphismProblem
    elements: tuple[IntegerMatrix, ...]
    maximum_metric_residual: float

    @property
    def order(self) -> int:
        return len(self.elements)


def enumerate_numerical_polarized_automorphisms(
    problem: NumericalPolarizedAutomorphismProblem,
) -> NumericalPolarizedAutomorphismGroup:
    """Enumerate integral isometries of a generic floating compatible metric."""

    metric = np.asarray(problem.metric, dtype=float)
    alternating = np.asarray(problem.polarization.matrix, dtype=int)
    size = metric.shape[0]
    scale = max(1.0, float(np.max(np.abs(metric))))
    tolerance = problem.tolerance * scale
    candidates_by_norm = {
        float(metric[index, index]): numerical_integer_vectors_of_norm(
            metric,
            float(metric[index, index]),
            tolerance=problem.tolerance,
        )
        for index in range(size)
    }
    candidate_images = {
        candidate: metric @ np.asarray(candidate, dtype=float)
        for candidates in candidates_by_norm.values()
        for candidate in candidates
    }
    columns: list[tuple[int, ...]] = []
    elements: list[IntegerMatrix] = []
    residuals: list[float] = []

    def extend(column_index: int) -> None:
        if column_index == size:
            matrix = columns_to_matrix(columns)
            if abs(determinant(matrix)) != 1:
                return
            m = np.asarray(matrix, dtype=int)
            if not np.array_equal(m.T @ alternating @ m, alternating):
                return
            residual = float(np.max(np.abs(m.T @ metric @ m - metric)))
            if residual > tolerance * 20.0:
                return
            elements.append(matrix)
            residuals.append(residual)
            return

        for candidate in candidates_by_norm[float(metric[column_index, column_index])]:
            image = candidate_images[candidate]
            if all(
                abs(float(np.asarray(columns[previous]) @ image) - metric[previous, column_index])
                <= tolerance * 20.0
                for previous in range(column_index)
            ):
                columns.append(candidate)
                extend(column_index + 1)
                columns.pop()

    extend(0)
    unique = tuple(sorted(set(elements)))
    if not unique:
        raise ArithmeticError("numerical automorphism enumeration lost the identity")
    return NumericalPolarizedAutomorphismGroup(
        problem=problem,
        elements=unique,
        maximum_metric_residual=max(residuals, default=0.0),
    )
