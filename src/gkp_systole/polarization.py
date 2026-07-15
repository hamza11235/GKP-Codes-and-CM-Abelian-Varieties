"""Exact validation and type extraction for integral polarization matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations
from math import gcd
from numbers import Integral
from typing import Sequence


IntMatrix = tuple[tuple[int, ...], ...]


class PolarizationError(ValueError):
    """Raised when a matrix cannot represent a full polarization form."""


def _as_integer_matrix(matrix: Sequence[Sequence[int]]) -> IntMatrix:
    rows = tuple(tuple(row) for row in matrix)
    if not rows:
        raise PolarizationError("the polarization matrix must be nonempty")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise PolarizationError("the polarization matrix must be square")
    if any(not isinstance(value, Integral) for row in rows for value in row):
        raise PolarizationError("the polarization matrix must have integer entries")
    return tuple(tuple(int(value) for value in row) for row in rows)


def determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Return an exact determinant using fraction-free Bareiss elimination."""

    work = [list(map(int, row)) for row in matrix]
    size = len(work)
    if size == 0:
        return 1
    if any(len(row) != size for row in work):
        raise ValueError("determinant requires a square matrix")
    if size == 1:
        return work[0][0]

    sign = 1
    previous_pivot = 1
    for pivot_index in range(size - 1):
        if work[pivot_index][pivot_index] == 0:
            swap_index = next(
                (
                    row_index
                    for row_index in range(pivot_index + 1, size)
                    if work[row_index][pivot_index] != 0
                ),
                None,
            )
            if swap_index is None:
                return 0
            work[pivot_index], work[swap_index] = (
                work[swap_index],
                work[pivot_index],
            )
            sign *= -1

        pivot = work[pivot_index][pivot_index]
        for row_index in range(pivot_index + 1, size):
            for column_index in range(pivot_index + 1, size):
                numerator = (
                    work[row_index][column_index] * pivot
                    - work[row_index][pivot_index]
                    * work[pivot_index][column_index]
                )
                work[row_index][column_index] = numerator // previous_pivot
        previous_pivot = pivot

    return sign * work[-1][-1]


def _minor_gcd(matrix: IntMatrix, order: int) -> int:
    """Return the gcd of all minors of the requested order."""

    size = len(matrix)
    result = 0
    index_sets = tuple(combinations(range(size), order))
    for row_indices in index_sets:
        for column_indices in index_sets:
            minor = tuple(
                tuple(matrix[row][column] for column in column_indices)
                for row in row_indices
            )
            result = gcd(result, abs(determinant(minor)))
            if result == 1:
                return 1
    return result


def smith_invariant_factors(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Compute Smith invariant factors from determinantal divisors.

    This exact, dependency-free implementation is intended for the small
    matrices (orders 2, 4, and 6) used in the initial project. It computes the
    gcd of all k-by-k minors. A specialized Smith normal form backend can
    replace it later for larger dimensions.
    """

    integer_matrix = _as_integer_matrix(matrix)
    size = len(integer_matrix)
    divisors = [1]
    for order in range(1, size + 1):
        divisor = _minor_gcd(integer_matrix, order)
        if divisor == 0:
            raise PolarizationError("the polarization matrix must be nonsingular")
        divisors.append(divisor)

    factors = tuple(
        divisors[index] // divisors[index - 1]
        for index in range(1, len(divisors))
    )
    if any(right % left for left, right in zip(factors, factors[1:])):
        raise ArithmeticError("computed Smith factors do not form a divisibility chain")
    return factors


@dataclass(frozen=True)
class Polarization:
    """An integral, nondegenerate alternating form in a lattice basis."""

    matrix: Sequence[Sequence[int]]
    _integer_matrix: IntMatrix = field(init=False, repr=False)
    _determinant: int = field(init=False, repr=False)
    _smith_factors: tuple[int, ...] = field(init=False, repr=False)
    _type: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matrix = _as_integer_matrix(self.matrix)
        size = len(matrix)
        if size % 2:
            raise PolarizationError("an alternating polarization matrix has even order")
        if any(matrix[row][row] != 0 for row in range(size)):
            raise PolarizationError("an alternating matrix must have zero diagonal")
        if any(
            matrix[row][column] != -matrix[column][row]
            for row in range(size)
            for column in range(size)
        ):
            raise PolarizationError("the polarization matrix must be alternating")

        matrix_determinant = determinant(matrix)
        if matrix_determinant == 0:
            raise PolarizationError("the polarization matrix must be nonsingular")

        factors = smith_invariant_factors(matrix)
        if any(factors[index] != factors[index + 1] for index in range(0, size, 2)):
            raise PolarizationError(
                "Smith invariant factors of an alternating form must occur in pairs"
            )
        polarization_type = tuple(factors[index] for index in range(0, size, 2))
        if any(
            right % left
            for left, right in zip(polarization_type, polarization_type[1:])
        ):
            raise PolarizationError("polarization type must form a divisibility chain")

        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "_integer_matrix", matrix)
        object.__setattr__(self, "_determinant", matrix_determinant)
        object.__setattr__(self, "_smith_factors", factors)
        object.__setattr__(self, "_type", polarization_type)

    @property
    def dimension(self) -> int:
        """Complex dimension of the associated polarized torus."""

        return len(self._integer_matrix) // 2

    @property
    def determinant(self) -> int:
        return self._determinant

    @property
    def smith_factors(self) -> tuple[int, ...]:
        return self._smith_factors

    @property
    def type(self) -> tuple[int, ...]:
        return self._type

    @property
    def kernel_order(self) -> int:
        """Order of Lambda-perp/Lambda, equal to abs(det(A))."""

        return abs(self._determinant)

    def verify_kernel_order(self) -> bool:
        expected = reduce(lambda left, right: left * right, self._type, 1) ** 2
        return self.kernel_order == expected
