"""Small exact linear-algebra helpers used by the Phase 1 engine."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence


Scalar = int | Fraction
RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]
IntegerMatrix = tuple[tuple[int, ...], ...]


def rational_matrix(matrix: Sequence[Sequence[Scalar]]) -> RationalMatrix:
    rows = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("matrix must be nonempty and square")
    return rows


def integer_matrix(matrix: Sequence[Sequence[int]]) -> IntegerMatrix:
    rows = tuple(tuple(int(value) for value in row) for row in matrix)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("matrix must be nonempty and square")
    return rows


def transpose(matrix: Sequence[Sequence[Scalar]]) -> RationalMatrix:
    rows = rational_matrix(matrix)
    return tuple(tuple(rows[row][column] for row in range(len(rows))) for column in range(len(rows)))


def matmul(
    left: Sequence[Sequence[Scalar]], right: Sequence[Sequence[Scalar]]
) -> RationalMatrix:
    left_rows = rational_matrix(left)
    right_rows = rational_matrix(right)
    if len(left_rows) != len(right_rows):
        raise ValueError("matrix dimensions do not agree")
    size = len(left_rows)
    return tuple(
        tuple(
            sum(
                (left_rows[row][inner] * right_rows[inner][column] for inner in range(size)),
                Fraction(0),
            )
            for column in range(size)
        )
        for row in range(size)
    )


def matvec(
    matrix: Sequence[Sequence[Scalar]], vector: Sequence[Scalar]
) -> RationalVector:
    rows = rational_matrix(matrix)
    if len(vector) != len(rows):
        raise ValueError("matrix and vector dimensions do not agree")
    return tuple(
        sum((value * Fraction(coordinate) for value, coordinate in zip(row, vector)), Fraction(0))
        for row in rows
    )


def bilinear(
    left: Sequence[Scalar],
    matrix: Sequence[Sequence[Scalar]],
    right: Sequence[Scalar],
) -> Fraction:
    image = matvec(matrix, right)
    return sum((Fraction(value) * coordinate for value, coordinate in zip(left, image)), Fraction(0))


def columns_to_matrix(columns: Sequence[Sequence[int]]) -> IntegerMatrix:
    if not columns:
        raise ValueError("at least one column is required")
    size = len(columns)
    if any(len(column) != size for column in columns):
        raise ValueError("columns must define a square matrix")
    return tuple(tuple(int(columns[column][row]) for column in range(size)) for row in range(size))


def commute(
    left: Sequence[Sequence[Scalar]], right: Sequence[Sequence[Scalar]]
) -> bool:
    return matmul(left, right) == matmul(right, left)


def congruence(
    transformation: Sequence[Sequence[int]],
    form: Sequence[Sequence[Scalar]],
) -> RationalMatrix:
    return matmul(matmul(transpose(transformation), form), transformation)


def identity_matrix(size: int) -> IntegerMatrix:
    return tuple(tuple(int(row == column) for column in range(size)) for row in range(size))


def determinant_rational(matrix: Sequence[Sequence[Scalar]]) -> Fraction:
    """Return an exact determinant by rational Gaussian elimination."""

    work = [list(row) for row in rational_matrix(matrix)]
    size = len(work)
    sign = 1
    result = Fraction(1)
    for column in range(size):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign *= -1
        pivot = work[column][column]
        result *= pivot
        for row in range(column + 1, size):
            coefficient = work[row][column] / pivot
            for inner in range(column + 1, size):
                work[row][inner] -= coefficient * work[column][inner]
    return sign * result


def is_positive_definite(matrix: Sequence[Sequence[Scalar]]) -> bool:
    """Apply Sylvester's criterion exactly to a symmetric rational matrix."""

    rows = rational_matrix(matrix)
    if rows != tuple(
        tuple(rows[column][row] for column in range(len(rows)))
        for row in range(len(rows))
    ):
        return False
    return all(
        determinant_rational(tuple(tuple(rows[row][column] for column in range(order)) for row in range(order))) > 0
        for order in range(1, len(rows) + 1)
    )


def fractional_part(value: Fraction) -> Fraction:
    return value - value.numerator // value.denominator


def canonical_mod_integer(vector: Iterable[Scalar]) -> RationalVector:
    return tuple(fractional_part(Fraction(value)) for value in vector)
