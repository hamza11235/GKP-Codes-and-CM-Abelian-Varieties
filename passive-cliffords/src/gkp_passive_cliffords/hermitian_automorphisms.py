"""Fast exact automorphism enumeration for CM-product Hermitian forms."""

from __future__ import annotations

from typing import Protocol, Sequence

from gkp_systole.polarization import Polarization
from gkp_systole.quadratic_hermitian import Element, ImaginaryQuadraticOrder

from .automorphisms import (
    PolarizedAutomorphismGroup,
    PolarizedAutomorphismProblem,
    integer_vectors_of_norm,
)
from .exact import IntegerMatrix


class HermitianCMForm(Protocol):
    order: ImaginaryQuadraticOrder

    @property
    def hermitian_matrix(self) -> Sequence[Sequence[Element]]: ...

    @property
    def metric_core(self) -> Sequence[Sequence[int]]: ...

    @property
    def alternating(self) -> Sequence[Sequence[int]]: ...

    @property
    def complex_structure_numerator(self) -> Sequence[Sequence[int]]: ...

    def validate(self) -> None: ...


HermitianVector = tuple[Element, ...]


def _integer_matmul(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> IntegerMatrix:
    """Multiply integral matrices without promoting entries to Fraction."""

    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    return tuple(
        tuple(
            sum(int(left[row][index]) * int(right[index][column]) for index in range(inner))
            for column in range(columns)
        )
        for row in range(rows)
    )


def _integer_transpose(matrix: Sequence[Sequence[int]]) -> IntegerMatrix:
    return tuple(tuple(int(matrix[row][column]) for row in range(len(matrix))) for column in range(len(matrix[0])))


def _integer_congruence(
    matrix: Sequence[Sequence[int]], form: Sequence[Sequence[int]]
) -> IntegerMatrix:
    return _integer_matmul(_integer_matmul(_integer_transpose(matrix), form), matrix)


def _integer_commutes(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> bool:
    return _integer_matmul(left, right) == _integer_matmul(right, left)


def _bareiss_determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Exact fraction-free determinant for a small integral matrix."""

    values = [list(map(int, row)) for row in matrix]
    size = len(values)
    if size == 1:
        return values[0][0]
    sign = 1
    denominator = 1
    for pivot_index in range(size - 1):
        pivot_row = next(
            (row for row in range(pivot_index, size) if values[row][pivot_index] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            values[pivot_index], values[pivot_row] = values[pivot_row], values[pivot_index]
            sign = -sign
        pivot = values[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = values[row][column] * pivot - values[row][pivot_index] * values[pivot_index][column]
                if numerator % denominator:
                    raise ArithmeticError("Bareiss division was not exact")
                values[row][column] = numerator // denominator
        denominator = pivot
        for row in range(pivot_index + 1, size):
            values[row][pivot_index] = 0
    return sign * values[-1][-1]


def _hermitian_pairing(
    order: ImaginaryQuadraticOrder,
    hermitian: Sequence[Sequence[Element]],
    left: HermitianVector,
    right: HermitianVector,
) -> Element:
    value = (0, 0)
    for row in range(len(left)):
        for column in range(len(right)):
            term = order.multiply(
                order.multiply(order.conjugate(left[row]), hermitian[row][column]),
                right[column],
            )
            value = order.add(value, term)
    return value


def _coordinates_to_hermitian_vector(
    coordinates: tuple[int, ...], rank: int
) -> HermitianVector:
    return tuple((coordinates[index], coordinates[rank + index]) for index in range(rank))


def _hermitian_columns_to_real_matrix(
    order: ImaginaryQuadraticOrder,
    columns: Sequence[HermitianVector],
) -> IntegerMatrix:
    """Realize an O-linear matrix in the basis ``(e_i, omega e_i)``."""

    rank = len(columns)
    trace = order.basis_trace
    norm = order.basis_norm
    result = [[0 for _ in range(2 * rank)] for _ in range(2 * rank)]
    for column, vector in enumerate(columns):
        for row, (first, second) in enumerate(vector):
            result[row][column] = first
            result[rank + row][column] = second
            result[row][rank + column] = -norm * second
            result[rank + row][rank + column] = first + trace * second
    return tuple(tuple(row) for row in result)


def enumerate_hermitian_cm_automorphisms(
    form: HermitianCMForm,
) -> PolarizedAutomorphismGroup:
    """Enumerate ``U(H,O)`` and return its exact real-lattice action.

    The search chooses only ``rank`` Hermitian columns rather than ``2*rank``
    independent real columns.  Holomorphicity is therefore built in, making
    the ternary CM benchmarks substantially faster than the generic backend.
    Every resulting real matrix is nevertheless checked against ``G``, ``A``,
    and ``J`` before it is returned.
    """

    form.validate()
    order = form.order
    complex_structure = getattr(form, "complex_structure_numerator", None)
    if complex_structure is None:
        complex_structure = order.complex_structure_numerator
    raw_hermitian = getattr(form, "hermitian_matrix", None)
    if raw_hermitian is None:
        # Binary ``QuadraticHermitianForm`` exposes its entries individually.
        off_diagonal = form.off_diagonal
        raw_hermitian = (
            ((form.a, 0), off_diagonal),
            (order.conjugate(off_diagonal), (form.c, 0)),
        )
    hermitian = tuple(tuple(value for value in row) for row in raw_hermitian)
    rank = len(hermitian)
    if any(len(row) != rank for row in hermitian):
        raise ValueError("Hermitian matrix must be square")
    diagonal = tuple(hermitian[index][index] for index in range(rank))
    if any(value[1] != 0 or value[0] <= 0 for value in diagonal):
        raise ValueError("Hermitian diagonal must consist of positive integers")

    candidates_by_norm: dict[int, tuple[HermitianVector, ...]] = {}
    for value in {entry[0] for entry in diagonal}:
        real_vectors = integer_vectors_of_norm(form.metric_core, 2 * value)
        candidates_by_norm[value] = tuple(
            _coordinates_to_hermitian_vector(vector, rank) for vector in real_vectors
        )

    columns: list[HermitianVector] = []
    real_automorphisms: list[IntegerMatrix] = []

    def extend(column_index: int) -> None:
        if column_index == rank:
            matrix = _hermitian_columns_to_real_matrix(order, columns)
            if abs(_bareiss_determinant(matrix)) != 1:
                return
            if _integer_congruence(matrix, form.metric_core) != tuple(
                tuple(int(value) for value in row) for row in form.metric_core
            ):
                return
            if _integer_congruence(matrix, form.alternating) != tuple(
                tuple(int(value) for value in row) for row in form.alternating
            ):
                return
            if not _integer_commutes(matrix, complex_structure):
                return
            real_automorphisms.append(matrix)
            return

        for candidate in candidates_by_norm[diagonal[column_index][0]]:
            if all(
                _hermitian_pairing(order, hermitian, columns[previous], candidate)
                == hermitian[previous][column_index]
                for previous in range(column_index)
            ):
                columns.append(candidate)
                extend(column_index + 1)
                columns.pop()

    extend(0)
    elements = tuple(sorted(set(real_automorphisms)))
    if not elements:
        raise ArithmeticError("Hermitian automorphism enumeration found no identity")
    problem = PolarizedAutomorphismProblem(
        polarization=Polarization(form.alternating),
        metric=form.metric_core,
        complex_structure=complex_structure,
    )
    return PolarizedAutomorphismGroup(problem=problem, elements=elements)
