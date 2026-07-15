"""Exact reconstruction of the Phase-7 type-(1,5) numerical record."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Sequence

from .conventions import MetricConvention
from .kernel import invert_rational_matrix
from .models import D4_ROOT_GRAM
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import RelativeSystoleResult, compute_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


TYPE_15_ALTERNATING = (
    (0, -2, 1, -1),
    (2, 0, -2, 1),
    (-1, 2, 0, -2),
    (1, -1, 2, 0),
)

TYPE_15_METRIC_CORE = (
    (90, -115, 120, -115),
    (-115, 150, -160, 150),
    (120, -160, 220, -190),
    (-115, 150, -190, 170),
)

TYPE_15_COMPLEX_STRUCTURE_NUMERATOR = (
    (68, -88, 96, -90),
    (35, -44, 42, -42),
    (41, -53, 68, -61),
    (61, -80, 104, -92),
)

TYPE_15_DUAL_CORE = (
    (52, 26, 32, 48),
    (26, 14, 15, 22),
    (32, 15, 22, 33),
    (48, 22, 33, 50),
)

TYPE_15_D4_CHANGE = (
    (-4, 1, 0, 1),
    (5, -1, 0, -2),
    (-5, -1, 3, 3),
    (5, 0, -2, -2),
)

TYPE_15_CM_ISOGENY_CHANGE = (
    (1, 68, 0, -88),
    (0, 35, 1, -44),
    (0, 41, 0, -53),
    (0, 61, 0, -80),
)

TYPE_15_CM_BLOCK = (
    (0, -10, 0, 0),
    (1, 0, 0, 0),
    (0, 0, 0, -10),
    (0, 0, 1, 0),
)


def _fraction_matrix(matrix) -> RationalMatrix:
    return tuple(tuple(Fraction(value) for value in row) for row in matrix)


def _transpose(matrix: Sequence[Sequence[Fraction | int]]) -> RationalMatrix:
    rows = _fraction_matrix(matrix)
    return tuple(tuple(rows[row][column] for row in range(len(rows))) for column in range(len(rows[0])))


def _multiply(left, right) -> RationalMatrix:
    a = _fraction_matrix(left)
    b = _fraction_matrix(right)
    return tuple(
        tuple(
            sum((a[row][inner] * b[inner][column] for inner in range(len(b))), Fraction(0))
            for column in range(len(b[0]))
        )
        for row in range(len(a))
    )


def _rank(matrix: Sequence[Sequence[Fraction | int]]) -> int:
    work = [list(map(Fraction, row)) for row in matrix]
    if not work:
        return 0
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or work[row][column] == 0:
                continue
            coefficient = work[row][column]
            work[row] = [value - coefficient * pivot_value for value, pivot_value in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def _one_dimensional_nullspace(matrix: Sequence[Sequence[Fraction | int]]) -> tuple[Fraction, ...]:
    work = [list(map(Fraction, row)) for row in matrix]
    rows = len(work)
    columns = len(work[0])
    pivots = []
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or work[row][column] == 0:
                continue
            coefficient = work[row][column]
            work[row] = [value - coefficient * pivot_value for value, pivot_value in zip(work[row], work[pivot_row])]
        pivots.append(column)
        pivot_row += 1
    free = [column for column in range(columns) if column not in pivots]
    if len(free) != 1:
        raise ArithmeticError(f"expected a one-dimensional nullspace, found {len(free)}")
    result = [Fraction(0) for _ in range(columns)]
    result[free[0]] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        result[pivot] = -sum(
            (work[row][column] * result[column] for column in free),
            Fraction(0),
        )
    return tuple(result)


def reconstruct_equal_distance_metric(
    lifts: Sequence[Sequence[Fraction | int]],
) -> RationalMatrix:
    """Recover ``G/ell^2`` by imposing equal length on active lifts."""

    if not lifts or any(len(vector) != 4 for vector in lifts):
        raise ValueError("the reconstruction requires nonempty four-dimensional lifts")
    pairs = tuple((row, column) for row in range(4) for column in range(row, 4))
    equations = []
    for raw_vector in lifts:
        vector = tuple(Fraction(value) for value in raw_vector)
        row = [
            vector[left] * vector[right] * (2 if left != right else 1)
            for left, right in pairs
        ]
        equations.append(tuple(row + [Fraction(-1)]))
    null_vector = _one_dimensional_nullspace(equations)
    if null_vector[-1] == 0:
        raise ArithmeticError("equal-distance equations did not determine the distance scale")
    normalized = tuple(value / null_vector[-1] for value in null_vector)
    matrix = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    for value, (row, column) in zip(normalized[:-1], pairs):
        matrix[row][column] = matrix[column][row] = value
    return tuple(tuple(row) for row in matrix)


def rational_commutant_dimension(matrix: Sequence[Sequence[int]]) -> int:
    """Dimension over Q of matrices commuting with the supplied matrix."""

    source = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    size = len(source)
    equations = []
    for row in range(size):
        for column in range(size):
            equation = [Fraction(0) for _ in range(size * size)]
            for inner in range(size):
                equation[row * size + inner] += source[inner][column]
                equation[inner * size + column] -= source[row][inner]
            equations.append(tuple(equation))
    return size * size - _rank(equations)


@dataclass(frozen=True)
class Type15CMCertificate:
    field: str
    order_discriminant: int
    elliptic_period: str
    commutant_dimension: int
    rational_isogeny_degree: int
    block_matrix: tuple[tuple[int, ...], ...]
    is_cm: bool


@dataclass(frozen=True)
class Type15ExactModel:
    """Exact scaled model ``G=G_core/sqrt(10)``, ``J=S/sqrt(10)``."""

    alternating: tuple[tuple[int, ...], ...] = TYPE_15_ALTERNATING
    metric_core: tuple[tuple[int, ...], ...] = TYPE_15_METRIC_CORE
    complex_structure_numerator: tuple[tuple[int, ...], ...] = TYPE_15_COMPLEX_STRUCTURE_NUMERATOR
    scale_radicand: int = 10

    @property
    def polarization(self) -> Polarization:
        return Polarization(self.alternating)

    @property
    def squared_systole(self) -> float:
        return 2.0 / sqrt(10.0)

    @property
    def exact_squared_systole(self) -> str:
        return "2/sqrt(10) = sqrt(2/5)"

    def validation_certificate(self) -> PPAVValidationResult:
        return validate_polarized_abelian_data(
            self.metric_core,
            self.complex_structure_numerator,
            self.alternating,
            scale_radicand=self.scale_radicand,
            expected_type=(1, 5),
        )

    def core_relative_systole(self) -> RelativeSystoleResult:
        self.validation_certificate()
        return compute_relative_systole(
            self.alternating,
            self.metric_core,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )

    def dual_d4_certificate(self) -> bool:
        inverse = invert_rational_matrix(self.metric_core)
        dual_core = tuple(tuple(10 * value for value in row) for row in inverse)
        if dual_core != _fraction_matrix(TYPE_15_DUAL_CORE):
            return False
        change = _fraction_matrix(TYPE_15_D4_CHANGE)
        transformed = _multiply(_multiply(_transpose(change), dual_core), change)
        return transformed == _fraction_matrix(D4_ROOT_GRAM) and abs(determinant(TYPE_15_D4_CHANGE)) == 1

    def cm_certificate(self) -> Type15CMCertificate:
        structure = _fraction_matrix(self.complex_structure_numerator)
        squared = _multiply(structure, structure)
        expected = tuple(
            tuple(Fraction(-10 if row == column else 0) for column in range(4))
            for row in range(4)
        )
        if squared != expected:
            raise ArithmeticError("integral CM endomorphism does not square to -10")
        change = _fraction_matrix(TYPE_15_CM_ISOGENY_CHANGE)
        conjugated = _multiply(_multiply(invert_rational_matrix(change), structure), change)
        if conjugated != _fraction_matrix(TYPE_15_CM_BLOCK):
            raise ArithmeticError("CM isogeny did not split the rational representation")
        commutant = rational_commutant_dimension(self.complex_structure_numerator)
        degree = abs(determinant(TYPE_15_CM_ISOGENY_CHANGE))
        return Type15CMCertificate(
            field="Q(sqrt(-10))",
            order_discriminant=-40,
            elliptic_period="i*sqrt(10)",
            commutant_dimension=commutant,
            rational_isogeny_degree=degree,
            block_matrix=TYPE_15_CM_BLOCK,
            is_cm=commutant == 8,
        )


TYPE_15_EXACT_MODEL = Type15ExactModel()
