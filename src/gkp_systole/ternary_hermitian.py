"""Ternary Hermitian CM polarizations for complex dimension three.

The module is the rank-three analogue of :mod:`quadratic_hermitian`.  A
positive Hermitian matrix over an imaginary-quadratic order defines a
polarized abelian threefold isogenous to ``E_Delta^3``.  We retain an exact
integral core for the metric, polarization, and relative-systole calculation;
floating point enters only when displaying the physically normalized value.

The bounded enumerator is a reconnaissance tool, not a classification up to
``GL(3, O_Delta)``.  Its bounds and lack of complete isometry reduction are
recorded explicitly so that a high-scoring candidate is never mislabeled a
global optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import sqrt
from typing import Iterable

from .conventions import MetricConvention
from .moduli_search import CompatibleMetricFamily
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .quadratic_hermitian import Element, ImaginaryQuadraticOrder
from .systole import RelativeSystoleResult, compute_relative_systole


def _complex_structure_numerator(
    order: ImaginaryQuadraticOrder,
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    """Numerator of multiplication by ``-i`` in ``(1, omega)`` coordinates."""

    b = order.basis_trace
    c = order.basis_norm
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(dimension))
        for row in range(dimension)
    )
    return tuple(
        tuple(
            (
                b * identity[row][column]
                if row < dimension and column < dimension
                else 2 * c * identity[row][column - dimension]
                if row < dimension and column >= dimension
                else -2 * identity[row - dimension][column]
                if row >= dimension and column < dimension
                else -b * identity[row - dimension][column - dimension]
            )
            for column in range(2 * dimension)
        )
        for row in range(2 * dimension)
    )


@dataclass(frozen=True, order=True)
class TernaryQuadraticHermitianForm:
    """A positive rank-three Hermitian form over an imaginary-quadratic order."""

    order: ImaginaryQuadraticOrder
    a: int
    b: int
    c: int
    z12_first: int = 0
    z12_second: int = 0
    z13_first: int = 0
    z13_second: int = 0
    z23_first: int = 0
    z23_second: int = 0

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("diagonal entries must be positive")
        if self.a * self.b - self.order.norm(self.z12) <= 0:
            raise ValueError("leading Hermitian minor must be positive")
        if self.determinant <= 0:
            raise ValueError("Hermitian form must be positive definite")

    @property
    def z12(self) -> Element:
        return (self.z12_first, self.z12_second)

    @property
    def z13(self) -> Element:
        return (self.z13_first, self.z13_second)

    @property
    def z23(self) -> Element:
        return (self.z23_first, self.z23_second)

    @property
    def is_coupled(self) -> bool:
        return any(value != (0, 0) for value in (self.z12, self.z13, self.z23))

    @property
    def determinant(self) -> int:
        order = self.order
        triple = order.multiply(order.multiply(self.z12, self.z23), order.conjugate(self.z13))
        return (
            self.a * self.b * self.c
            + order.twice_real_part(triple)
            - self.a * order.norm(self.z23)
            - self.b * order.norm(self.z13)
            - self.c * order.norm(self.z12)
        )

    @property
    def hermitian_matrix(self) -> tuple[tuple[Element, ...], ...]:
        order = self.order
        return (
            ((self.a, 0), self.z12, self.z13),
            (order.conjugate(self.z12), (self.b, 0), self.z23),
            (order.conjugate(self.z13), order.conjugate(self.z23), (self.c, 0)),
        )

    @property
    def complex_structure_numerator(self) -> tuple[tuple[int, ...], ...]:
        return _complex_structure_numerator(self.order, 3)

    @property
    def metric_core(self) -> tuple[tuple[int, ...], ...]:
        """Exact core of ``2 Re(v^* H w)/sqrt(|Delta|)``."""

        order = self.order
        basis = tuple((mode, (1, 0)) for mode in range(3)) + tuple(
            (mode, (0, 1)) for mode in range(3)
        )
        matrix = self.hermitian_matrix
        rows: list[tuple[int, ...]] = []
        for left_mode, left_scalar in basis:
            row = []
            for right_mode, right_scalar in basis:
                value = order.multiply(
                    order.multiply(order.conjugate(left_scalar), matrix[left_mode][right_mode]),
                    right_scalar,
                )
                row.append(order.twice_real_part(value))
            rows.append(tuple(row))
        result = tuple(rows)
        if any(result[row][column] != result[column][row] for row in range(6) for column in range(6)):
            raise ArithmeticError("real Hermitian representation is not symmetric")
        return result

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        metric = self.metric_core
        structure = self.complex_structure_numerator
        radicand = self.order.radicand
        product_matrix = tuple(
            tuple(
                sum(metric[row][inner] * structure[inner][column] for inner in range(6))
                for column in range(6)
            )
            for row in range(6)
        )
        if any(value % radicand for row in product_matrix for value in row):
            raise ArithmeticError("Hermitian form does not give an integral Riemann form")
        return tuple(tuple(value // radicand for value in row) for row in product_matrix)

    @property
    def polarization_type(self) -> tuple[int, int, int]:
        return Polarization(self.alternating).type

    def validation_certificate(self) -> PPAVValidationResult:
        if determinant(self.metric_core) != self.order.radicand**3 * self.determinant**2:
            raise ArithmeticError("real and Hermitian determinants disagree")
        polarization = Polarization(self.alternating)
        if polarization.kernel_order != self.determinant**2:
            raise ArithmeticError("polarization degree and Hermitian determinant disagree")
        return validate_polarized_abelian_data(
            self.metric_core,
            self.complex_structure_numerator,
            self.alternating,
            scale_radicand=self.order.radicand,
            expected_type=self.polarization_type,
        )

    def validate(self) -> None:
        self.validation_certificate()

    def compute_core_relative_systole(self) -> RelativeSystoleResult:
        self.validate()
        return compute_relative_systole(
            self.alternating,
            self.metric_core,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )


@dataclass(frozen=True)
class TernaryQuadraticHermitianResult:
    form: TernaryQuadraticHermitianForm
    core_systole_result: RelativeSystoleResult

    @property
    def squared_systole_coefficient(self) -> Fraction:
        value = self.core_systole_result.squared_systole
        if not isinstance(value, Fraction):
            raise ArithmeticError("exact core calculation returned a float")
        return value

    @property
    def squared_systole(self) -> float:
        return float(self.squared_systole_coefficient) / sqrt(self.form.order.radicand)


def _unit_permutation_key(form: TernaryQuadraticHermitianForm) -> tuple[int, ...]:
    """Canonical key for mode permutations and diagonal order-unit changes."""

    order = form.order
    matrix = form.hermitian_matrix
    diagonal = (form.a, form.b, form.c)
    mode_permutations = tuple(
        permutation
        for permutation in permutations(range(3))
        if tuple(diagonal[index] for index in permutation) == diagonal
    )
    keys: list[tuple[int, ...]] = []
    for permutation in mode_permutations:
        for units in product(order.units, repeat=3):
            transformed: list[list[Element]] = []
            for row in range(3):
                transformed_row = []
                for column in range(3):
                    value = order.multiply(
                        order.multiply(
                            order.conjugate(units[row]),
                            matrix[permutation[row]][permutation[column]],
                        ),
                        units[column],
                    )
                    transformed_row.append(value)
                transformed.append(transformed_row)
            if any(transformed[index][index][1] for index in range(3)):
                raise ArithmeticError("unit transformation made a diagonal non-rational")
            keys.append(
                (
                    transformed[0][0][0],
                    transformed[1][1][0],
                    transformed[2][2][0],
                    *transformed[0][1],
                    *transformed[0][2],
                    *transformed[1][2],
                )
            )
    return min(keys)


def bounded_ternary_hermitian_forms(
    order: ImaginaryQuadraticOrder,
    target_determinant: int,
    *,
    maximum_diagonal: int = 10,
    off_diagonal_bound: int = 1,
    requested_types: Iterable[tuple[int, int, int]] | None = None,
) -> tuple[TernaryQuadraticHermitianForm, ...]:
    """Enumerate a bounded collection of positive ternary Hermitian forms.

    ``a <= b <= c`` is imposed, and the determinant equation is solved for
    ``c`` rather than searched.  The coefficient box is deliberately small
    and reproducible.  No completeness modulo Hermitian isometry is claimed.
    """

    if target_determinant <= 0 or maximum_diagonal <= 0 or off_diagonal_bound < 0:
        raise ValueError("determinant/diagonal bounds must be positive and coefficient bound nonnegative")
    allowed = None if requested_types is None else {tuple(value) for value in requested_types}
    elements = tuple(product(range(-off_diagonal_bound, off_diagonal_bound + 1), repeat=2))
    forms: dict[tuple[int, ...], TernaryQuadraticHermitianForm] = {}
    for a in range(1, maximum_diagonal + 1):
        for b in range(a, maximum_diagonal + 1):
            for z12 in elements:
                leading = a * b - order.norm(z12)
                if leading <= 0:
                    continue
                for z13 in elements:
                    norm13 = order.norm(z13)
                    for z23 in elements:
                        triple = order.multiply(
                            order.multiply(z12, z23),
                            order.conjugate(z13),
                        )
                        constant = (
                            order.twice_real_part(triple)
                            - a * order.norm(z23)
                            - b * norm13
                        )
                        numerator = target_determinant - constant
                        if numerator % leading:
                            continue
                        c = numerator // leading
                        if c < b or c > maximum_diagonal:
                            continue
                        form = TernaryQuadraticHermitianForm(
                            order,
                            a,
                            b,
                            c,
                            z12[0],
                            z12[1],
                            z13[0],
                            z13[1],
                            z23[0],
                            z23[1],
                        )
                        if form.determinant != target_determinant:
                            raise ArithmeticError("determinant solver produced an inconsistent form")
                        key = _unit_permutation_key(form)
                        current = forms.get(key)
                        if current is None or form < current:
                            forms[key] = form
    values = tuple(sorted(forms.values()))
    if allowed is not None:
        values = tuple(form for form in values if form.polarization_type in allowed)
    return values


def survey_ternary_hermitian_polarizations(
    discriminants: Iterable[int],
    target_determinant: int,
    *,
    maximum_diagonal: int = 10,
    off_diagonal_bound: int = 1,
    requested_types: Iterable[tuple[int, int, int]] | None = None,
) -> tuple[TernaryQuadraticHermitianResult, ...]:
    """Rank an exact bounded survey of CM-product threefold polarizations."""

    results: list[TernaryQuadraticHermitianResult] = []
    for discriminant in discriminants:
        order = ImaginaryQuadraticOrder(int(discriminant))
        for form in bounded_ternary_hermitian_forms(
            order,
            target_determinant,
            maximum_diagonal=maximum_diagonal,
            off_diagonal_bound=off_diagonal_bound,
            requested_types=requested_types,
        ):
            results.append(
                TernaryQuadraticHermitianResult(form, form.compute_core_relative_systole())
            )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                -item.squared_systole,
                abs(item.form.order.discriminant),
                item.form,
            ),
        )
    )


def ternary_hermitian_moduli_family(
    form: TernaryQuadraticHermitianForm,
    *,
    name: str | None = None,
) -> CompatibleMetricFamily:
    """Use an exact ternary CM candidate as a 12-dimensional search center."""

    form.validate()
    radicand = form.order.radicand
    physical_metric = tuple(
        tuple(float(value) / sqrt(radicand) for value in row)
        for row in form.metric_core
    )
    core = form.compute_core_relative_systole()
    coefficient = core.squared_systole
    return CompatibleMetricFamily.from_reference(
        name=name or f"g=3 type {form.polarization_type} around {form.order.label}",
        alternating=form.alternating,
        reference_metric=physical_metric,
        reference_exact_ell_squared=f"({coefficient})/sqrt({radicand})",
        reference_ell_squared=float(coefficient) / sqrt(radicand),
        reference_cm=(
            f"E_{form.order.discriminant}^3 with ternary Hermitian form "
            f"diag=({form.a},{form.b},{form.c})"
        ),
    )


# Exact high-scoring representatives found by the bounded Phase-8 scan.  They
# are named benchmarks, not assertions of global optimality.
G3_TYPE_112_GAUSSIAN_FORM = TernaryQuadraticHermitianForm(
    ImaginaryQuadraticOrder(-4),
    1,
    2,
    2,
    0,
    0,
    0,
    0,
    -1,
    -1,
)

G3_TYPE_113_EISENSTEIN_FORM = TernaryQuadraticHermitianForm(
    ImaginaryQuadraticOrder(-3),
    2,
    2,
    2,
    -1,
    0,
    -1,
    0,
    0,
    1,
)

G3_TYPE_122_GAUSSIAN_FORM = TernaryQuadraticHermitianForm(
    ImaginaryQuadraticOrder(-4),
    2,
    2,
    3,
    0,
    0,
    -1,
    -1,
    -1,
    -1,
)

G3_NONUNIFORM_CM_BENCHMARKS = (
    G3_TYPE_112_GAUSSIAN_FORM,
    G3_TYPE_113_EISENSTEIN_FORM,
    G3_TYPE_122_GAUSSIAN_FORM,
)
