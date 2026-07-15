"""Binary Hermitian polarizations over general imaginary-quadratic orders.

For a negative discriminant ``Delta`` write

``O_Delta = Z[omega]``,  ``omega = (B + i*sqrt(|Delta|))/2``

with ``B`` equal to zero or one and ``omega^2 - B*omega + C = 0``.  A positive
binary Hermitian matrix over this order gives a polarization on ``E_Delta^2``.
The physical metric is stored exactly as ``G_core/sqrt(|Delta|)``.

The bounded enumerator is deliberately described as a candidate enumerator:
it imposes standard elementary reductions but does not claim a complete class
enumeration of binary Hermitian forms for arbitrary discriminant.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Iterable

from .conventions import MetricConvention
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import RelativeSystoleResult, compute_relative_systole


Element = tuple[int, int]


@dataclass(frozen=True, order=True)
class ImaginaryQuadraticOrder:
    discriminant: int

    def __post_init__(self) -> None:
        if self.discriminant >= 0 or self.discriminant % 4 not in (0, 1):
            raise ValueError("discriminant must be negative and congruent to 0 or 1 mod 4")

    @property
    def radicand(self) -> int:
        return -self.discriminant

    @property
    def basis_trace(self) -> int:
        return self.discriminant % 2

    @property
    def basis_norm(self) -> int:
        b = self.basis_trace
        return (b * b - self.discriminant) // 4

    @property
    def label(self) -> str:
        return f"O({self.discriminant})"

    def add(self, left: Element, right: Element) -> Element:
        return (left[0] + right[0], left[1] + right[1])

    def multiply(self, left: Element, right: Element) -> Element:
        b = self.basis_trace
        c = self.basis_norm
        return (
            left[0] * right[0] - c * left[1] * right[1],
            left[0] * right[1] + left[1] * right[0] + b * left[1] * right[1],
        )

    def conjugate(self, value: Element) -> Element:
        return (value[0] + self.basis_trace * value[1], -value[1])

    def norm(self, value: Element) -> int:
        product = self.multiply(value, self.conjugate(value))
        if product[1] != 0:
            raise ArithmeticError("quadratic norm did not land in Z")
        return product[0]

    def twice_real_part(self, value: Element) -> int:
        return 2 * value[0] + self.basis_trace * value[1]

    @property
    def units(self) -> tuple[Element, ...]:
        values = {
            (first, second)
            for first in range(-2, 3)
            for second in range(-2, 3)
            if self.norm((first, second)) == 1
        }
        return tuple(sorted(values))

    @property
    def complex_structure_numerator(self) -> tuple[tuple[int, ...], ...]:
        """Numerator of multiplication by ``-i`` in two-mode coordinates."""

        b = self.basis_trace
        c = self.basis_norm
        return (
            (b, 0, 2 * c, 0),
            (0, b, 0, 2 * c),
            (-2, 0, -b, 0),
            (0, -2, 0, -b),
        )


def _canonical_associate(order: ImaginaryQuadraticOrder, value: Element) -> Element:
    return max(order.multiply(value, unit) for unit in order.units)


def _nearest_residue_representatives(
    order: ImaginaryQuadraticOrder,
    value: Element,
    modulus: int,
) -> tuple[Element, ...]:
    """Return every shortest representative of ``value mod modulus*O``.

    The two integral coordinates of ``O_Delta`` make this a two-dimensional
    closest-vector problem.  Searching two coordinates around the rounded
    quotient is sufficient after completing the square for the positive
    quadratic norm form; the extra one-coordinate margin also makes boundary
    ties explicit and deterministic.
    """

    centers = tuple(round(-coordinate / modulus) for coordinate in value)
    candidates = {
        (value[0] + modulus * first, value[1] + modulus * second)
        for first in range(centers[0] - 2, centers[0] + 3)
        for second in range(centers[1] - 2, centers[1] + 3)
    }
    minimum = min(order.norm(candidate) for candidate in candidates)
    return tuple(sorted(candidate for candidate in candidates if order.norm(candidate) == minimum))


def _is_voronoi_reduced(
    order: ImaginaryQuadraticOrder,
    value: Element,
    modulus: int,
) -> bool:
    baseline = order.norm(value)
    for first in range(-2, 3):
        for second in range(-2, 3):
            translated = (value[0] + modulus * first, value[1] + modulus * second)
            if order.norm(translated) < baseline:
                return False
    return True


@dataclass(frozen=True, order=True)
class QuadraticHermitianForm:
    """The matrix ``[[a,z],[conj(z),c]]`` over an imaginary-quadratic order."""

    order: ImaginaryQuadraticOrder
    a: int
    c: int
    first: int = 0
    second: int = 0

    def __post_init__(self) -> None:
        if self.a <= 0 or self.c <= 0:
            raise ValueError("diagonal entries must be positive")
        if self.determinant <= 0:
            raise ValueError("Hermitian form must be positive definite")

    @property
    def off_diagonal(self) -> Element:
        return (self.first, self.second)

    @property
    def off_diagonal_norm(self) -> int:
        return self.order.norm(self.off_diagonal)

    @property
    def determinant(self) -> int:
        return self.a * self.c - self.off_diagonal_norm

    @property
    def is_coupled(self) -> bool:
        return self.off_diagonal != (0, 0)

    @property
    def metric_core(self) -> tuple[tuple[int, ...], ...]:
        """Exact core of ``2 Re(v^* M w)/sqrt(|Delta|)``."""

        order = self.order
        one = (1, 0)
        omega = (0, 1)
        basis = ((0, one), (1, one), (0, omega), (1, omega))
        z = self.off_diagonal
        matrix = (
            ((self.a, 0), z),
            (order.conjugate(z), (self.c, 0)),
        )
        rows: list[tuple[int, ...]] = []
        for left_mode, left_scalar in basis:
            row = []
            for right_mode, right_scalar in basis:
                entry = matrix[left_mode][right_mode]
                value = order.multiply(
                    order.multiply(order.conjugate(left_scalar), entry),
                    right_scalar,
                )
                row.append(order.twice_real_part(value))
            rows.append(tuple(row))
        result = tuple(rows)
        if any(result[i][j] != result[j][i] for i in range(4) for j in range(4)):
            raise ArithmeticError("real Hermitian representation is not symmetric")
        return result

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        metric = self.metric_core
        structure = self.order.complex_structure_numerator
        radicand = self.order.radicand
        product = tuple(
            tuple(
                sum(metric[row][inner] * structure[inner][column] for inner in range(4))
                for column in range(4)
            )
            for row in range(4)
        )
        if any(value % radicand for row in product for value in row):
            raise ArithmeticError("Hermitian form does not give an integral Riemann form")
        return tuple(tuple(value // radicand for value in row) for row in product)

    @property
    def polarization_type(self) -> tuple[int, int]:
        return Polarization(self.alternating).type

    def validation_certificate(self) -> PPAVValidationResult:
        if determinant(self.metric_core) != self.order.radicand**2 * self.determinant**2:
            raise ArithmeticError("real and Hermitian determinants disagree")
        polarization = Polarization(self.alternating)
        if polarization.kernel_order != self.determinant**2:
            raise ArithmeticError("polarization degree and Hermitian determinant disagree")
        return validate_polarized_abelian_data(
            self.metric_core,
            self.order.complex_structure_numerator,
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


def reduce_quadratic_hermitian_form(
    form: QuadraticHermitianForm,
) -> QuadraticHermitianForm:
    """Canonicalize the elementary binary-Hermitian reduction orbit.

    This quotients the transformations used by the bounded enumerator:
    integral shears, exchange of the two modes, and multiplication of a mode
    by a unit.  It is an exact and idempotent candidate deduplication step.  It
    deliberately does not claim to solve arbitrary Hermitian-form isometry.
    """

    order = form.order
    determinant_value = form.determinant
    a = form.a
    c = form.c
    value = form.off_diagonal

    for _ in range(64):
        residues = _nearest_residue_representatives(order, value, a)
        value = max(_canonical_associate(order, candidate) for candidate in residues)
        numerator = determinant_value + order.norm(value)
        if numerator % a:
            raise ArithmeticError("Hermitian shear did not preserve integral diagonal data")
        c = numerator // a

        if c < a:
            a, c = c, a
            value = order.conjugate(value)
            continue

        if a == c:
            value = max(
                _canonical_associate(order, value),
                _canonical_associate(order, order.conjugate(value)),
            )
        reduced = QuadraticHermitianForm(order, a, c, value[0], value[1])
        if reduced.determinant != determinant_value:
            raise ArithmeticError("Hermitian reduction changed the determinant")
        return reduced
    raise ArithmeticError("binary Hermitian reduction did not converge")


def deduplicate_quadratic_hermitian_forms(
    forms: Iterable[QuadraticHermitianForm],
) -> tuple[QuadraticHermitianForm, ...]:
    """Remove duplicates under the exact elementary reduction above."""

    return tuple(sorted({reduce_quadratic_hermitian_form(form) for form in forms}))


@dataclass(frozen=True)
class QuadraticHermitianResult:
    form: QuadraticHermitianForm
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


def bounded_quadratic_hermitian_forms(
    order: ImaginaryQuadraticOrder,
    target_determinant: int,
    *,
    maximum_diagonal: int = 12,
) -> tuple[QuadraticHermitianForm, ...]:
    """Enumerate elementary-reduced candidates within a diagonal bound."""

    if target_determinant <= 0 or maximum_diagonal <= 0:
        raise ValueError("target determinant and maximum diagonal must be positive")
    forms: set[QuadraticHermitianForm] = set()
    for a in range(1, maximum_diagonal + 1):
        for first in range(-a, a + 1):
            for second in range(-a, a + 1):
                value = (first, second)
                if not _is_voronoi_reduced(order, value, a):
                    continue
                if value != _canonical_associate(order, value):
                    continue
                numerator = target_determinant + order.norm(value)
                if numerator % a:
                    continue
                c = numerator // a
                if c < a or c > maximum_diagonal:
                    continue
                form = QuadraticHermitianForm(order, a, c, first, second)
                if form.determinant == target_determinant:
                    forms.add(form)
    return deduplicate_quadratic_hermitian_forms(forms)


def survey_quadratic_hermitian_polarizations(
    discriminants: Iterable[int],
    target_determinant: int,
    *,
    maximum_diagonal: int = 12,
) -> tuple[QuadraticHermitianResult, ...]:
    """Rank a bounded collection of coupled and product CM polarizations."""

    results: list[QuadraticHermitianResult] = []
    for discriminant in discriminants:
        order = ImaginaryQuadraticOrder(int(discriminant))
        for form in bounded_quadratic_hermitian_forms(
            order,
            target_determinant,
            maximum_diagonal=maximum_diagonal,
        ):
            results.append(QuadraticHermitianResult(form, form.compute_core_relative_systole()))
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
