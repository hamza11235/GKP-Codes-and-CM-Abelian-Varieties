"""Polarizations on the hexagonal CM surface from Eisenstein Hermitian forms.

We use ``omega = exp(2*pi*i/3)``, so ``omega^2 + omega + 1 = 0``.
For a Hermitian matrix ``M`` over ``Z[omega]`` the compatible physical
metric has the form ``G = G_core / sqrt(3)``.  The integral matrix
``G_core`` is used for exact CVP enumeration; consequently the returned
core squared systole is the coefficient of ``1/sqrt(3)`` in the physical
squared systole.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt, sqrt

from .conventions import MetricConvention
from .metric import Metric
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import RelativeSystoleResult, compute_relative_systole


EISENSTEIN_COMPLEX_STRUCTURE_NUMERATOR = (
    (-1, 0, 2, 0),
    (0, -1, 0, 2),
    (-2, 0, 1, 0),
    (0, -2, 0, 1),
)


def _multiply(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def eisenstein_norm(first: int, second: int) -> int:
    """Return the norm of ``first + second*omega``."""

    return first * first - first * second + second * second


def _eisenstein_associates(first: int, second: int) -> tuple[tuple[int, int], ...]:
    # Multiplication by omega sends (m,n) to (-n,m-n).
    values = []
    current = (first, second)
    for _ in range(3):
        values.extend((current, (-current[0], -current[1])))
        current = (-current[1], current[0] - current[1])
    return tuple(values)


def _canonical_eisenstein_associate(first: int, second: int) -> tuple[int, int]:
    return max(_eisenstein_associates(first, second))


def _in_eisenstein_voronoi_cell(first: int, second: int, scale: int) -> bool:
    """Test membership in the closed Voronoi cell of ``scale*Z[omega]``."""

    return (
        abs(2 * first - second) <= scale
        and abs(first + second) <= scale
        and abs(first - 2 * second) <= scale
    )


@dataclass(frozen=True, order=True)
class EisensteinHermitianForm:
    """The Hermitian matrix [[a,z],[conj(z),c]] over Z[omega]."""

    a: int
    c: int
    first: int = 0
    second: int = 0

    def __post_init__(self) -> None:
        if self.a <= 0 or self.c <= 0:
            raise ValueError("diagonal entries must be positive")
        if self.determinant <= 0:
            raise ValueError("the Hermitian form must be positive definite")

    @property
    def off_diagonal_norm(self) -> int:
        return eisenstein_norm(self.first, self.second)

    @property
    def determinant(self) -> int:
        return self.a * self.c - self.off_diagonal_norm

    @property
    def is_coupled(self) -> bool:
        return self.first != 0 or self.second != 0

    @property
    def metric_core(self) -> tuple[tuple[int, ...], ...]:
        """Integral core with physical metric ``metric_core / sqrt(3)``.

        Coordinates are ``(x1,x2,y1,y2)`` for
        ``(x1 + omega*y1, x2 + omega*y2)``.
        """

        a, c, m, n = self.a, self.c, self.first, self.second
        return (
            (2 * a, 2 * m - n, -a, -m - n),
            (2 * m - n, 2 * c, -m + 2 * n, -c),
            (-a, -m + 2 * n, 2 * a, 2 * m - n),
            (-m - n, -c, 2 * m - n, 2 * c),
        )

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        product = _multiply(self.metric_core, EISENSTEIN_COMPLEX_STRUCTURE_NUMERATOR)
        if any(value % 3 for row in product for value in row):
            raise ArithmeticError("Eisenstein Riemann form is not integral")
        return tuple(tuple(value // 3 for value in row) for row in product)

    @property
    def polarization_type(self) -> tuple[int, int]:
        return Polarization(self.alternating).type

    def validation_certificate(self) -> PPAVValidationResult:
        Metric(self.metric_core)
        polarization = Polarization(self.alternating)
        if polarization.dimension != 2:
            raise ArithmeticError("Eisenstein surface has the wrong dimension")
        if determinant(self.metric_core) != 9 * self.determinant * self.determinant:
            raise ArithmeticError("real and complex determinants disagree")
        if polarization.kernel_order != self.determinant * self.determinant:
            raise ArithmeticError("polarization degree and kernel order disagree")

        return validate_polarized_abelian_data(
            self.metric_core,
            EISENSTEIN_COMPLEX_STRUCTURE_NUMERATOR,
            self.alternating,
            scale_radicand=3,
            expected_type=self.polarization_type,
        )

    def validate(self) -> None:
        self.validation_certificate()

    def compute_core_relative_systole(self) -> RelativeSystoleResult:
        """Compute exact ``q`` such that physical ``ell^2 = q/sqrt(3)``."""

        self.validate()
        return compute_relative_systole(
            self.alternating,
            self.metric_core,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )


@dataclass(frozen=True)
class EisensteinPolarizationResult:
    form: EisensteinHermitianForm
    core_systole_result: RelativeSystoleResult

    @property
    def squared_systole_coefficient(self) -> Fraction:
        """Exact coefficient ``q`` in ``ell^2 = q/sqrt(3)``."""

        value = self.core_systole_result.squared_systole
        if not isinstance(value, Fraction):
            raise ArithmeticError("Eisenstein core calculation was not exact")
        return value

    @property
    def squared_systole(self) -> float:
        return float(self.squared_systole_coefficient) / sqrt(3)

    @property
    def systole(self) -> float:
        return sqrt(self.squared_systole)

    @property
    def metric_convention(self) -> MetricConvention:
        return self.core_systole_result.metric_convention


def reduced_eisenstein_hermitian_forms(
    target_determinant: int,
) -> tuple[EisensteinHermitianForm, ...]:
    """Enumerate a reduced domain for binary Eisenstein Hermitian forms.

    We impose ``a <= c``, reduce ``z`` into the Voronoi cell of
    ``a*Z[omega]``, and quotient multiplication of ``z`` by the six units.
    The covering-radius bound gives ``a^2 <= 3*det/2``.
    """

    if target_determinant <= 0:
        raise ValueError("target determinant must be positive")
    forms: list[EisensteinHermitianForm] = []
    maximum_a = isqrt((3 * target_determinant) // 2) + 2
    for a in range(1, maximum_a + 1):
        for first in range(-a, a + 1):
            for second in range(-a, a + 1):
                if not _in_eisenstein_voronoi_cell(first, second, a):
                    continue
                if (first, second) != _canonical_eisenstein_associate(first, second):
                    continue
                numerator = target_determinant + eisenstein_norm(first, second)
                if numerator % a:
                    continue
                c = numerator // a
                if c < a:
                    continue
                form = EisensteinHermitianForm(a, c, first, second)
                if form.determinant == target_determinant:
                    forms.append(form)
    return tuple(sorted(set(forms)))


def survey_eisenstein_cm_polarizations(
    target_determinant: int,
) -> tuple[EisensteinPolarizationResult, ...]:
    """Rank reduced polarizations of fixed determinant on E_omega squared."""

    results = tuple(
        EisensteinPolarizationResult(form, form.compute_core_relative_systole())
        for form in reduced_eisenstein_hermitian_forms(target_determinant)
    )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                -item.squared_systole,
                item.form.a + item.form.c,
                item.form,
            ),
        )
    )
