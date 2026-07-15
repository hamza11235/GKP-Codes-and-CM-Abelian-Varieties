"""Certified scans of product-polarized CM abelian surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import prod

from .benchmarks import canonical_alternating
from .conventions import MetricConvention
from .cm import ReducedQuadraticForm, reduced_primitive_forms
from .kernel import invert_rational_matrix
from .metric import Metric
from .models import PeriodModel, RationalMatrix, ScaledSystoleResult
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import compute_relative_systole


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column] for inner in range(len(right))),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _scale(matrix: RationalMatrix, scalar: Fraction) -> RationalMatrix:
    return tuple(tuple(scalar * value for value in row) for row in matrix)


@dataclass(frozen=True)
class CMProductSurface:
    """A product of two CM elliptic curves with a product polarization.

    Both factors use ideal classes of the same imaginary-quadratic order.  In
    the lattice-coordinate ordering ``(a1, a2, b1, b2)``, the alternating form
    is ``J_D``.  The compatible polarization metric is the orthogonal sum of
    ``d_i`` times the area-one metric on factor ``i``.
    """

    first_form: ReducedQuadraticForm
    second_form: ReducedQuadraticForm
    polarization_type: tuple[int, int]

    def __post_init__(self) -> None:
        if self.first_form.discriminant != self.second_form.discriminant:
            raise ValueError("the exact product model requires a common CM discriminant")
        d1, d2 = self.polarization_type
        if d1 <= 0 or d2 <= 0 or d2 % d1:
            raise ValueError("polarization type must satisfy positive d1 dividing d2")

    @property
    def discriminant(self) -> int:
        return self.first_form.discriminant

    @property
    def scale_radicand(self) -> int:
        return abs(self.discriminant)

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        return canonical_alternating(self.polarization_type)

    @property
    def principal_period_model(self) -> PeriodModel:
        first = self.first_form
        second = self.second_form
        zero = Fraction(0)
        return PeriodModel(
            name=(
                f"CM product surface Delta={self.discriminant}, "
                f"forms={first.as_tuple()}x{second.as_tuple()}"
            ),
            real_part=((first.tau_real, zero), (zero, second.tau_real)),
            imaginary_core=(
                (first.tau_imaginary_core, zero),
                (zero, second.tau_imaginary_core),
            ),
            scale_radicand=self.scale_radicand,
            source="Product of two reduced ideal classes of one imaginary-quadratic order.",
            cm_field=f"imaginary quadratic order of discriminant {self.discriminant}",
        )

    @property
    def metric_core(self) -> RationalMatrix:
        """Core of the compatible polarization metric ``G/sqrt(|Delta|)``."""

        forms = (self.first_form, self.second_form)
        degrees = self.polarization_type
        size = 4
        matrix = [[Fraction(0) for _ in range(size)] for _ in range(size)]
        for mode, (form, degree) in enumerate(zip(forms, degrees)):
            a_index = mode
            b_index = mode + 2
            matrix[a_index][a_index] = Fraction(2 * degree * form.a)
            matrix[a_index][b_index] = Fraction(-degree * form.b)
            matrix[b_index][a_index] = Fraction(-degree * form.b)
            matrix[b_index][b_index] = Fraction(2 * degree * form.c)
        return tuple(tuple(row) for row in matrix)

    @property
    def metric_numeric(self) -> tuple[tuple[float, ...], ...]:
        scale = self.principal_period_model.metric_scale
        return tuple(
            tuple(float(value) * scale for value in row)
            for row in self.metric_core
        )

    @property
    def generalized_period_numeric(
        self,
    ) -> tuple[tuple[complex, ...], tuple[tuple[complex, ...], ...]]:
        """Return the generalized period blocks ``(D, D*Omega)``."""

        d1, d2 = self.polarization_type
        left = ((complex(d1), complex(0)), (complex(0), complex(d2)))
        omega = self.principal_period_model.period_numeric
        right = (
            (d1 * omega[0][0], d1 * omega[0][1]),
            (d2 * omega[1][0], d2 * omega[1][1]),
        )
        return left, right

    @property
    def complex_structure_core(self) -> RationalMatrix:
        inverse_alternating = invert_rational_matrix(self.alternating)
        return _scale(
            _multiply(inverse_alternating, self.metric_core),
            Fraction(-1),
        )

    def validation_certificate(self) -> PPAVValidationResult:
        self.principal_period_model.validate()
        polarization = Polarization(self.alternating)
        if polarization.type != self.polarization_type:
            raise ArithmeticError("product polarization has the wrong type")
        Metric(self.metric_core)

        expected_determinant = (
            self.scale_radicand ** 2 * prod(self.polarization_type) ** 2
        )
        if determinant(self.metric_core) != expected_determinant:
            raise ArithmeticError("polarization metric has the wrong covolume")

        return validate_polarized_abelian_data(
            self.metric_core,
            self.complex_structure_core,
            self.alternating,
            scale_radicand=self.scale_radicand,
            expected_type=self.polarization_type,
        )

    def validate(self) -> None:
        self.validation_certificate()

    def compute_relative_systole(self) -> ScaledSystoleResult:
        self.validate()
        core_result = compute_relative_systole(
            self.alternating,
            self.metric_core,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        return ScaledSystoleResult(self.principal_period_model, core_result)


@dataclass(frozen=True)
class CMProductSurfaceResult:
    surface: CMProductSurface
    systole_result: ScaledSystoleResult

    @property
    def squared_systole(self) -> float:
        return self.systole_result.squared_systole


def survey_cm_product_surfaces(
    maximum_absolute_discriminant: int,
    polarization_type: tuple[int, int],
) -> tuple[CMProductSurfaceResult, ...]:
    """Rank exact same-order CM product surfaces for a fixed type."""

    if maximum_absolute_discriminant < 3:
        raise ValueError("maximum discriminant size must be at least 3")
    d1, d2 = polarization_type
    if d1 <= 0 or d2 <= 0 or d2 % d1:
        raise ValueError("polarization type must satisfy positive d1 dividing d2")

    results: list[CMProductSurfaceResult] = []
    for absolute in range(3, maximum_absolute_discriminant + 1):
        discriminant = -absolute
        if discriminant % 4 not in (0, 1):
            continue
        forms = reduced_primitive_forms(discriminant)
        pairs = (
            combinations_with_replacement(forms, 2)
            if d1 == d2
            else product(forms, repeat=2)
        )
        for first, second in pairs:
            surface = CMProductSurface(first, second, polarization_type)
            results.append(
                CMProductSurfaceResult(surface, surface.compute_relative_systole())
            )

    return tuple(
        sorted(
            results,
            key=lambda item: (
                -item.squared_systole,
                abs(item.surface.discriminant),
                item.surface.first_form.as_tuple(),
                item.surface.second_form.as_tuple(),
            ),
        )
    )
