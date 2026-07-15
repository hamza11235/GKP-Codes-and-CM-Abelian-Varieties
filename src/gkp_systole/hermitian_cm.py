"""Polarizations on the square CM surface from Gaussian Hermitian forms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from .conventions import MetricConvention
from .metric import Metric
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import RelativeSystoleResult, compute_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


GAUSSIAN_COMPLEX_STRUCTURE = (
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (-1, 0, 0, 0),
    (0, -1, 0, 0),
)


def _multiply(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _canonical_gaussian_associate(real: int, imaginary: int) -> tuple[int, int]:
    """Choose one representative modulo multiplication by a Gaussian unit."""

    return max(
        (real, imaginary),
        (-imaginary, real),
        (-real, -imaginary),
        (imaginary, -real),
    )


@dataclass(frozen=True, order=True)
class GaussianHermitianForm:
    """The Hermitian matrix [[a,z],[conj(z),c]] over Z[i]."""

    a: int
    c: int
    real: int = 0
    imaginary: int = 0

    def __post_init__(self) -> None:
        if self.a <= 0 or self.c <= 0:
            raise ValueError("diagonal entries must be positive")
        if self.determinant <= 0:
            raise ValueError("the Hermitian form must be positive definite")

    @property
    def determinant(self) -> int:
        return self.a * self.c - self.real * self.real - self.imaginary * self.imaginary

    @property
    def is_coupled(self) -> bool:
        return self.real != 0 or self.imaginary != 0

    @property
    def hermitian_entries(self) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
        z = complex(self.real, self.imaginary)
        return ((complex(self.a), z), (z.conjugate(), complex(self.c)))

    @property
    def metric(self) -> tuple[tuple[int, ...], ...]:
        """Real representation in coordinates (x1,x2,y1,y2)."""

        a, c, m, n = self.a, self.c, self.real, self.imaginary
        return (
            (a, m, 0, n),
            (m, c, -n, 0),
            (0, -n, a, m),
            (n, 0, m, c),
        )

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        return _multiply(self.metric, GAUSSIAN_COMPLEX_STRUCTURE)

    @property
    def polarization_type(self) -> tuple[int, int]:
        return Polarization(self.alternating).type

    def validation_certificate(self) -> PPAVValidationResult:
        metric = Metric(self.metric)
        polarization = Polarization(self.alternating)
        if metric.dimension != 4 or polarization.dimension != 2:
            raise ArithmeticError("Gaussian surface has the wrong dimension")
        if determinant(self.metric) != self.determinant * self.determinant:
            raise ArithmeticError("real and complex determinants disagree")
        if polarization.kernel_order != self.determinant * self.determinant:
            raise ArithmeticError("polarization degree and kernel order disagree")

        return validate_polarized_abelian_data(
            self.metric,
            GAUSSIAN_COMPLEX_STRUCTURE,
            self.alternating,
            expected_type=self.polarization_type,
        )

    def validate(self) -> None:
        self.validation_certificate()

    def compute_relative_systole(self) -> RelativeSystoleResult:
        self.validate()
        return compute_relative_systole(
            self.alternating,
            self.metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )


@dataclass(frozen=True)
class GaussianPolarizationResult:
    form: GaussianHermitianForm
    systole_result: RelativeSystoleResult

    @property
    def squared_systole(self) -> float:
        return float(self.systole_result.squared_systole)


def reduced_gaussian_hermitian_forms(
    target_determinant: int,
) -> tuple[GaussianHermitianForm, ...]:
    """Enumerate a reduced domain for binary Gaussian Hermitian forms.

    We impose ``a <= c`` and reduce the real and imaginary parts of the
    off-diagonal entry into ``[-a/2,a/2]``.  Multiplication by Gaussian units
    is removed by a canonical associate.  The determinant inequality then
    gives ``a^2 <= 2*det``, so the search is finite.
    """

    if target_determinant <= 0:
        raise ValueError("target determinant must be positive")
    forms: list[GaussianHermitianForm] = []
    maximum_a = isqrt(2 * target_determinant) + 1
    for a in range(1, maximum_a + 1):
        for real in range(-a, a + 1):
            for imaginary in range(-a, a + 1):
                if 2 * abs(real) > a or 2 * abs(imaginary) > a:
                    continue
                if (real, imaginary) != _canonical_gaussian_associate(real, imaginary):
                    continue
                numerator = target_determinant + real * real + imaginary * imaginary
                if numerator % a:
                    continue
                c = numerator // a
                if c < a:
                    continue
                form = GaussianHermitianForm(a, c, real, imaginary)
                if form.determinant == target_determinant:
                    forms.append(form)
    return tuple(sorted(set(forms)))


def survey_gaussian_cm_polarizations(
    target_determinant: int,
) -> tuple[GaussianPolarizationResult, ...]:
    """Rank reduced polarizations of fixed determinant on E_i squared."""

    results = tuple(
        GaussianPolarizationResult(form, form.compute_relative_systole())
        for form in reduced_gaussian_hermitian_forms(target_determinant)
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
