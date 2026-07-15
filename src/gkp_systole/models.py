"""Verified low-dimensional polarized abelian-variety models."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt, sqrt
from typing import Sequence

from .benchmarks import canonical_alternating
from .conventions import MetricConvention
from .kernel import invert_rational_matrix
from .metric import Metric
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_ppav_data
from .systole import RelativeSystoleResult, compute_relative_systole
from .uniform import UniformRelativeSystoleResult, compute_uniform_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _as_fraction_matrix(matrix: Sequence[Sequence[int | Fraction]]) -> RationalMatrix:
    return tuple(tuple(Fraction(value) for value in row) for row in matrix)


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


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


def _add(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(len(left[0])))
        for row in range(len(left))
    )


def _block(
    top_left: RationalMatrix,
    top_right: RationalMatrix,
    bottom_left: RationalMatrix,
    bottom_right: RationalMatrix,
) -> RationalMatrix:
    dimension = len(top_left)
    return tuple(
        tuple(top_left[row]) + tuple(top_right[row])
        for row in range(dimension)
    ) + tuple(
        tuple(bottom_left[row]) + tuple(bottom_right[row])
        for row in range(dimension)
    )


def period_metric_core(
    real_part: Sequence[Sequence[int | Fraction]],
    imaginary_core: Sequence[Sequence[int | Fraction]],
    scale_radicand: int,
) -> RationalMatrix:
    """Derive G_core when Im(Omega)=Y_core/sqrt(r).

    For the normalized period lattice ``Z^g + Omega Z^g``, the principal
    Riemann metric is

        [[Y^-1, Y^-1 X], [X Y^-1, X Y^-1 X + Y]].

    Factoring out ``1/sqrt(r)`` gives the rational matrix returned here.
    """

    if scale_radicand <= 0:
        raise ValueError("scale radicand must be positive")
    real = _as_fraction_matrix(real_part)
    imaginary = _as_fraction_matrix(imaginary_core)
    if real != _transpose(real) or imaginary != _transpose(imaginary):
        raise ValueError("period real and imaginary parts must be symmetric")
    inverse_imaginary = invert_rational_matrix(imaginary)
    scaled_inverse = _scale(inverse_imaginary, Fraction(scale_radicand))
    top_right = _multiply(scaled_inverse, real)
    bottom_left = _multiply(real, scaled_inverse)
    bottom_right = _add(_multiply(bottom_left, real), imaginary)
    return _block(scaled_inverse, top_right, bottom_left, bottom_right)


@dataclass(frozen=True)
class ScaledSystoleResult:
    """A certified core result with an algebraic overall metric scale."""

    model: "PeriodModel"
    core_result: RelativeSystoleResult

    @property
    def squared_systole_coefficient(self) -> Fraction:
        return Fraction(self.core_result.squared_systole)

    @property
    def squared_systole(self) -> float:
        return float(self.squared_systole_coefficient) / sqrt(self.model.scale_radicand)

    @property
    def systole(self) -> float:
        return sqrt(self.squared_systole)

    @property
    def class_multiplicity(self) -> int:
        return self.core_result.class_multiplicity

    @property
    def lift_multiplicity(self) -> int:
        return self.core_result.lift_multiplicity

    @property
    def certified(self) -> bool:
        return self.core_result.certified

    @property
    def metric_convention(self) -> MetricConvention:
        return self.core_result.metric_convention

    @property
    def squared_systole_expression(self) -> str:
        radicand = self.model.scale_radicand
        square_factor = 1
        for candidate in range(1, isqrt(radicand) + 1):
            if radicand % (candidate * candidate) == 0:
                square_factor = candidate
        coefficient = self.squared_systole_coefficient / square_factor
        reduced_radicand = radicand // (square_factor * square_factor)
        if reduced_radicand == 1:
            return str(coefficient)
        if coefficient == 1:
            return f"1/sqrt({reduced_radicand})"
        return f"({coefficient})/sqrt({reduced_radicand})"


@dataclass(frozen=True)
class ScaledUniformSystoleResult:
    """A certified uniform-type SVP result with an algebraic metric scale."""

    model: "PeriodModel"
    core_result: UniformRelativeSystoleResult

    @property
    def lambda1_squared_coefficient(self) -> Fraction:
        return Fraction(self.core_result.lambda1_squared)

    @property
    def lambda1_squared(self) -> float:
        return float(self.lambda1_squared_coefficient) / sqrt(self.model.scale_radicand)

    @property
    def squared_systole_coefficient(self) -> Fraction:
        return Fraction(self.core_result.squared_systole)

    @property
    def squared_systole(self) -> float:
        return float(self.squared_systole_coefficient) / sqrt(self.model.scale_radicand)

    @property
    def systole(self) -> float:
        return sqrt(self.squared_systole)

    @property
    def class_multiplicity(self) -> int:
        return self.core_result.class_multiplicity

    @property
    def lift_multiplicity(self) -> int:
        return self.core_result.lift_multiplicity

    @property
    def certified(self) -> bool:
        return self.core_result.certified

    @property
    def metric_convention(self) -> MetricConvention:
        return self.core_result.metric_convention


@dataclass(frozen=True)
class PeriodModel:
    """A principally polarized model with exact quadratic period data."""

    name: str
    real_part: RationalMatrix
    imaginary_core: RationalMatrix
    scale_radicand: int
    source: str
    cm_field: str

    @property
    def dimension(self) -> int:
        return len(self.real_part)

    @property
    def principal_alternating(self) -> tuple[tuple[int, ...], ...]:
        return canonical_alternating((1,) * self.dimension)

    @property
    def qubit_alternating(self) -> tuple[tuple[int, ...], ...]:
        return canonical_alternating((2,) * self.dimension)

    @property
    def metric_core(self) -> RationalMatrix:
        return period_metric_core(
            self.real_part,
            self.imaginary_core,
            self.scale_radicand,
        )

    @property
    def complex_structure_core(self) -> RationalMatrix:
        """Return ``J_num`` when ``J=J_num/sqrt(scale_radicand)``."""

        inverse_alternating = invert_rational_matrix(self.principal_alternating)
        return _scale(
            _multiply(inverse_alternating, self.metric_core),
            Fraction(-1),
        )

    @property
    def metric_scale(self) -> float:
        return 1.0 / sqrt(self.scale_radicand)

    @property
    def metric_numeric(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) * self.metric_scale for value in row)
            for row in self.metric_core
        )

    @property
    def period_numeric(self) -> tuple[tuple[complex, ...], ...]:
        return tuple(
            tuple(
                complex(
                    float(self.real_part[row][column]),
                    float(self.imaginary_core[row][column]) * self.metric_scale,
                )
                for column in range(self.dimension)
            )
            for row in range(self.dimension)
        )

    def validation_certificate(self) -> PPAVValidationResult:
        if self.real_part != _transpose(self.real_part):
            raise ValueError("period real part is not symmetric")
        Metric(self.imaginary_core)
        return validate_ppav_data(
            self.metric_core,
            self.complex_structure_core,
            self.principal_alternating,
            scale_radicand=self.scale_radicand,
        )

    def validate(self) -> None:
        """Raise unless the period data define the claimed exact PPAV."""

        self.validation_certificate()

    def compute_qubit_systole(self) -> ScaledSystoleResult:
        return self.compute_uniform_systole(2)

    def compute_uniform_systole(self, d: int) -> ScaledSystoleResult:
        """Compute the type ``(d,...,d)`` relative systole."""

        if d <= 1:
            raise ValueError("uniform polarization level must be greater than one")
        self.validate()
        core_result = compute_relative_systole(
            canonical_alternating((d,) * self.dimension),
            self.metric_core,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        return ScaledSystoleResult(self, core_result)

    def compute_uniform_systole_via_svp(self, d: int) -> ScaledUniformSystoleResult:
        """Compute a uniform type using one exact principal-lattice SVP."""

        self.validate()
        core_result = compute_uniform_relative_systole(
            self.metric_core,
            d,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        return ScaledUniformSystoleResult(self, core_result)


D4_PERIOD_MODEL = PeriodModel(
    name="D4 principally polarized abelian surface",
    real_part=(
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    ),
    imaginary_core=(
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    ),
    scale_radicand=2,
    source=(
        "Derived in this repository from the covolume-one symplectic D4 lattice; "
        "Conway--Sloane, appendix to Buser--Sarnak (1994), DOI 10.1007/BF01232233."
    ),
    cm_field="Q(sqrt(-2))",
)


KLEIN_QUARTIC_PERIOD_MODEL = PeriodModel(
    name="Jacobian of the Klein quartic",
    real_part=(
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1, 2), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1, 2)),
    ),
    imaginary_core=(
        (Fraction(3), Fraction(2), Fraction(2)),
        (Fraction(2), Fraction(3), Fraction(2)),
        (Fraction(2), Fraction(2), Fraction(3)),
    ),
    scale_radicand=28,
    source=(
        "Bochnak--Kucharz--Silhol (1997), Example 4.16, "
        "Publications Mathématiques de l'IHÉS 86, pp. 61--62."
    ),
    cm_field="Q(sqrt(-7))",
)


VERIFIED_PERIOD_MODELS = (D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL)


D4_ROOT_GRAM = (
    (2, -1, 0, 0),
    (-1, 2, -1, -1),
    (0, -1, 2, 0),
    (0, -1, 0, 2),
)

D4_PRINCIPAL_ALTERNATING_ROOT_BASIS = (
    (0, -1, 0, 1),
    (1, 0, 0, -1),
    (0, 0, 0, 1),
    (-1, 1, -1, 0),
)

D4_SYMPLECTIC_CHANGE = (
    (-1, -1, -1, -1),
    (-1, -1, -1, 0),
    (-1, 0, 0, 0),
    (-1, 0, -1, 0),
)


def validate_d4_derivation() -> None:
    """Verify the period model is the normalized symplectic D4 lattice."""

    root_metric = _as_fraction_matrix(D4_ROOT_GRAM)
    root_alternating = _as_fraction_matrix(D4_PRINCIPAL_ALTERNATING_ROOT_BASIS)
    change = _as_fraction_matrix(D4_SYMPLECTIC_CHANGE)
    if abs(determinant(D4_SYMPLECTIC_CHANGE)) != 1:
        raise ArithmeticError("D4 basis change is not unimodular")
    transformed_alternating = _multiply(
        _multiply(_transpose(change), root_alternating),
        change,
    )
    if transformed_alternating != _as_fraction_matrix(
        D4_PERIOD_MODEL.principal_alternating
    ):
        raise ArithmeticError("D4 basis change is not symplectic")
    transformed_metric = _multiply(
        _multiply(_transpose(change), root_metric),
        change,
    )
    if transformed_metric != D4_PERIOD_MODEL.metric_core:
        raise ArithmeticError("D4 period metric does not match the root lattice")

    # Compatibility in the root basis, before the symplectic change.
    inverse_alternating = invert_rational_matrix(root_alternating)
    complex_core = _scale(
        _multiply(inverse_alternating, root_metric),
        Fraction(-1),
    )
    square = _multiply(complex_core, complex_core)
    expected = tuple(
        tuple(Fraction(-2 if row == column else 0) for column in range(4))
        for row in range(4)
    )
    if square != expected:
        raise ArithmeticError("normalized D4 metric is not polarization-compatible")
