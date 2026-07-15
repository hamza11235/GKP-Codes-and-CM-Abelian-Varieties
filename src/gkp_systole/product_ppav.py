"""Certified products of exact principally polarized abelian varieties.

Products are not expected to be higher-dimensional optimizers: their shortest
vector is bottlenecked by the weakest factor.  They are nevertheless essential
baselines.  They provide exact higher-dimensional PPAV inputs, known scaling
laws, and a way to exercise the uniform-type SVP solver without enumerating an
exponentially large kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Sequence

from .cm import ReducedQuadraticForm, cm_elliptic_period_model
from .conventions import MetricConvention
from .models import PeriodModel
from .ppav import PPAVValidationResult, validate_ppav_data
from .uniform import UniformRelativeSystoleResult, compute_uniform_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


def block_diagonal(
    matrices: Sequence[Sequence[Sequence[int | Fraction]]],
) -> RationalMatrix:
    """Return the exact block diagonal of nonempty square matrices."""

    blocks = tuple(
        tuple(tuple(Fraction(value) for value in row) for row in matrix)
        for matrix in matrices
    )
    if not blocks:
        raise ValueError("at least one matrix block is required")
    if any(not block or any(len(row) != len(block) for row in block) for block in blocks):
        raise ValueError("every block must be a nonempty square matrix")

    size = sum(len(block) for block in blocks)
    result = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    offset = 0
    for block in blocks:
        block_size = len(block)
        for row in range(block_size):
            for column in range(block_size):
                result[offset + row][offset + column] = block[row][column]
        offset += block_size
    return tuple(tuple(row) for row in result)


def _integer_block_diagonal(
    matrices: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[int, ...], ...]:
    rational = block_diagonal(matrices)
    if any(value.denominator != 1 for row in rational for value in row):
        raise ArithmeticError("integral block diagonal unexpectedly became nonintegral")
    return tuple(tuple(value.numerator for value in row) for row in rational)


@dataclass(frozen=True)
class ProductUniformSystoleResult:
    """Uniform-type SVP result with the product's algebraic metric scale."""

    model: "ProductPPAVModel"
    core_result: UniformRelativeSystoleResult

    @property
    def lambda1_squared_coefficient(self) -> Fraction:
        return Fraction(self.core_result.lambda1_squared)

    @property
    def lambda1_squared(self) -> float:
        return float(self.lambda1_squared_coefficient) / sqrt(
            self.model.scale_radicand
        )

    @property
    def squared_systole_coefficient(self) -> Fraction:
        return Fraction(self.core_result.squared_systole)

    @property
    def squared_systole(self) -> float:
        return float(self.squared_systole_coefficient) / sqrt(
            self.model.scale_radicand
        )

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
    def full_kernel_class_count(self) -> int:
        """Number of nonzero classes the general CVP path would enumerate."""

        return self.core_result.level ** (2 * self.model.dimension) - 1


@dataclass(frozen=True)
class ProductPPAVModel:
    """An exact product of PPAV factors sharing one square-root scale."""

    name: str
    factors: tuple[PPAVValidationResult, ...]

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("a product requires at least one PPAV factor")
        if any(not factor.principal for factor in self.factors):
            raise ValueError("all product factors must be principally polarized")
        radicands = {factor.scale_radicand for factor in self.factors}
        if len(radicands) != 1:
            raise ValueError(
                "exact products currently require a common scale radicand"
            )
        self.validation_certificate()

    @classmethod
    def repeat(
        cls,
        name: str,
        factor: PPAVValidationResult,
        copies: int,
    ) -> "ProductPPAVModel":
        if copies <= 0:
            raise ValueError("copies must be positive")
        return cls(name=name, factors=(factor,) * copies)

    @property
    def dimension(self) -> int:
        return sum(factor.dimension for factor in self.factors)

    @property
    def scale_radicand(self) -> int:
        return self.factors[0].scale_radicand

    @property
    def metric_core(self) -> RationalMatrix:
        return block_diagonal(tuple(factor.metric_core for factor in self.factors))

    @property
    def complex_structure_core(self) -> RationalMatrix:
        return block_diagonal(
            tuple(factor.complex_structure_numerator for factor in self.factors)
        )

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        return _integer_block_diagonal(
            tuple(factor.polarization.matrix for factor in self.factors)
        )

    def validation_certificate(self) -> PPAVValidationResult:
        return validate_ppav_data(
            self.metric_core,
            self.complex_structure_core,
            self.alternating,
            scale_radicand=self.scale_radicand,
        )

    def compute_uniform_systole(self, level: int = 2) -> ProductUniformSystoleResult:
        self.validation_certificate()
        core_result = compute_uniform_relative_systole(
            self.metric_core,
            level,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )
        return ProductUniformSystoleResult(self, core_result)


def repeated_period_model(model: PeriodModel, copies: int) -> ProductPPAVModel:
    """Repeat one verified exact period model as a product PPAV."""

    return ProductPPAVModel.repeat(
        f"({model.name})^{copies}",
        model.validation_certificate(),
        copies,
    )


def square_product_model(dimension: int) -> ProductPPAVModel:
    """Return the decomposable square-CM PPAV ``E_i^dimension``."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    factor = validate_ppav_data(
        ((1, 0), (0, 1)),
        ((0, 1), (-1, 0)),
    )
    return ProductPPAVModel.repeat(f"square CM product g={dimension}", factor, dimension)


def hexagonal_product_model(dimension: int) -> ProductPPAVModel:
    """Return the decomposable Eisenstein-CM PPAV ``E_omega^dimension``."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    factor = cm_elliptic_period_model(
        ReducedQuadraticForm(1, 1, 1)
    ).validation_certificate()
    return ProductPPAVModel.repeat(
        f"hexagonal CM product g={dimension}", factor, dimension
    )
