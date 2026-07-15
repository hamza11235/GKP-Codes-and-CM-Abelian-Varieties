"""Exact ``E8`` principally polarized abelian fourfold benchmark.

We realize the ``E8`` lattice in its standard coordinate model

``D8 union (D8 + (1/2,...,1/2))``

and use the paired quarter-turn

``(x1,x2,...,x7,x8) -> (x2,-x1,...,x8,-x7)``.

The rotation preserves ``E8``.  In the integral basis below it therefore gives
an integral complex structure ``J``.  With the Euclidean Gram matrix ``G``, the
form ``A=GJ`` is integral, alternating, and unimodular.  Hence ``(G,J,A)`` is
an exact principally polarized abelian fourfold.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .conventions import MetricConvention
from .kernel import invert_rational_matrix
from .ppav import PPAVValidationResult, validate_ppav_data
from .systole import RelativeSystoleResult, compute_relative_systole
from .uniform import UniformRelativeSystoleResult, compute_uniform_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    )


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _integer_matrix(matrix: RationalMatrix) -> tuple[tuple[int, ...], ...]:
    if any(value.denominator != 1 for row in matrix for value in row):
        raise ArithmeticError("the E8 coordinate transformation is not integral")
    return tuple(tuple(value.numerator for value in row) for row in matrix)


def _e8_basis_vectors() -> tuple[tuple[Fraction, ...], ...]:
    """Return an integral basis of E8 as eight Euclidean coordinate vectors."""

    half_vector = (Fraction(1, 2),) * 8
    d8_roots: list[tuple[Fraction, ...]] = []
    for index in range(7):
        root = [Fraction(0) for _ in range(8)]
        root[index] = Fraction(1)
        root[index + 1] = Fraction(-1)
        d8_roots.append(tuple(root))
    final_root = [Fraction(0) for _ in range(8)]
    final_root[6] = Fraction(1)
    final_root[7] = Fraction(1)
    d8_roots.append(tuple(final_root))

    # Replacing e1-e2 by the spinor vector gives a determinant-one basis of
    # D8 union (D8 + spinor).
    return (half_vector,) + tuple(d8_roots[1:])


def _basis_matrix(vectors: tuple[tuple[Fraction, ...], ...]) -> RationalMatrix:
    """Put Euclidean basis vectors in columns."""

    return tuple(
        tuple(vectors[column][row] for column in range(len(vectors)))
        for row in range(len(vectors[0]))
    )


def _paired_quarter_turn() -> RationalMatrix:
    matrix = [[Fraction(0) for _ in range(8)] for _ in range(8)]
    for index in range(0, 8, 2):
        matrix[index][index + 1] = Fraction(1)
        matrix[index + 1][index] = Fraction(-1)
    return tuple(tuple(row) for row in matrix)


E8_EUCLIDEAN_BASIS_VECTORS = _e8_basis_vectors()
E8_EUCLIDEAN_BASIS_MATRIX = _basis_matrix(E8_EUCLIDEAN_BASIS_VECTORS)
E8_EUCLIDEAN_COMPLEX_STRUCTURE = _paired_quarter_turn()

E8_GRAM: tuple[tuple[int, ...], ...] = _integer_matrix(
    _multiply(_transpose(E8_EUCLIDEAN_BASIS_MATRIX), E8_EUCLIDEAN_BASIS_MATRIX)
)

E8_COMPLEX_STRUCTURE: tuple[tuple[int, ...], ...] = _integer_matrix(
    _multiply(
        _multiply(
            invert_rational_matrix(E8_EUCLIDEAN_BASIS_MATRIX),
            E8_EUCLIDEAN_COMPLEX_STRUCTURE,
        ),
        E8_EUCLIDEAN_BASIS_MATRIX,
    )
)

E8_PRINCIPAL_ALTERNATING: tuple[tuple[int, ...], ...] = _integer_matrix(
    _multiply(
        tuple(tuple(Fraction(value) for value in row) for row in E8_GRAM),
        tuple(
            tuple(Fraction(value) for value in row)
            for row in E8_COMPLEX_STRUCTURE
        ),
    )
)


@dataclass(frozen=True)
class E8BenchmarkModel:
    """The exact ``E8`` PPAV and its uniform GKP-code benchmarks."""

    name: str = "E8 principally polarized abelian fourfold"
    source: str = (
        "Standard E8 coordinate lattice with a paired quarter-turn complex "
        "structure; packing optimality follows from Viazovska (2016)."
    )
    cm_description: str = "Gaussian CM; unpolarized torus is isogenous to E_i^4"

    @property
    def dimension(self) -> int:
        return 4

    @property
    def metric(self) -> tuple[tuple[int, ...], ...]:
        return E8_GRAM

    @property
    def complex_structure(self) -> tuple[tuple[int, ...], ...]:
        return E8_COMPLEX_STRUCTURE

    @property
    def principal_alternating(self) -> tuple[tuple[int, ...], ...]:
        return E8_PRINCIPAL_ALTERNATING

    def validation_certificate(self) -> PPAVValidationResult:
        return validate_ppav_data(
            self.metric,
            self.complex_structure,
            self.principal_alternating,
        )

    def validate(self) -> None:
        self.validation_certificate()

    def uniform_alternating(self, level: int) -> tuple[tuple[int, ...], ...]:
        if level <= 1:
            raise ValueError("uniform polarization level must be greater than one")
        return tuple(
            tuple(level * value for value in row)
            for row in self.principal_alternating
        )

    def compute_uniform_systole(self, level: int = 2) -> UniformRelativeSystoleResult:
        self.validate()
        return compute_uniform_relative_systole(
            self.metric,
            level,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )

    def compute_full_uniform_systole(
        self,
        level: int = 2,
    ) -> RelativeSystoleResult:
        """Independent full-kernel CVP calculation for validation."""

        self.validate()
        return compute_relative_systole(
            self.uniform_alternating(level),
            self.metric,
            metric_convention=MetricConvention.FIXED_PRINCIPAL,
        )

    def compute_qubit_systole(self) -> UniformRelativeSystoleResult:
        return self.compute_uniform_systole(2)


E8_PPAV_MODEL = E8BenchmarkModel()


def validate_e8_derivation() -> None:
    """Re-run every exact structural check in the E8 construction."""

    certificate = E8_PPAV_MODEL.validation_certificate()
    if certificate.dimension != 4:
        raise ArithmeticError("E8 must define a complex fourfold")
    if certificate.polarization_type != (1, 1, 1, 1):
        raise ArithmeticError("the E8 Riemann form is not principal")
    if certificate.physical_metric_determinant != 1:
        raise ArithmeticError("the E8 lattice does not have covolume one")
