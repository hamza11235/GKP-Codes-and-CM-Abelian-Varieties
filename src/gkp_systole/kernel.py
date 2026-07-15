"""Exact enumeration of the finite logical displacement group."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Iterable, Sequence

from .polarization import Polarization


RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _lcm(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return abs(left * right) // gcd(left, right)


def _fractional_part(value: Fraction) -> Fraction:
    """Return the unique representative of value modulo Z in [0, 1)."""

    return value - value.numerator // value.denominator


def canonical_mod_integer(vector: Iterable[Fraction]) -> RationalVector:
    """Reduce a rational vector coordinatewise modulo the integer lattice."""

    return tuple(_fractional_part(Fraction(value)) for value in vector)


def invert_rational_matrix(
    matrix: Sequence[Sequence[Fraction | int]],
) -> RationalMatrix:
    """Return an exact rational inverse using Gauss--Jordan elimination."""

    rows = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    size = len(rows)
    if not rows or any(len(row) != size for row in rows):
        raise ValueError("matrix must be nonempty and square")

    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(int(row_index == column_index)) for column_index in range(size)]
        for row_index, row in enumerate(rows)
    ]

    for column in range(size):
        pivot_row = next(
            (row for row in range(column, size) if augmented[row][column] != 0),
            None,
        )
        if pivot_row is None:
            raise ValueError("matrix must be nonsingular")
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            coefficient = augmented[row][column]
            if coefficient == 0:
                continue
            augmented[row] = [
                value - coefficient * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]

    return tuple(tuple(row[size:]) for row in augmented)


def invert_integer_matrix(matrix: Sequence[Sequence[int]]) -> RationalMatrix:
    """Backward-compatible exact inverse for an integral matrix."""

    return invert_rational_matrix(matrix)


def transpose(matrix: Sequence[Sequence[Fraction | int]]) -> RationalMatrix:
    rows = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be nonempty and rectangular")
    return tuple(
        tuple(rows[row][column] for row in range(len(rows)))
        for column in range(len(rows[0]))
    )


def matvec(
    matrix: Sequence[Sequence[Fraction | int]],
    vector: Sequence[Fraction | int],
) -> RationalVector:
    return tuple(
        sum(
            (Fraction(value) * Fraction(coordinate) for value, coordinate in zip(row, vector)),
            Fraction(0),
        )
        for row in matrix
    )


@dataclass(frozen=True, order=True)
class KernelElement:
    """A canonical representative of a class in Lambda-perp/Lambda."""

    coordinates: RationalVector

    def __post_init__(self) -> None:
        canonical = canonical_mod_integer(self.coordinates)
        object.__setattr__(self, "coordinates", canonical)

    @property
    def is_zero(self) -> bool:
        return all(value == 0 for value in self.coordinates)

    @property
    def order(self) -> int:
        """Additive order of this element in the quotient group."""

        return reduce(_lcm, (value.denominator for value in self.coordinates), 1)

    def __add__(self, other: "KernelElement") -> "KernelElement":
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("kernel elements must have the same dimension")
        return KernelElement(
            tuple(left + right for left, right in zip(self.coordinates, other.coordinates))
        )

    def as_strings(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.coordinates)


@dataclass(frozen=True)
class KernelGroup:
    """The finite group K = Lambda-perp/Lambda of a polarization."""

    polarization: Polarization
    generators: tuple[KernelElement, ...]
    elements: tuple[KernelElement, ...]

    @classmethod
    def from_polarization(cls, polarization: Polarization) -> "KernelGroup":
        alternating_transpose = transpose(polarization.matrix)
        inverse_transpose = invert_integer_matrix(alternating_transpose)

        # Columns of A^{-T} generate Lambda-perp in lattice coordinates.
        generators = tuple(
            KernelElement(
                tuple(
                    inverse_transpose[row][column]
                    for row in range(len(inverse_transpose))
                )
            )
            for column in range(len(inverse_transpose))
        )
        generators = tuple(generator for generator in generators if not generator.is_zero)

        zero = KernelElement(tuple(Fraction(0) for _ in range(2 * polarization.dimension)))
        seen = {zero}
        queue = deque([zero])
        while queue:
            element = queue.popleft()
            for generator in generators:
                candidate = element + generator
                if candidate not in seen:
                    seen.add(candidate)
                    queue.append(candidate)
                    if len(seen) > polarization.kernel_order:
                        raise ArithmeticError(
                            "kernel enumeration exceeded the determinant bound"
                        )

        if len(seen) != polarization.kernel_order:
            raise ArithmeticError(
                "kernel enumeration did not recover the expected number of classes: "
                f"found {len(seen)}, expected {polarization.kernel_order}"
            )

        elements = tuple(sorted(seen))
        group = cls(
            polarization=polarization,
            generators=generators,
            elements=elements,
        )
        group._verify_dual_membership()
        return group

    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def nonzero_elements(self) -> tuple[KernelElement, ...]:
        return tuple(element for element in self.elements if not element.is_zero)

    @property
    def exponent(self) -> int:
        return reduce(_lcm, (element.order for element in self.elements), 1)

    def _verify_dual_membership(self) -> None:
        alternating_transpose = transpose(self.polarization.matrix)
        for element in self.elements:
            image = matvec(alternating_transpose, element.coordinates)
            if any(value.denominator != 1 for value in image):
                raise ArithmeticError(
                    "enumerated representative is not in the symplectic dual lattice"
                )
