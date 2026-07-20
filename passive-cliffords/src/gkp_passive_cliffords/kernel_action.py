"""Induced finite symplectic action of polarized automorphisms on K(L)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from gkp_systole.kernel import KernelElement, KernelGroup

from .automorphisms import PolarizedAutomorphismGroup
from .exact import IntegerMatrix, canonical_mod_integer


Permutation = tuple[int, ...]


def kernel_pairing(
    kernel: KernelGroup, left: KernelElement, right: KernelElement
) -> Fraction:
    """Return the additive commutator pairing in Q/Z, represented in [0,1)."""

    alternating = kernel.polarization.matrix
    value = sum(
        (
            left.coordinates[row]
            * alternating[row][column]
            * right.coordinates[column]
            for row in range(len(alternating))
            for column in range(len(alternating))
        ),
        Fraction(0),
    )
    return value - value.numerator // value.denominator


def act_on_kernel_element(
    matrix: Sequence[Sequence[int]], element: KernelElement
) -> KernelElement:
    image = tuple(
        sum(
            (Fraction(value) * coordinate for value, coordinate in zip(row, element.coordinates)),
            Fraction(0),
        )
        for row in matrix
    )
    return KernelElement(canonical_mod_integer(image))


def induced_permutation(matrix: IntegerMatrix, kernel: KernelGroup) -> Permutation:
    positions = {element: index for index, element in enumerate(kernel.elements)}
    images = tuple(act_on_kernel_element(matrix, element) for element in kernel.elements)
    try:
        permutation = tuple(positions[image] for image in images)
    except KeyError as error:
        raise ArithmeticError("automorphism did not preserve the polarization kernel") from error
    if len(set(permutation)) != kernel.order:
        raise ArithmeticError("induced kernel action is not bijective")
    return permutation


def preserves_kernel_pairing(
    permutation: Permutation, kernel: KernelGroup
) -> bool:
    table = _kernel_pairing_table(kernel)
    return _permutation_preserves_pairing(permutation, table)


def _kernel_pairing_table(kernel: KernelGroup) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(kernel_pairing(kernel, left, right) for right in kernel.elements)
        for left in kernel.elements
    )


def _permutation_preserves_pairing(
    permutation: Permutation,
    table: tuple[tuple[Fraction, ...], ...],
) -> bool:
    return all(
        table[left][right] == table[permutation[left]][permutation[right]]
        for left in range(len(table))
        for right in range(len(table))
    )


def _permutation_preserves_pairing_on_generators(
    permutation: Permutation,
    kernel: KernelGroup,
    positions: dict[KernelElement, int],
) -> bool:
    """Check a bilinear pairing on a generating set instead of all of K x K."""

    for left in kernel.generators:
        left_index = positions[left]
        left_image = kernel.elements[permutation[left_index]]
        for right in kernel.generators:
            right_index = positions[right]
            right_image = kernel.elements[permutation[right_index]]
            if kernel_pairing(kernel, left, right) != kernel_pairing(
                kernel, left_image, right_image
            ):
                return False
    return True


@dataclass(frozen=True)
class LogicalActionResult:
    """Exact passive-Clifford image and kernel for a fully enumerated group."""

    automorphism_group: PolarizedAutomorphismGroup
    kernel_group: KernelGroup
    action_by_automorphism: tuple[Permutation, ...]
    image: tuple[Permutation, ...]
    kernel_automorphisms: tuple[IntegerMatrix, ...]

    @property
    def automorphism_order(self) -> int:
        return self.automorphism_group.order

    @property
    def image_order(self) -> int:
        return len(self.image)

    @property
    def action_kernel_order(self) -> int:
        return len(self.kernel_automorphisms)

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension_g": self.kernel_group.polarization.dimension,
            "polarization_type": self.kernel_group.polarization.type,
            "kernel_order": self.kernel_group.order,
            "polarized_automorphism_order": self.automorphism_order,
            "logical_image_order": self.image_order,
            "action_kernel_order": self.action_kernel_order,
            "pairing_verified": True,
        }


@dataclass(frozen=True)
class LogicalActionOrders:
    """Memory-light image/kernel orders determined on kernel generators."""

    automorphism_order: int
    image_order: int
    action_kernel_order: int
    pairing_verified: bool = True


def compute_logical_action_orders(
    automorphism_group: PolarizedAutomorphismGroup,
) -> LogicalActionOrders:
    """Compute only action orders, avoiding full permutations of every kernel point.

    The images of a generating set determine an automorphism of the finite
    kernel.  Pairing preservation is checked once for each distinct image,
    which makes population-scale surveys much faster when many geometric
    automorphisms have the same logical action.
    """

    kernel = KernelGroup.from_polarization(automorphism_group.problem.polarization)
    generators = kernel.generators
    identity_signature = generators
    signatures: set[tuple[KernelElement, ...]] = set()
    kernel_order = 0
    for matrix in automorphism_group.elements:
        signature = tuple(act_on_kernel_element(matrix, generator) for generator in generators)
        signatures.add(signature)
        if signature == identity_signature:
            kernel_order += 1

    for signature in signatures:
        for left_index, left in enumerate(generators):
            for right_index, right in enumerate(generators):
                if kernel_pairing(kernel, left, right) != kernel_pairing(
                    kernel,
                    signature[left_index],
                    signature[right_index],
                ):
                    raise ArithmeticError("an induced generator action failed pairing preservation")

    image_order = len(signatures)
    if automorphism_group.order != kernel_order * image_order:
        raise ArithmeticError("finite group image/kernel orders failed the homomorphism identity")
    return LogicalActionOrders(
        automorphism_order=automorphism_group.order,
        image_order=image_order,
        action_kernel_order=kernel_order,
    )


def compute_logical_action(
    automorphism_group: PolarizedAutomorphismGroup,
) -> LogicalActionResult:
    kernel = KernelGroup.from_polarization(automorphism_group.problem.polarization)
    permutations = tuple(
        induced_permutation(matrix, kernel) for matrix in automorphism_group.elements
    )
    positions = {element: index for index, element in enumerate(kernel.elements)}
    if any(
        not _permutation_preserves_pairing_on_generators(
            permutation, kernel, positions
        )
        for permutation in permutations
    ):
        raise ArithmeticError("an induced action failed to preserve the commutator pairing")

    identity = tuple(range(kernel.order))
    kernel_automorphisms = tuple(
        matrix
        for matrix, permutation in zip(automorphism_group.elements, permutations)
        if permutation == identity
    )
    image = tuple(sorted(set(permutations)))
    if automorphism_group.order != len(kernel_automorphisms) * len(image):
        raise ArithmeticError("finite group image/kernel orders failed the homomorphism identity")
    return LogicalActionResult(
        automorphism_group=automorphism_group,
        kernel_group=kernel,
        action_by_automorphism=permutations,
        image=image,
        kernel_automorphisms=kernel_automorphisms,
    )
