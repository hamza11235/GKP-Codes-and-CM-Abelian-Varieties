"""Phase-7 consolidation helpers.

This module turns arithmetic CM candidates into full-dimensional compatible
metric search centers, supplies a reproducible result-ledger schema, and gives
the cyclotomic-five benchmark an exhaustive interval-arithmetic certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import product
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .kernel import KernelGroup
from .moduli_search import CompatibleMetricFamily
from .polarization import Polarization
from .quadratic_hermitian import QuadraticHermitianForm
from .quartic_cm import CyclotomicFivePolarization


def quadratic_hermitian_moduli_family(
    form: QuadraticHermitianForm,
    *,
    name: str | None = None,
) -> CompatibleMetricFamily:
    """Use an exact quadratic-order candidate as a six-dimensional center."""

    form.validate()
    radicand = form.order.radicand
    physical_metric = tuple(
        tuple(float(value) / sqrt(radicand) for value in row)
        for row in form.metric_core
    )
    core = form.compute_core_relative_systole()
    coefficient = core.squared_systole
    return CompatibleMetricFamily.from_reference(
        name=name or f"type {form.polarization_type} around {form.order.label}",
        alternating=form.alternating,
        reference_metric=physical_metric,
        reference_exact_ell_squared=f"({coefficient})/sqrt({radicand})",
        reference_ell_squared=float(coefficient) / sqrt(radicand),
        reference_cm=(
            f"E_{form.order.discriminant}^2 with Hermitian form "
            f"({form.a},{form.c},{form.first},{form.second})"
        ),
    )


def cyclotomic_five_moduli_family(
    polarization: CyclotomicFivePolarization,
    *,
    name: str | None = None,
) -> CompatibleMetricFamily:
    """Use a simple ``Q(zeta_5)`` polarization as a moduli-search center."""

    polarization.validate()
    result = polarization.compute_relative_systole()
    return CompatibleMetricFamily.from_reference(
        name=name or f"type {polarization.polarization_type} around Q(zeta_5)",
        alternating=polarization.alternating,
        reference_metric=polarization.metric,
        reference_exact_ell_squared="interval-certified algebraic value",
        reference_ell_squared=float(result.squared_systole),
        reference_cm=(
            "Q(zeta_5), primitive CM type {1,2}, "
            f"alpha=({polarization.m},{polarization.n})"
        ),
    )


@dataclass(frozen=True)
class IntervalSystoleCertificate:
    """Serializable exhaustive interval certificate for one relative systole."""

    decimal_places: int
    lower_bound: str
    upper_bound: str
    interval_width: str
    class_multiplicity: int
    lift_multiplicity: int
    next_candidate_lower_bound: str
    separation_gap: str
    maximum_enumeration_radius: str
    algebraic_expression: str | None = None
    annihilating_polynomial: tuple[int, ...] | None = None
    certified: bool = True


def _interval_bounds(value, mp):
    return mp.mpf(value.a._mpi_[0]), mp.mpf(value.b._mpi_[1])


def _cyclotomic_five_interval_metric(
    polarization: CyclotomicFivePolarization,
    decimal_places: int,
):
    import mpmath as mp

    mp.mp.dps = decimal_places
    mp.iv.dps = decimal_places
    rows = [
        [mp.iv.cos(2 * mp.iv.pi * embedding * power / 5) for power in range(4)]
        for embedding in (1, 2)
    ]
    rows += [
        [mp.iv.sin(2 * mp.iv.pi * embedding * power / 5) for power in range(4)]
        for embedding in (1, 2)
    ]
    basis = mp.iv.matrix(rows)
    standard = mp.iv.matrix(
        ((0, 0, -1, 0), (0, 0, 0, -1), (1, 0, 0, 0), (0, 1, 0, 0))
    )
    structure = basis**-1 * standard * basis
    alternating = mp.iv.matrix(polarization.alternating)
    metric = -alternating * structure
    return (metric + metric.T) / 2


def certify_cyclotomic_five_systole_interval(
    polarization: CyclotomicFivePolarization,
    *,
    decimal_places: int = 70,
) -> IntervalSystoleCertificate:
    """Certify the cyclotomic relative systole by exhaustive interval CVP.

    A Gershgorin lower bound for the interval metric bounds every integer lift
    that could improve the initial candidate.  Every lift inside that finite
    region is then evaluated with outward-rounded ``mpmath.iv`` arithmetic.
    The returned gap separates all shortest lifts from the next candidate.
    """

    if decimal_places < 30:
        raise ValueError("interval certification requires at least 30 decimal places")
    import mpmath as mp

    polarization.validate()
    metric = _cyclotomic_five_interval_metric(polarization, decimal_places)
    size = 4
    bounds = [[_interval_bounds(metric[row, column], mp) for column in range(size)] for row in range(size)]

    gershgorin = []
    for row in range(size):
        diagonal_lower = bounds[row][row][0]
        off_diagonal_upper = mp.fsum(
            max(abs(bounds[row][column][0]), abs(bounds[row][column][1]))
            for column in range(size)
            if column != row
        )
        gershgorin.append(diagonal_lower - off_diagonal_upper)
    eigenvalue_lower = min(gershgorin)
    if eigenvalue_lower <= 0:
        raise ArithmeticError("interval Gershgorin bound did not certify positivity")

    def quadratic_interval(coordinates: Sequence[Fraction], shift: Sequence[int]):
        vector = [
            mp.iv.mpf(value.numerator) / value.denominator + integer
            for value, integer in zip(coordinates, shift)
        ]
        result = mp.iv.mpf(0)
        for row in range(size):
            for column in range(size):
                result += vector[row] * metric[row, column] * vector[column]
        return result

    all_candidates = []
    maximum_radius = mp.mpf("0")
    kernel = KernelGroup.from_polarization(Polarization(polarization.alternating))
    for element in kernel.nonzero_elements:
        initial = tuple(round(-float(value)) for value in element.coordinates)
        initial_interval = quadratic_interval(element.coordinates, initial)
        _, initial_upper = _interval_bounds(initial_interval, mp)
        radius = mp.sqrt(initial_upper / eigenvalue_lower)
        maximum_radius = max(maximum_radius, radius)
        ranges = tuple(
            range(
                int(mp.ceil(-mp.mpf(value.numerator) / value.denominator - radius)),
                int(mp.floor(-mp.mpf(value.numerator) / value.denominator + radius)) + 1,
            )
            for value in element.coordinates
        )
        class_candidates = []
        for shift in product(*ranges):
            euclidean_squared = mp.fsum(
                (mp.mpf(value.numerator) / value.denominator + integer) ** 2
                for value, integer in zip(element.coordinates, shift)
            )
            if eigenvalue_lower * euclidean_squared > initial_upper:
                continue
            interval = quadratic_interval(element.coordinates, shift)
            lower, upper = _interval_bounds(interval, mp)
            class_candidates.append((lower, upper, element, tuple(shift)))
        if not class_candidates:
            raise ArithmeticError("interval enumeration found no lift for a kernel class")
        class_upper = min(item[1] for item in class_candidates)
        all_candidates.extend(item for item in class_candidates if item[0] <= class_upper)

    global_upper = min(item[1] for item in all_candidates)
    possible_shortest = [item for item in all_candidates if item[0] <= global_upper]
    global_lower = min(item[0] for item in possible_shortest)
    shortest_upper = max(item[1] for item in possible_shortest)
    remaining = [item for item in all_candidates if item not in possible_shortest]
    next_lower = min(item[0] for item in remaining) if remaining else mp.inf
    gap = next_lower - shortest_upper
    common_lower = max(item[0] for item in possible_shortest)
    common_upper = min(item[1] for item in possible_shortest)
    certified = gap > 0 and common_lower <= common_upper
    if not certified:
        raise ArithmeticError("intervals did not separate the shortest lifts")

    shortest_classes = {item[2] for item in possible_shortest}
    shortest_lifts = {(item[2], item[3]) for item in possible_shortest}
    digits = decimal_places
    expression = None
    polynomial = None
    if (polarization.m, polarization.n) == (2, -1):
        expression = "sqrt(4/25 + 8*sqrt(5)/125)"
        polynomial = (3125, 0, -1000, 0, 16)

    return IntervalSystoleCertificate(
        decimal_places=decimal_places,
        lower_bound=mp.nstr(global_lower, digits),
        upper_bound=mp.nstr(global_upper, digits),
        interval_width=mp.nstr(global_upper - global_lower, digits),
        class_multiplicity=len(shortest_classes),
        lift_multiplicity=len(shortest_lifts),
        next_candidate_lower_bound=mp.nstr(next_lower, digits),
        separation_gap=mp.nstr(gap, digits),
        maximum_enumeration_radius=mp.nstr(maximum_radius, digits),
        algebraic_expression=expression,
        annihilating_polynomial=polynomial,
        certified=True,
    )


@dataclass(frozen=True)
class SystoleLedgerEntry:
    candidate_id: str
    phase: int
    dimension_g: int
    polarization_type: str
    family: str
    cm_data: str
    ell_squared_decimal: str
    ell_squared_exact: str
    class_multiplicity: int
    lift_multiplicity: int
    metric_convention: str
    arithmetic_status: str
    search_status: str
    search_scope: str
    notes: str = ""


def write_systole_ledger(
    entries: Iterable[SystoleLedgerEntry],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write the same sorted ledger to JSON and CSV."""

    records = sorted(entries, key=lambda entry: (entry.dimension_g, entry.polarization_type, entry.candidate_id))
    dictionaries = [asdict(entry) for entry in records]
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.write_text(json.dumps(dictionaries, indent=2) + "\n", encoding="utf-8")
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(SystoleLedgerEntry.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(dictionaries)
