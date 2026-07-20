"""Phase 5 bounded-population survey of CM distance and passive symmetry.

The population is defined *before* inspecting the gate results.  It reuses the
documented Phase-7 binary and Phase-8 ternary Hermitian candidate bounds from
the systole project.  These are elementary-reduced bounded candidate lists,
not complete isometry-class enumerations and not samples from a canonical
probability measure on the CM locus.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import log, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from gkp_systole import (
    ImaginaryQuadraticOrder,
    QuadraticHermitianForm,
    TernaryQuadraticHermitianForm,
    bounded_quadratic_hermitian_forms,
    bounded_ternary_hermitian_forms,
)

from .finite_symplectic import elementary_prime_symplectic_order
from .hermitian_automorphisms import enumerate_hermitian_cm_automorphisms
from .kernel_action import compute_logical_action_orders
from .release_io import load_json_artifact


HermitianForm = QuadraticHermitianForm | TernaryQuadraticHermitianForm


@dataclass(frozen=True)
class Phase5PopulationSpec:
    dimension_g: int
    polarization_type: tuple[int, ...]
    determinant: int
    maximum_discriminant_absolute: int
    maximum_diagonal: int
    off_diagonal_bound: int | None = None


PHASE5_POPULATION_SPECS = (
    Phase5PopulationSpec(2, (1, 3), 3, 160, 16),
    Phase5PopulationSpec(2, (1, 5), 5, 160, 16),
    Phase5PopulationSpec(3, (1, 1, 2), 2, 40, 5, 1),
    Phase5PopulationSpec(3, (1, 1, 3), 3, 40, 5, 1),
    Phase5PopulationSpec(3, (1, 2, 2), 4, 40, 5, 1),
)


def phase5_discriminants(maximum_absolute: int) -> tuple[int, ...]:
    return tuple(
        -absolute
        for absolute in range(3, maximum_absolute + 1)
        if (-absolute) % 4 in (0, 1)
    )


def phase5_candidate_forms(spec: Phase5PopulationSpec) -> tuple[HermitianForm, ...]:
    forms: list[HermitianForm] = []
    for discriminant in phase5_discriminants(spec.maximum_discriminant_absolute):
        order = ImaginaryQuadraticOrder(discriminant)
        if spec.dimension_g == 2:
            candidates = bounded_quadratic_hermitian_forms(
                order,
                spec.determinant,
                maximum_diagonal=spec.maximum_diagonal,
            )
        elif spec.dimension_g == 3:
            candidates = bounded_ternary_hermitian_forms(
                order,
                spec.determinant,
                maximum_diagonal=spec.maximum_diagonal,
                off_diagonal_bound=int(spec.off_diagonal_bound or 0),
                requested_types=(spec.polarization_type,),
            )
        else:
            raise ValueError("Phase 5 currently supports dimensions two and three")
        forms.extend(
            form for form in candidates if form.polarization_type == spec.polarization_type
        )
    return tuple(forms)


def _form_parameters(form: HermitianForm) -> tuple[int, ...]:
    if isinstance(form, QuadraticHermitianForm):
        return (form.a, form.c, form.first, form.second)
    return (
        form.a,
        form.b,
        form.c,
        form.z12_first,
        form.z12_second,
        form.z13_first,
        form.z13_second,
        form.z23_first,
        form.z23_second,
    )


def _signed_token(value: int) -> str:
    return f"m{-value}" if value < 0 else str(value)


def _candidate_id(form: HermitianForm) -> str:
    type_token = "_".join(str(value) for value in form.polarization_type)
    parameter_token = "_".join(_signed_token(value) for value in _form_parameters(form))
    return (
        f"g{len(form.polarization_type)}_type_{type_token}_"
        f"delta_m{-form.order.discriminant}_h_{parameter_token}"
    )


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _generic_minimal_image_order(polarization_type: Sequence[int]) -> int:
    """Image order of the unavoidable group {+I,-I} on K(L)."""

    exponent = max(int(value) for value in polarization_type)
    return 1 if exponent <= 2 else 2


@dataclass(frozen=True)
class Phase5PopulationRecord:
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    discriminant: int
    radicand: int
    form_parameters: tuple[int, ...]
    determinant: int
    coupled: bool
    ell_squared_coefficient: str
    ell_squared_exact: str
    ell_squared_numeric: float
    class_multiplicity: int
    lift_multiplicity: int
    polarized_automorphism_order: int
    logical_image_order: int
    action_kernel_order: int
    generic_minimal_image_order: int
    logical_image_enhancement: str
    full_symplectic_target_order: int
    target_coverage: str
    extra_passive_symmetry: bool
    pairing_verified: bool
    deduplication_scope: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["form_parameters"] = list(self.form_parameters)
        return result


def evaluate_phase5_form(form: HermitianForm) -> Phase5PopulationRecord:
    systole = form.compute_core_relative_systole()
    coefficient = Fraction(systole.squared_systole)
    group = enumerate_hermitian_cm_automorphisms(form)
    action = compute_logical_action_orders(group)
    target = elementary_prime_symplectic_order(form.polarization_type)
    generic_image = _generic_minimal_image_order(form.polarization_type)
    enhancement = Fraction(action.image_order, generic_image)
    coverage = Fraction(action.image_order, target)
    radicand = form.order.radicand
    coefficient_text = _fraction_text(coefficient)
    return Phase5PopulationRecord(
        candidate_id=_candidate_id(form),
        dimension_g=len(form.polarization_type),
        polarization_type=form.polarization_type,
        discriminant=form.order.discriminant,
        radicand=radicand,
        form_parameters=_form_parameters(form),
        determinant=form.determinant,
        coupled=form.is_coupled,
        ell_squared_coefficient=coefficient_text,
        ell_squared_exact=f"{coefficient_text}/sqrt({radicand})",
        ell_squared_numeric=float(coefficient) / sqrt(radicand),
        class_multiplicity=systole.class_multiplicity,
        lift_multiplicity=systole.lift_multiplicity,
        polarized_automorphism_order=action.automorphism_order,
        logical_image_order=action.image_order,
        action_kernel_order=action.action_kernel_order,
        generic_minimal_image_order=generic_image,
        logical_image_enhancement=str(enhancement),
        full_symplectic_target_order=target,
        target_coverage=str(coverage),
        extra_passive_symmetry=action.image_order > generic_image,
        pairing_verified=action.pairing_verified,
        deduplication_scope=(
            "elementary binary Hermitian reduction"
            if isinstance(form, QuadraticHermitianForm)
            else "ternary unit/permutation reduction"
        ),
    )


def survey_phase5_population(
    specs: Iterable[Phase5PopulationSpec] = PHASE5_POPULATION_SPECS,
    *,
    workers: int = 1,
) -> tuple[Phase5PopulationRecord, ...]:
    forms = tuple(form for spec in specs for form in phase5_candidate_forms(spec))
    if workers <= 1:
        records = tuple(evaluate_phase5_form(form) for form in forms)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = tuple(executor.map(evaluate_phase5_form, forms, chunksize=4))
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.dimension_g,
                record.polarization_type,
                -record.ell_squared_numeric,
                record.candidate_id,
            ),
        )
    )


def _pareto_ids(records: Sequence[Phase5PopulationRecord]) -> set[str]:
    result: set[str] = set()
    for candidate in records:
        dominated = any(
            other.ell_squared_numeric >= candidate.ell_squared_numeric - 1e-13
            and other.logical_image_order >= candidate.logical_image_order
            and (
                other.ell_squared_numeric > candidate.ell_squared_numeric + 1e-13
                or other.logical_image_order > candidate.logical_image_order
            )
            for other in records
        )
        if not dominated:
            result.add(candidate.candidate_id)
    return result


def phase5_population_rows(
    records: Sequence[Phase5PopulationRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_type: dict[tuple[int, ...], list[Phase5PopulationRecord]] = {}
    for record in records:
        by_type.setdefault(record.polarization_type, []).append(record)
    for polarization_type, group in by_type.items():
        pareto = _pareto_ids(group)
        ordered = sorted(group, key=lambda item: (-item.ell_squared_numeric, item.candidate_id))
        ranks = {record.candidate_id: index + 1 for index, record in enumerate(ordered)}
        for record in group:
            row = record.as_dict()
            row["distance_rank_within_type"] = ranks[record.candidate_id]
            row["pareto_optimal"] = record.candidate_id in pareto
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["dimension_g"],
            tuple(row["polarization_type"]),
            row["distance_rank_within_type"],
            row["candidate_id"],
        ),
    )


def _correlation_distance_log_image(records: Sequence[Phase5PopulationRecord]) -> float | None:
    if len(records) < 2:
        return None
    distances = [record.ell_squared_numeric for record in records]
    images = [log(record.logical_image_order) for record in records]
    mean_distance = mean(distances)
    mean_image = mean(images)
    distance_variance = sum((value - mean_distance) ** 2 for value in distances)
    image_variance = sum((value - mean_image) ** 2 for value in images)
    if distance_variance == 0 or image_variance == 0:
        return None
    covariance = sum(
        (distance - mean_distance) * (image - mean_image)
        for distance, image in zip(distances, images)
    )
    return covariance / sqrt(distance_variance * image_variance)


def phase5_population_summary(
    records: Sequence[Phase5PopulationRecord],
) -> list[dict[str, object]]:
    by_type: dict[tuple[int, ...], list[Phase5PopulationRecord]] = {}
    for record in records:
        by_type.setdefault(record.polarization_type, []).append(record)
    summaries: list[dict[str, object]] = []
    for polarization_type, group in sorted(by_type.items()):
        best_distance = max(group, key=lambda record: record.ell_squared_numeric)
        maximum_image = max(record.logical_image_order for record in group)
        extra = sum(record.extra_passive_symmetry for record in group)
        pareto = _pareto_ids(group)
        distance_ordered = sorted(group, key=lambda record: record.ell_squared_numeric, reverse=True)
        upper_quartile = distance_ordered[: max(1, (len(distance_ordered) + 3) // 4)]
        enhanced_group = [record for record in group if record.extra_passive_symmetry]
        minimal_group = [record for record in group if not record.extra_passive_symmetry]
        summaries.append(
            {
                "dimension_g": len(polarization_type),
                "polarization_type": list(polarization_type),
                "candidate_count": len(group),
                "extra_passive_symmetry_count": extra,
                "extra_passive_symmetry_fraction": extra / len(group),
                "mean_ell_squared": mean(record.ell_squared_numeric for record in group),
                "median_ell_squared": median(record.ell_squared_numeric for record in group),
                "mean_ell_squared_enhanced": (
                    mean(record.ell_squared_numeric for record in enhanced_group)
                    if enhanced_group
                    else None
                ),
                "mean_ell_squared_minimal_image": (
                    mean(record.ell_squared_numeric for record in minimal_group)
                    if minimal_group
                    else None
                ),
                "upper_quartile_extra_symmetry_fraction": sum(
                    record.extra_passive_symmetry for record in upper_quartile
                )
                / len(upper_quartile),
                "best_distance_candidate_id": best_distance.candidate_id,
                "best_ell_squared_exact": best_distance.ell_squared_exact,
                "best_distance_logical_image_order": best_distance.logical_image_order,
                "best_distance_has_maximum_image": (
                    best_distance.logical_image_order == maximum_image
                ),
                "maximum_logical_image_order": maximum_image,
                "maximum_image_candidate_count": sum(
                    record.logical_image_order == maximum_image for record in group
                ),
                "logical_image_histogram": {
                    str(order): count
                    for order, count in sorted(
                        Counter(record.logical_image_order for record in group).items()
                    )
                },
                "pareto_candidate_count": len(pareto),
                "distance_log_image_correlation": _correlation_distance_log_image(group),
            }
        )
    return summaries


def write_phase5_population_ledger(
    records: Sequence[Phase5PopulationRecord],
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = phase5_population_rows(records)
    summary = phase5_population_summary(records)
    json_path = output / "phase5_cm_population.json"
    csv_path = output / "phase5_cm_population.csv"
    summary_path = output / "phase5_cm_population_summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    fieldnames = tuple(rows[0]) if rows else ()
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    return json_path, csv_path, summary_path


def load_phase5_population_ledger(
    data_directory: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = Path(data_directory)
    records = load_json_artifact(data / "phase5_cm_population.json")
    summary = load_json_artifact(data / "phase5_cm_population_summary.json")
    return records, summary
