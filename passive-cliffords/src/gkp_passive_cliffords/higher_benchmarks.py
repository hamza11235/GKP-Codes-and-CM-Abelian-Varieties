"""Exact D4/Bolza and Klein-quartic passive-Clifford benchmarks."""

from __future__ import annotations

from functools import lru_cache

from gkp_systole.models import D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL, PeriodModel
from gkp_systole.polarization import Polarization

from .automorphisms import (
    PolarizedAutomorphismGroup,
    PolarizedAutomorphismProblem,
    enumerate_polarized_automorphisms,
)
from .exact import IntegerMatrix
from .kernel_action import LogicalActionResult, compute_logical_action


def _scale_alternating(
    alternating: tuple[tuple[int, ...], ...], level: int
) -> tuple[tuple[int, ...], ...]:
    if level <= 1:
        raise ValueError("logical level must be greater than one")
    return tuple(tuple(level * value for value in row) for row in alternating)


@lru_cache(maxsize=None)
def period_model_automorphism_elements(model: PeriodModel) -> tuple[IntegerMatrix, ...]:
    """Enumerate the polarized automorphisms once, before reducing at a level."""

    model.validate()
    problem = PolarizedAutomorphismProblem(
        polarization=Polarization(model.principal_alternating),
        metric=model.metric_core,
        complex_structure=model.complex_structure_core,
    )
    return enumerate_polarized_automorphisms(problem).elements


@lru_cache(maxsize=None)
def period_model_logical_action(
    model: PeriodModel, level: int
) -> LogicalActionResult:
    """Compute the passive Clifford image for ``(X,L_0^level)`` exactly."""

    problem = PolarizedAutomorphismProblem(
        polarization=Polarization(_scale_alternating(model.principal_alternating, level)),
        metric=model.metric_core,
        complex_structure=model.complex_structure_core,
    )
    group = PolarizedAutomorphismGroup(
        problem=problem,
        elements=period_model_automorphism_elements(model),
    )
    return compute_logical_action(group)


def phase2_benchmark_table(levels: tuple[int, ...] = (2, 3)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in (D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL):
        for level in levels:
            action = period_model_logical_action(model, level)
            rows.append(
                {
                    "model": model.name,
                    "cm_field": model.cm_field,
                    "level": level,
                    **action.as_dict(),
                }
            )
    return rows
