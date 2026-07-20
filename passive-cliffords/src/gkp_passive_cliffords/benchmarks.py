"""One-mode exact benchmarks for the Phase 1 action engine."""

from __future__ import annotations

from dataclasses import dataclass

from gkp_systole.polarization import Polarization

from .automorphisms import PolarizedAutomorphismProblem, enumerate_polarized_automorphisms
from .kernel_action import LogicalActionResult, compute_logical_action


@dataclass(frozen=True)
class EllipticBenchmark:
    name: str
    metric: tuple[tuple[int, int], tuple[int, int]]


SQUARE = EllipticBenchmark("square CM", ((1, 0), (0, 1)))
HEXAGONAL = EllipticBenchmark("hexagonal CM", ((2, 1), (1, 2)))
GENERIC_RECTANGULAR = EllipticBenchmark("generic rectangular", ((1, 0), (0, 2)))


def elliptic_logical_action(
    benchmark: EllipticBenchmark, level: int
) -> LogicalActionResult:
    polarization = Polarization(((0, level), (-level, 0)))
    problem = PolarizedAutomorphismProblem(
        polarization=polarization,
        metric=benchmark.metric,
    )
    return compute_logical_action(enumerate_polarized_automorphisms(problem))


def phase1_benchmark_table(levels: tuple[int, ...] = (2, 3, 5)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in (GENERIC_RECTANGULAR, SQUARE, HEXAGONAL):
        for level in levels:
            action = elliptic_logical_action(benchmark, level)
            rows.append({"benchmark": benchmark.name, "level": level, **action.as_dict()})
    return rows
