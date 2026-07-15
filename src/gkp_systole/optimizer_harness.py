"""Common CM-blind objective harness for compatible-metric optimization.

The harness deliberately knows only the polarized metric family and the
relative-systole objective.  Arithmetic reconstruction and CM classification
belong to a later analysis stage and are not consulted while optimizing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
import json
from math import isfinite, sqrt
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .conventions import MetricConvention
from .moduli_search import CompatibleMetricFamily, high_precision_coordinate_systole
from .systole import compute_relative_systole
from .type15 import TYPE_15_EXACT_MODEL


class EvaluationMode(str, Enum):
    """Numerical level used for an objective evaluation."""

    SCREEN = "screen"
    VERIFY = "verify"


class CoordinateOutOfBounds(ValueError):
    """Raised when an optimizer requests a point outside the declared box."""


class EvaluationBudgetExceeded(RuntimeError):
    """Raised when the declared screening budget has been exhausted."""


@dataclass(frozen=True)
class OptimizationRunConfig:
    """Frozen metadata defining one reproducible optimization experiment."""

    experiment_id: str
    optimizer_name: str
    seed: int
    bounds: tuple[tuple[float, float], ...]
    evaluation_budget: int
    verification_decimal_places: int = 80
    compatibility_tolerance: float = 2e-7
    checkpoint_interval: int = 100

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be nonempty")
        if not self.optimizer_name.strip():
            raise ValueError("optimizer_name must be nonempty")
        if not self.bounds:
            raise ValueError("at least one coordinate bound is required")
        for lower, upper in self.bounds:
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("each coordinate bound must be finite with lower < upper")
        if self.evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        if self.verification_decimal_places < 30:
            raise ValueError("verification requires at least 30 decimal places")
        if self.compatibility_tolerance <= 0:
            raise ValueError("compatibility_tolerance must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

    @classmethod
    def symmetric_box(
        cls,
        *,
        experiment_id: str,
        optimizer_name: str,
        seed: int,
        coordinate_dimension: int,
        radius: float,
        evaluation_budget: int,
        verification_decimal_places: int = 80,
        compatibility_tolerance: float = 2e-7,
        checkpoint_interval: int = 100,
    ) -> "OptimizationRunConfig":
        if coordinate_dimension <= 0 or radius <= 0 or not isfinite(radius):
            raise ValueError("coordinate dimension and radius must be positive")
        return cls(
            experiment_id=experiment_id,
            optimizer_name=optimizer_name,
            seed=seed,
            bounds=tuple((-float(radius), float(radius)) for _ in range(coordinate_dimension)),
            evaluation_budget=evaluation_budget,
            verification_decimal_places=verification_decimal_places,
            compatibility_tolerance=compatibility_tolerance,
            checkpoint_interval=checkpoint_interval,
        )


@dataclass(frozen=True)
class CompatibilityDiagnostics:
    """Numerical residuals for a polarization-compatible metric."""

    minimum_eigenvalue: float
    symmetry_residual: float
    determinant_residual: float
    complex_structure_residual: float
    valid: bool


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """One unique objective value with active logical-lift metadata."""

    sequence: int
    mode: EvaluationMode
    coordinates: tuple[float, ...]
    squared_systole: float
    loss: float
    class_multiplicity: int
    lift_multiplicity: int
    active_classes: tuple[tuple[str, ...], ...]
    active_lifts: tuple[tuple[str, ...], ...]
    diagnostics: CompatibilityDiagnostics
    high_precision_squared_systole: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class ObjectiveTraceEntry:
    """One optimizer request, including cache hits."""

    call_index: int
    sequence: int
    mode: EvaluationMode
    coordinates: tuple[float, ...]
    squared_systole: float
    loss: float
    cache_hit: bool


@dataclass(frozen=True)
class HarnessSummary:
    request_count: int
    unique_evaluation_count: int
    screening_evaluation_count: int
    cache_hits: int
    best_squared_systole: float | None
    best_coordinates: tuple[float, ...] | None


def _fraction_label(value) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _coordinate_key(coordinates: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in coordinates)
    if not all(isfinite(value) for value in result):
        raise ValueError("coordinates must be finite")
    return result


def _diagnose_metric(
    family: CompatibleMetricFamily,
    metric: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> CompatibilityDiagnostics:
    alternating = np.asarray(family.alternating, dtype=float)
    gram = np.asarray(metric, dtype=float)
    symmetry_residual = float(np.max(np.abs(gram - gram.T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh((gram + gram.T) / 2.0)))
    expected_determinant = abs(float(np.linalg.det(alternating)))
    determinant_residual = abs(float(np.linalg.det(gram)) - expected_determinant)
    complex_structure = -np.linalg.solve(alternating, gram)
    complex_structure_residual = float(
        np.max(np.abs(complex_structure @ complex_structure + np.eye(gram.shape[0])))
    )
    scale = max(1.0, expected_determinant)
    valid = (
        minimum_eigenvalue > 0.0
        and symmetry_residual <= tolerance
        and determinant_residual <= tolerance * scale
        and complex_structure_residual <= tolerance
    )
    return CompatibilityDiagnostics(
        minimum_eigenvalue=minimum_eigenvalue,
        symmetry_residual=symmetry_residual,
        determinant_residual=determinant_residual,
        complex_structure_residual=complex_structure_residual,
        valid=valid,
    )


def type15_compatible_metric_family() -> CompatibleMetricFamily:
    """Six-dimensional compatible-metric chart centered at the exact record.

    The returned family carries the exact type-``(1,5)`` record as its chart
    origin, but the objective harness does not inspect its CM certificate.
    """

    scale = sqrt(TYPE_15_EXACT_MODEL.scale_radicand)
    metric = tuple(
        tuple(float(value) / scale for value in row)
        for row in TYPE_15_EXACT_MODEL.metric_core
    )
    return CompatibleMetricFamily.from_reference(
        name="type (1,5) exact-record compatible chart",
        alternating=TYPE_15_EXACT_MODEL.alternating,
        reference_metric=metric,
        reference_exact_ell_squared=TYPE_15_EXACT_MODEL.exact_squared_systole,
        reference_ell_squared=TYPE_15_EXACT_MODEL.squared_systole,
        reference_cm="withheld from the optimization objective",
    )


class OptimizerHarness:
    """Thread-safe objective, trace, cache, budget, and checkpoint manager."""

    CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(
        self,
        family: CompatibleMetricFamily,
        config: OptimizationRunConfig,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if len(config.bounds) != family.coordinate_dimension:
            raise ValueError("run bounds do not match the family coordinate dimension")
        self.family = family
        self.config = config
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._cache: dict[tuple[EvaluationMode, tuple[float, ...]], ObjectiveEvaluation] = {}
        self._trace: list[ObjectiveTraceEntry] = []
        self._screening_evaluations = 0
        self._cache_hits = 0
        self._next_sequence = 0
        self._lock = RLock()

    @property
    def trace(self) -> tuple[ObjectiveTraceEntry, ...]:
        with self._lock:
            return tuple(self._trace)

    @property
    def remaining_budget(self) -> int:
        with self._lock:
            return self.config.evaluation_budget - self._screening_evaluations

    def _inside_bounds(self, coordinates: Sequence[float]) -> bool:
        return all(
            lower <= value <= upper
            for value, (lower, upper) in zip(coordinates, self.config.bounds)
        )

    def _record_call(self, evaluation: ObjectiveEvaluation, *, cache_hit: bool) -> None:
        with self._lock:
            if cache_hit:
                self._cache_hits += 1
            self._trace.append(
                ObjectiveTraceEntry(
                    call_index=len(self._trace),
                    sequence=evaluation.sequence,
                    mode=evaluation.mode,
                    coordinates=evaluation.coordinates,
                    squared_systole=evaluation.squared_systole,
                    loss=evaluation.loss,
                    cache_hit=cache_hit,
                )
            )

    def _reserve_sequence(self) -> int:
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            return sequence

    def _compute_screen(
        self,
        coordinates: tuple[float, ...],
        *,
        sequence: int,
    ) -> ObjectiveEvaluation:
        started = perf_counter()
        metric = self.family.metric(coordinates)
        diagnostics = _diagnose_metric(
            self.family,
            metric,
            tolerance=self.config.compatibility_tolerance,
        )
        if not diagnostics.valid:
            raise ArithmeticError(f"invalid compatible metric: {diagnostics}")
        result = compute_relative_systole(
            self.family.alternating,
            metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        active_classes = tuple(
            tuple(_fraction_label(value) for value in element.coordinates)
            for element in result.shortest_classes
        )
        active_lifts = tuple(
            tuple(_fraction_label(value) for value in lift)
            for class_result in result.shortest_class_results
            for lift in class_result.lifts
        )
        value = float(result.squared_systole)
        return ObjectiveEvaluation(
            sequence=sequence,
            mode=EvaluationMode.SCREEN,
            coordinates=coordinates,
            squared_systole=value,
            loss=-value,
            class_multiplicity=result.class_multiplicity,
            lift_multiplicity=result.lift_multiplicity,
            active_classes=active_classes,
            active_lifts=active_lifts,
            diagnostics=diagnostics,
            high_precision_squared_systole=None,
            elapsed_seconds=perf_counter() - started,
        )

    def evaluate(
        self,
        coordinates: Sequence[float],
        *,
        mode: EvaluationMode | str = EvaluationMode.SCREEN,
    ) -> ObjectiveEvaluation:
        """Evaluate one point and record the optimizer request."""

        coordinate_tuple = _coordinate_key(coordinates)
        if len(coordinate_tuple) != self.family.coordinate_dimension:
            raise ValueError("coordinate vector has the wrong dimension")
        if not self._inside_bounds(coordinate_tuple):
            raise CoordinateOutOfBounds("coordinates lie outside the declared experiment box")
        evaluation_mode = EvaluationMode(mode)
        key = (evaluation_mode, coordinate_tuple)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self._record_call(cached, cache_hit=True)
            return cached

        if evaluation_mode is EvaluationMode.SCREEN:
            with self._lock:
                if self._screening_evaluations >= self.config.evaluation_budget:
                    raise EvaluationBudgetExceeded("screening evaluation budget exhausted")
                self._screening_evaluations += 1
            evaluation = self._compute_screen(
                coordinate_tuple,
                sequence=self._reserve_sequence(),
            )
        else:
            screen_key = (EvaluationMode.SCREEN, coordinate_tuple)
            with self._lock:
                screen = self._cache.get(screen_key)
            if screen is None:
                screen = self.evaluate(coordinate_tuple, mode=EvaluationMode.SCREEN)
            started = perf_counter()
            high_precision = high_precision_coordinate_systole(
                self.family,
                coordinate_tuple,
                decimal_places=self.config.verification_decimal_places,
            )
            evaluation = ObjectiveEvaluation(
                sequence=self._reserve_sequence(),
                mode=EvaluationMode.VERIFY,
                coordinates=coordinate_tuple,
                squared_systole=screen.squared_systole,
                loss=screen.loss,
                class_multiplicity=screen.class_multiplicity,
                lift_multiplicity=screen.lift_multiplicity,
                active_classes=screen.active_classes,
                active_lifts=screen.active_lifts,
                diagnostics=screen.diagnostics,
                high_precision_squared_systole=high_precision,
                elapsed_seconds=perf_counter() - started,
            )

        with self._lock:
            existing = self._cache.setdefault(key, evaluation)
            if existing is not evaluation:
                evaluation = existing
                cache_hit = True
            else:
                cache_hit = False
            should_checkpoint = (
                self.checkpoint_path is not None
                and len(self._cache) % self.config.checkpoint_interval == 0
            )
        self._record_call(evaluation, cache_hit=cache_hit)
        if should_checkpoint:
            self.save_checkpoint()
        return evaluation

    def objective(self, coordinates: Sequence[float]) -> float:
        """Return a minimizer-compatible loss, penalizing out-of-box points."""

        try:
            return self.evaluate(coordinates, mode=EvaluationMode.SCREEN).loss
        except CoordinateOutOfBounds:
            return float("inf")

    def evaluate_many(
        self,
        coordinate_rows: Iterable[Sequence[float]],
        *,
        mode: EvaluationMode | str = EvaluationMode.SCREEN,
        workers: int = 1,
    ) -> tuple[ObjectiveEvaluation, ...]:
        """Evaluate a batch in deterministic input order."""

        rows = tuple(_coordinate_key(row) for row in coordinate_rows)
        if workers <= 0:
            raise ValueError("workers must be positive")
        if workers == 1:
            return tuple(self.evaluate(row, mode=mode) for row in rows)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = tuple(executor.submit(self.evaluate, row, mode=mode) for row in rows)
            return tuple(future.result() for future in futures)

    def summary(self) -> HarnessSummary:
        with self._lock:
            screen_values = tuple(
                evaluation
                for (mode, _), evaluation in self._cache.items()
                if mode is EvaluationMode.SCREEN
            )
            best = max(screen_values, key=lambda item: item.squared_systole) if screen_values else None
            return HarnessSummary(
                request_count=len(self._trace),
                unique_evaluation_count=len(self._cache),
                screening_evaluation_count=self._screening_evaluations,
                cache_hits=self._cache_hits,
                best_squared_systole=best.squared_systole if best else None,
                best_coordinates=best.coordinates if best else None,
            )

    def _metadata_record(self) -> dict[str, object]:
        return {
            "record_type": "metadata",
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "family_name": self.family.name,
            "alternating": self.family.alternating,
            "coordinate_dimension": self.family.coordinate_dimension,
            "run_config": asdict(self.config),
        }

    @staticmethod
    def _evaluation_record(evaluation: ObjectiveEvaluation) -> dict[str, object]:
        return {
            "record_type": "evaluation",
            "sequence": evaluation.sequence,
            "mode": evaluation.mode.value,
            "coordinates": evaluation.coordinates,
            "squared_systole": evaluation.squared_systole,
            "loss": evaluation.loss,
            "class_multiplicity": evaluation.class_multiplicity,
            "lift_multiplicity": evaluation.lift_multiplicity,
            "active_classes": evaluation.active_classes,
            "active_lifts": evaluation.active_lifts,
            "diagnostics": asdict(evaluation.diagnostics),
            "high_precision_squared_systole": evaluation.high_precision_squared_systole,
            "elapsed_seconds": evaluation.elapsed_seconds,
        }

    @staticmethod
    def _trace_record(entry: ObjectiveTraceEntry) -> dict[str, object]:
        return {
            "record_type": "trace",
            "call_index": entry.call_index,
            "sequence": entry.sequence,
            "mode": entry.mode.value,
            "coordinates": entry.coordinates,
            "squared_systole": entry.squared_systole,
            "loss": entry.loss,
            "cache_hit": entry.cache_hit,
        }

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically write a resumable JSON-lines checkpoint."""

        target = Path(path) if path is not None else self.checkpoint_path
        if target is None:
            raise ValueError("no checkpoint path was supplied")
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            evaluations = sorted(self._cache.values(), key=lambda item: (item.sequence, item.mode.value))
            records = [self._metadata_record()] + [
                self._evaluation_record(evaluation) for evaluation in evaluations
            ] + [self._trace_record(entry) for entry in self._trace]
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            temporary.replace(target)
        return target

    @classmethod
    def from_checkpoint(
        cls,
        family: CompatibleMetricFamily,
        path: str | Path,
    ) -> "OptimizerHarness":
        """Resume cached evaluations and experiment metadata from JSON lines."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records or records[0].get("record_type") != "metadata":
            raise ValueError("checkpoint is missing its metadata record")
        metadata = records[0]
        if metadata.get("schema_version") != cls.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if metadata.get("family_name") != family.name:
            raise ValueError("checkpoint family does not match the supplied chart")
        raw_config = metadata["run_config"]
        config = OptimizationRunConfig(
            **{
                **raw_config,
                "bounds": tuple(tuple(float(value) for value in pair) for pair in raw_config["bounds"]),
            }
        )
        harness = cls(family, config, checkpoint_path=source)
        for record in records[1:]:
            if record.get("record_type") == "evaluation":
                diagnostics = CompatibilityDiagnostics(**record["diagnostics"])
                evaluation = ObjectiveEvaluation(
                    sequence=int(record["sequence"]),
                    mode=EvaluationMode(record["mode"]),
                    coordinates=tuple(float(value) for value in record["coordinates"]),
                    squared_systole=float(record["squared_systole"]),
                    loss=float(record["loss"]),
                    class_multiplicity=int(record["class_multiplicity"]),
                    lift_multiplicity=int(record["lift_multiplicity"]),
                    active_classes=tuple(tuple(row) for row in record["active_classes"]),
                    active_lifts=tuple(tuple(row) for row in record["active_lifts"]),
                    diagnostics=diagnostics,
                    high_precision_squared_systole=record["high_precision_squared_systole"],
                    elapsed_seconds=float(record["elapsed_seconds"]),
                )
                harness._cache[(evaluation.mode, evaluation.coordinates)] = evaluation
            elif record.get("record_type") == "trace":
                harness._trace.append(
                    ObjectiveTraceEntry(
                        call_index=int(record["call_index"]),
                        sequence=int(record["sequence"]),
                        mode=EvaluationMode(record["mode"]),
                        coordinates=tuple(float(value) for value in record["coordinates"]),
                        squared_systole=float(record["squared_systole"]),
                        loss=float(record["loss"]),
                        cache_hit=bool(record["cache_hit"]),
                    )
                )
            else:
                raise ValueError("unknown checkpoint record type")
        harness._screening_evaluations = sum(
            mode is EvaluationMode.SCREEN for mode, _ in harness._cache
        )
        harness._cache_hits = sum(entry.cache_hit for entry in harness._trace)
        harness._next_sequence = 1 + max(
            (evaluation.sequence for evaluation in harness._cache.values()),
            default=-1,
        )
        return harness
