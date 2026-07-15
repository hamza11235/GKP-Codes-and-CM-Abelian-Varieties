"""General relative-systole calculation from polarization and metric data."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Sequence

from .conventions import (
    MetricConvention,
    NormalizationMetadata,
    coerce_metric_convention,
)
from .cvp import ClosestVectorResult, closest_integer_translate
from .kernel import KernelElement, KernelGroup
from .metric import Metric, Scalar
from .polarization import Polarization


def _equal(left: Scalar, right: Scalar, tolerance: float) -> bool:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left == right
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= tolerance * scale


@dataclass(frozen=True)
class RelativeSystoleResult:
    polarization: Polarization
    metric: Metric
    kernel: KernelGroup
    squared_systole: Scalar
    shortest_classes: tuple[KernelElement, ...]
    shortest_class_results: tuple[ClosestVectorResult, ...]
    class_results: tuple[ClosestVectorResult, ...]
    metric_convention: MetricConvention

    @property
    def systole(self) -> float:
        return sqrt(float(self.squared_systole))

    @property
    def class_multiplicity(self) -> int:
        return len(self.shortest_classes)

    @property
    def lift_multiplicity(self) -> int:
        return sum(len(result.lifts) for result in self.shortest_class_results)

    @property
    def certified(self) -> bool:
        return all(result.certified for result in self.class_results)

    @property
    def normalization(self) -> NormalizationMetadata:
        return NormalizationMetadata(
            metric_convention=self.metric_convention,
            dimension_g=self.polarization.dimension,
            polarization_type=self.polarization.type,
            metric_determinant=self.metric.determinant,
        )

    def normalization_record(self) -> dict[str, object]:
        """Return serializable comparison metadata, including ``ell^2``."""

        return {
            **self.normalization.as_dict(),
            "ell_squared": self.squared_systole,
        }


def compute_relative_systole(
    alternating: Polarization | Sequence[Sequence[int]],
    metric: Metric | Sequence[Sequence[float | int | Fraction]],
    *,
    metric_convention: MetricConvention | str,
    tolerance: float = 1e-12,
) -> RelativeSystoleResult:
    """Compute the shortest nonzero point of ``Lambda-perp/Lambda``."""

    polarization = alternating if isinstance(alternating, Polarization) else Polarization(alternating)
    metric_object = metric if isinstance(metric, Metric) else Metric(metric)
    convention = coerce_metric_convention(metric_convention)
    if metric_object.dimension != 2 * polarization.dimension:
        raise ValueError("polarization and metric dimensions do not match")

    kernel = KernelGroup.from_polarization(polarization)
    class_results = tuple(
        closest_integer_translate(element, metric_object, tolerance=tolerance)
        for element in kernel.nonzero_elements
    )
    minimum = min(result.squared_distance for result in class_results)
    shortest_results = tuple(
        result
        for result in class_results
        if _equal(result.squared_distance, minimum, tolerance)
    )
    return RelativeSystoleResult(
        polarization=polarization,
        metric=metric_object,
        kernel=kernel,
        squared_systole=minimum,
        shortest_classes=tuple(result.element for result in shortest_results),
        shortest_class_results=shortest_results,
        class_results=class_results,
        metric_convention=convention,
    )
