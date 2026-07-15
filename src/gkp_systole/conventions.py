"""Explicit metric-normalization conventions for relative systoles.

The relative-systole solver only sees an alternating form and a metric.  The
distinction recorded here describes how that metric was chosen when the
polarization changes.  Keeping this provenance explicit prevents values from
different normalization conventions from being compared accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Sequence

from .metric import Metric, Scalar


class MetricConvention(str, Enum):
    """How the supplied physical metric is related to the polarization."""

    FIXED_PRINCIPAL = "fixed_principal_metric"
    POLARIZATION_SCALED = "polarization_scaled_metric"


def coerce_metric_convention(
    value: MetricConvention | str,
) -> MetricConvention:
    """Return a validated :class:`MetricConvention`."""

    if isinstance(value, MetricConvention):
        return value
    try:
        return MetricConvention(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in MetricConvention)
        raise ValueError(f"unknown metric convention {value!r}; expected one of {allowed}") from error


def uniform_metric(
    principal_metric: Metric | Sequence[Sequence[int | float | Fraction]],
    level: int,
    convention: MetricConvention | str,
) -> Metric:
    """Return the metric used for a uniform level-``d`` polarization.

    Under ``FIXED_PRINCIPAL`` the original metric is held fixed, as in the
    lattice identity ``ell(L^d) = lambda_1(L)/d``.  Under
    ``POLARIZATION_SCALED`` the metric is multiplied by ``d`` together with
    the Riemann form, so squared lengths acquire an additional factor ``d``.
    """

    if level <= 0:
        raise ValueError("uniform polarization level must be positive")
    metric = principal_metric if isinstance(principal_metric, Metric) else Metric(principal_metric)
    selected = coerce_metric_convention(convention)
    factor = 1 if selected is MetricConvention.FIXED_PRINCIPAL else level
    return Metric(
        tuple(
            tuple(factor * value for value in row)
            for row in metric.matrix
        )
    )


@dataclass(frozen=True)
class NormalizationMetadata:
    """Normalization provenance attached to a relative-systole result."""

    metric_convention: MetricConvention
    dimension_g: int
    polarization_type: tuple[int, ...]
    metric_determinant: Scalar

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_convention": self.metric_convention.value,
            "dimension_g": self.dimension_g,
            "polarization_type": self.polarization_type,
            "metric_determinant": self.metric_determinant,
        }
