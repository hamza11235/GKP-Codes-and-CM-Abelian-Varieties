"""A first simple quartic-CM family from the fifth cyclotomic field.

Let ``K=Q(zeta_5)``, choose the primitive CM type ``{zeta -> zeta,
zeta -> zeta^2}``, and use the lattice ``O_K`` with power basis
``1,zeta,zeta^2,zeta^3``.  For

``alpha = m + n*(zeta + zeta^-1)``

totally positive in the real subfield, the element

``xi = alpha*(zeta-zeta^-1)/5``

defines the integral alternating pairing ``Tr_K/Q(xi*conj(x)*y)``.  This is a
concrete instance of the standard ``(K,Phi,a,xi)`` CM construction.  The
polarization matrix is exact; the compatible metric is evaluated numerically
from the two complex embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

import numpy as np

from .conventions import MetricConvention
from .metric import Metric
from .polarization import Polarization
from .systole import RelativeSystoleResult, compute_relative_systole


def _cyclotomic_trace(power: int) -> int:
    return 4 if power % 5 == 0 else -1


@dataclass(frozen=True, order=True)
class CyclotomicFivePolarization:
    """Polarization attached to ``alpha=m+n*(zeta+zeta^-1)``."""

    m: int
    n: int = 0

    def __post_init__(self) -> None:
        if not self.is_totally_positive:
            raise ValueError("alpha must be totally positive in Q(sqrt(5))")

    @property
    def field(self) -> str:
        return "Q(zeta_5)"

    @property
    def cm_type(self) -> tuple[int, int]:
        return (1, 2)

    @property
    def ideal_basis(self) -> tuple[str, ...]:
        return ("1", "zeta", "zeta^2", "zeta^3")

    @property
    def xi(self) -> str:
        return f"({self.m}+{self.n}(zeta+zeta^-1))*(zeta-zeta^-1)/5"

    @property
    def real_trace(self) -> int:
        return 2 * self.m - self.n

    @property
    def real_norm(self) -> int:
        return self.m * self.m - self.m * self.n - self.n * self.n

    @property
    def is_totally_positive(self) -> bool:
        return self.real_trace > 0 and self.real_norm > 0

    @property
    def simple_cm(self) -> bool:
        return True

    def _pairing_entry(self, power: int) -> int:
        numerator = self.m * (
            _cyclotomic_trace(power + 1) - _cyclotomic_trace(power - 1)
        ) + self.n * (
            _cyclotomic_trace(power + 2) - _cyclotomic_trace(power - 2)
        )
        if numerator % 5:
            raise ArithmeticError("cyclotomic trace pairing is not integral")
        return numerator // 5

    @property
    def alternating(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self._pairing_entry(column - row) for column in range(4))
            for row in range(4)
        )

    @property
    def polarization_type(self) -> tuple[int, int]:
        return Polarization(self.alternating).type

    @property
    def embedding_matrix(self) -> tuple[tuple[float, ...], ...]:
        embeddings = (1, 2)
        rows = [
            [cos(2.0 * pi * embedding * power / 5.0) for power in range(4)]
            for embedding in embeddings
        ]
        rows += [
            [sin(2.0 * pi * embedding * power / 5.0) for power in range(4)]
            for embedding in embeddings
        ]
        return tuple(tuple(value for value in row) for row in rows)

    @property
    def complex_structure(self) -> tuple[tuple[float, ...], ...]:
        basis = np.asarray(self.embedding_matrix, dtype=float)
        standard = np.block(
            [[np.zeros((2, 2)), -np.eye(2)], [np.eye(2), np.zeros((2, 2))]]
        )
        structure = np.linalg.solve(basis, standard @ basis)
        return tuple(tuple(float(value) for value in row) for row in structure)

    @property
    def metric(self) -> tuple[tuple[float, ...], ...]:
        alternating = np.asarray(self.alternating, dtype=float)
        structure = np.asarray(self.complex_structure, dtype=float)
        metric = -alternating @ structure
        metric = (metric + metric.T) / 2.0
        return tuple(tuple(float(value) for value in row) for row in metric)

    def validation_residuals(self) -> dict[str, float]:
        alternating = np.asarray(self.alternating, dtype=float)
        structure = np.asarray(self.complex_structure, dtype=float)
        metric = np.asarray(self.metric, dtype=float)
        return {
            "J_squared": float(np.max(np.abs(structure @ structure + np.eye(4)))),
            "metric_symmetry": float(np.max(np.abs(metric - metric.T))),
            "A_equals_GJ": float(np.max(np.abs(metric @ structure - alternating))),
            "determinant": abs(float(np.linalg.det(metric)) - abs(float(np.linalg.det(alternating)))),
        }

    def validate(self) -> None:
        polarization = Polarization(self.alternating)
        Metric(self.metric)
        if polarization.type != self.polarization_type:
            raise ArithmeticError("cyclotomic polarization type is inconsistent")
        residuals = self.validation_residuals()
        if max(residuals.values()) > 2e-10:
            raise ArithmeticError(f"cyclotomic CM compatibility failed: {residuals}")
        if self.real_norm != polarization.type[0] * polarization.type[1]:
            raise ArithmeticError("real norm and polarization degree disagree")

    def compute_relative_systole(self) -> RelativeSystoleResult:
        self.validate()
        return compute_relative_systole(
            self.alternating,
            self.metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )


@dataclass(frozen=True)
class CyclotomicFiveResult:
    polarization: CyclotomicFivePolarization
    systole_result: RelativeSystoleResult

    @property
    def squared_systole(self) -> float:
        return float(self.systole_result.squared_systole)


def survey_cyclotomic_five_polarizations(
    maximum_absolute_coefficient: int,
    *,
    polarization_types: Iterable[tuple[int, int]] | None = None,
) -> tuple[CyclotomicFiveResult, ...]:
    """Rank a bounded set of totally-positive polarizations on ``O_Q(zeta5)``."""

    if maximum_absolute_coefficient <= 0:
        raise ValueError("coefficient bound must be positive")
    requested = None if polarization_types is None else set(polarization_types)
    results = []
    for m in range(-maximum_absolute_coefficient, maximum_absolute_coefficient + 1):
        for n in range(-maximum_absolute_coefficient, maximum_absolute_coefficient + 1):
            trace = 2 * m - n
            norm = m * m - m * n - n * n
            if trace <= 0 or norm <= 1:
                continue
            polarization = CyclotomicFivePolarization(m, n)
            if requested is not None and polarization.polarization_type not in requested:
                continue
            results.append(
                CyclotomicFiveResult(
                    polarization,
                    polarization.compute_relative_systole(),
                )
            )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.polarization.polarization_type,
                -item.squared_systole,
                item.polarization,
            ),
        )
    )


def high_precision_cyclotomic_five_systole(
    polarization: CyclotomicFivePolarization,
    *,
    decimal_places: int = 70,
) -> str:
    """Rebuild the cyclotomic metric and solve its CVPs with ``mpmath``."""

    import mpmath as mp

    from .moduli_search import high_precision_metric_systole

    mp.mp.dps = decimal_places
    embeddings = (1, 2)
    rows = [
        [mp.cos(2 * mp.pi * embedding * power / 5) for power in range(4)]
        for embedding in embeddings
    ]
    rows += [
        [mp.sin(2 * mp.pi * embedding * power / 5) for power in range(4)]
        for embedding in embeddings
    ]
    basis = mp.matrix(rows)
    standard = mp.matrix(
        (
            (0, 0, -1, 0),
            (0, 0, 0, -1),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
        )
    )
    structure = basis**-1 * standard * basis
    alternating = mp.matrix(polarization.alternating)
    metric = -alternating * structure
    metric_rows = tuple(
        tuple(mp.nstr(metric[row, column], decimal_places) for column in range(4))
        for row in range(4)
    )
    return high_precision_metric_systole(
        polarization.alternating,
        metric_rows,
        decimal_places,
    )
