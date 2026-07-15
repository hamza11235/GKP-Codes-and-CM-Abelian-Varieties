"""Scans of symplectic deformations at fixed polarization.

Rational deformations preserve the alternating form exactly, but also remain
in the starting variety's rational isogeny class.  The pi-parameter scans
below deliberately leave that rational orbit and provide generic real
controls.  This module does not certify individual endomorphism rings.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import pi
from random import Random
from typing import Sequence

from .conventions import MetricConvention, coerce_metric_convention
from .metric import Metric
from .kernel import KernelGroup
from .polarization import Polarization
from .systole import RelativeSystoleResult, compute_relative_systole


RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _fraction_matrix(matrix: Sequence[Sequence[int | Fraction]]) -> RationalMatrix:
    return tuple(tuple(Fraction(value) for value in row) for row in matrix)


def _identity(size: int) -> RationalMatrix:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column] for inner in range(len(right))),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def symplectic_transvection(
    alternating: Sequence[Sequence[int]],
    vector: Sequence[int],
    parameter: int | Fraction,
) -> RationalMatrix:
    """Return ``S = I + t v v^T A``, which satisfies ``S^T A S = A``."""

    polarization = Polarization(alternating)
    size = 2 * polarization.dimension
    if len(vector) != size:
        raise ValueError("transvection vector has the wrong dimension")
    a = _fraction_matrix(polarization.matrix)
    v = tuple(Fraction(value) for value in vector)
    t = Fraction(parameter)
    row = tuple(
        sum((v[inner] * a[inner][column] for inner in range(size)), Fraction(0))
        for column in range(size)
    )
    identity = _identity(size)
    result = tuple(
        tuple(identity[i][j] + t * v[i] * row[j] for j in range(size))
        for i in range(size)
    )
    if _multiply(_multiply(_transpose(result), a), result) != a:
        raise ArithmeticError("constructed matrix is not symplectic")
    return result


def deform_metric(
    metric: Sequence[Sequence[int | Fraction]],
    transformation: RationalMatrix,
) -> RationalMatrix:
    """Pull back a metric by ``S``: ``G -> S^T G S``."""

    g = _fraction_matrix(metric)
    result = _multiply(_multiply(_transpose(transformation), g), transformation)
    Metric(result)
    return result


def compose(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    """Compose two lattice-coordinate transformations."""

    return _multiply(left, right)


@dataclass(frozen=True)
class DeformationSample:
    index: int
    transformation: RationalMatrix
    metric: RationalMatrix
    systole_result: RelativeSystoleResult
    parameters: tuple[tuple[tuple[int, ...], Fraction], ...]

    @property
    def squared_systole(self) -> Fraction:
        return Fraction(self.systole_result.squared_systole)


@dataclass(frozen=True)
class DeformationScan:
    polarization: Polarization
    baseline_result: RelativeSystoleResult
    samples: tuple[DeformationSample, ...]
    metric_scale_label: str = "1"

    @property
    def best_sample(self) -> DeformationSample:
        return max(self.samples, key=lambda sample: sample.squared_systole)

    @property
    def number_beating_baseline(self) -> int:
        baseline = Fraction(self.baseline_result.squared_systole)
        return sum(sample.squared_systole > baseline for sample in self.samples)

    @property
    def number_tying_baseline(self) -> int:
        baseline = Fraction(self.baseline_result.squared_systole)
        return sum(sample.squared_systole == baseline for sample in self.samples)


def scan_symplectic_deformations(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[int | Fraction]],
    *,
    sample_count: int,
    seed: int,
    steps: int = 4,
    numerator_bound: int = 5,
    denominator: int = 100,
    vector_bound: int = 2,
    metric_scale_label: str = "1",
    metric_convention: MetricConvention | str = MetricConvention.POLARIZATION_SCALED,
) -> DeformationScan:
    """Generate deterministic rational deformations and compute exact systoles."""

    if sample_count <= 0 or steps <= 0:
        raise ValueError("sample_count and steps must be positive")
    if numerator_bound <= 0 or denominator <= 0 or vector_bound <= 0:
        raise ValueError("scan bounds must be positive")

    polarization = Polarization(alternating)
    base_metric = _fraction_matrix(metric)
    Metric(base_metric)
    convention = coerce_metric_convention(metric_convention)
    baseline = compute_relative_systole(
        polarization,
        base_metric,
        metric_convention=convention,
    )
    size = 2 * polarization.dimension
    random = Random(seed)
    samples: list[DeformationSample] = []

    for index in range(sample_count):
        transformation = _identity(size)
        parameters = []
        for _ in range(steps):
            vector = tuple(random.randint(-vector_bound, vector_bound) for _ in range(size))
            while not any(vector):
                vector = tuple(random.randint(-vector_bound, vector_bound) for _ in range(size))
            numerator = random.randint(1, numerator_bound)
            if random.randrange(2):
                numerator = -numerator
            parameter = Fraction(numerator, denominator)
            elementary = symplectic_transvection(polarization.matrix, vector, parameter)
            transformation = compose(transformation, elementary)
            parameters.append((vector, parameter))

        deformed = deform_metric(base_metric, transformation)
        result = compute_relative_systole(
            polarization,
            deformed,
            metric_convention=convention,
        )
        samples.append(
            DeformationSample(
                index=index,
                transformation=transformation,
                metric=deformed,
                systole_result=result,
                parameters=tuple(parameters),
            )
        )

    return DeformationScan(
        polarization=polarization,
        baseline_result=baseline,
        samples=tuple(samples),
        metric_scale_label=metric_scale_label,
    )


FloatMatrix = tuple[tuple[float, ...], ...]


def _float_matrix(matrix: Sequence[Sequence[int | Fraction | float]]) -> FloatMatrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _float_transvection(
    alternating: Sequence[Sequence[int]],
    vector: Sequence[int],
    parameter: float,
) -> FloatMatrix:
    size = len(alternating)
    a = _float_matrix(alternating)
    row = tuple(
        sum(float(vector[inner]) * a[inner][column] for inner in range(size))
        for column in range(size)
    )
    return tuple(
        tuple(
            float(int(i == j)) + parameter * float(vector[i]) * row[j]
            for j in range(size)
        )
        for i in range(size)
    )


def _float_multiply(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(len(right)))
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


def _float_transpose(matrix: FloatMatrix) -> FloatMatrix:
    return tuple(
        tuple(matrix[i][j] for i in range(len(matrix)))
        for j in range(len(matrix[0]))
    )


def _float_deform_metric(metric: FloatMatrix, transformation: FloatMatrix) -> FloatMatrix:
    raw = _float_multiply(
        _float_multiply(_float_transpose(transformation), metric),
        transformation,
    )
    return tuple(
        tuple((raw[i][j] + raw[j][i]) / 2.0 for j in range(len(raw)))
        for i in range(len(raw))
    )


@dataclass(frozen=True)
class PiDeformationSample:
    """A real deformation whose parameters are rational multiples of pi."""

    index: int
    transformation: FloatMatrix
    metric: FloatMatrix
    systole_result: RelativeSystoleResult
    parameters: tuple[tuple[tuple[int, ...], Fraction], ...]

    @property
    def squared_systole(self) -> float:
        return float(self.systole_result.squared_systole)


@dataclass(frozen=True)
class PiDeformationScan:
    polarization: Polarization
    baseline_result: RelativeSystoleResult
    samples: tuple[PiDeformationSample, ...]
    metric_scale: float = 1.0

    @property
    def best_sample(self) -> PiDeformationSample:
        return max(self.samples, key=lambda sample: sample.squared_systole)

    @property
    def number_beating_baseline(self) -> int:
        baseline = float(self.baseline_result.squared_systole)
        return sum(sample.squared_systole > baseline + 1e-11 for sample in self.samples)


def scan_pi_symplectic_deformations(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[int | Fraction | float]],
    *,
    sample_count: int,
    seed: int,
    amplitude: float,
    steps: int = 4,
    vector_bound: int = 2,
    coefficient_denominator: int = 10_000_000,
    metric_scale: float = 1.0,
    metric_convention: MetricConvention | str = MetricConvention.POLARIZATION_SCALED,
) -> PiDeformationScan:
    """Sample generic real deformations using ``t = rational*pi``.

    The dense rational coefficient grid approximates continuous sampling,
    while the factor pi prevents the transformation from being rational.
    Results are screened with floating-point CVP and can be independently
    reevaluated by :func:`high_precision_pi_systole`.
    """

    if sample_count <= 0 or steps <= 0:
        raise ValueError("sample_count and steps must be positive")
    if amplitude <= 0 or vector_bound <= 0 or coefficient_denominator <= 0:
        raise ValueError("scan bounds must be positive")
    polarization = Polarization(alternating)
    base_metric = _float_matrix(metric)
    convention = coerce_metric_convention(metric_convention)
    baseline = compute_relative_systole(
        polarization,
        base_metric,
        metric_convention=convention,
    )
    size = len(base_metric)
    identity = tuple(
        tuple(float(int(i == j)) for j in range(size))
        for i in range(size)
    )
    max_numerator = max(1, int(amplitude * coefficient_denominator / pi))
    random = Random(seed)
    samples = []

    for index in range(sample_count):
        transformation = identity
        parameters = []
        for _ in range(steps):
            vector = tuple(random.randint(-vector_bound, vector_bound) for _ in range(size))
            while not any(vector):
                vector = tuple(random.randint(-vector_bound, vector_bound) for _ in range(size))
            numerator = random.randint(-max_numerator, max_numerator)
            while numerator == 0:
                numerator = random.randint(-max_numerator, max_numerator)
            coefficient = Fraction(numerator, coefficient_denominator)
            elementary = _float_transvection(
                polarization.matrix,
                vector,
                float(coefficient) * pi,
            )
            transformation = _float_multiply(transformation, elementary)
            parameters.append((vector, coefficient))
        deformed = _float_deform_metric(base_metric, transformation)
        result = compute_relative_systole(
            polarization,
            deformed,
            metric_convention=convention,
        )
        samples.append(
            PiDeformationSample(
                index=index,
                transformation=transformation,
                metric=deformed,
                systole_result=result,
                parameters=tuple(parameters),
            )
        )
    return PiDeformationScan(
        polarization=polarization,
        baseline_result=baseline,
        samples=tuple(samples),
        metric_scale=metric_scale,
    )


def high_precision_pi_systole(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[int | Fraction | float]],
    parameters: Sequence[tuple[Sequence[int], Fraction]],
    *,
    decimal_places: int = 80,
) -> str:
    """Recompute one pi-parameter sample with high-precision CVP.

    A separate mpmath Cholesky branch-and-bound search is used rather than the
    double-precision screening solver.  The returned decimal is the unscaled
    squared systole; callers apply any common symbolic metric scale.
    """

    import mpmath as mp

    mp.mp.dps = decimal_places
    polarization = Polarization(alternating)
    size = len(metric)
    a = mp.matrix([[mp.mpf(value) for value in row] for row in polarization.matrix])
    g = mp.matrix([[mp.mpf(str(value)) for value in row] for row in metric])
    s = mp.eye(size)
    for vector, coefficient in parameters:
        v = mp.matrix([[mp.mpf(value)] for value in vector])
        t = mp.mpf(coefficient.numerator) / coefficient.denominator * mp.pi
        elementary = mp.eye(size) + t * v * (v.T * a)
        s = s * elementary
    deformed = s.T * g * s
    upper = mp.cholesky(deformed).T
    kernel = KernelGroup.from_polarization(polarization)
    global_best = None
    for element in kernel.nonzero_elements:
        coordinates = [mp.mpf(value.numerator) / value.denominator for value in element.coordinates]
        current = [0 for _ in range(size)]
        initial = [int(mp.floor(-value + mp.mpf("0.5"))) for value in coordinates]
        x0 = mp.matrix([[coordinates[i] + initial[i]] for i in range(size)])
        class_best = (x0.T * deformed * x0)[0]

        def recurse(index, partial):
            nonlocal class_best
            if index < 0:
                if partial < class_best:
                    class_best = partial
                return
            tail = mp.fsum(
                upper[index, column] * (coordinates[column] + current[column])
                for column in range(index + 1, size)
            )
            diagonal = upper[index, index]
            remaining = max(mp.mpf("0"), class_best - partial)
            center = -coordinates[index] - tail / diagonal
            radius = mp.sqrt(remaining) / abs(diagonal)
            lower_integer = int(mp.ceil(center - radius - mp.mpf("1e-60")))
            upper_integer = int(mp.floor(center + radius + mp.mpf("1e-60")))
            candidates = sorted(
                range(lower_integer, upper_integer + 1),
                key=lambda value: abs(mp.mpf(value) - center),
            )
            for integer in candidates:
                current[index] = integer
                row_value = diagonal * (coordinates[index] + integer) + tail
                new_partial = partial + row_value * row_value
                if new_partial <= class_best + mp.mpf("1e-60"):
                    recurse(index - 1, new_partial)

        recurse(size - 1, mp.mpf("0"))
        if global_best is None or class_best < global_best:
            global_best = class_best
    return mp.nstr(global_best, decimal_places)
