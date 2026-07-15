"""Full-dimensional compatible-metric searches at fixed polarization.

For a reference compatible pair ``(A,G0)``, the symmetric space of compatible
metrics is the orbit

``G(H) = exp(H)^T G0 exp(H)``,

where ``H`` lies in the Cartan complement

``H^T A + A H = 0`` and ``H^T G0 = G0 H``.

Its real dimension is ``g(g+1)``: six coordinates for abelian surfaces and
twelve for abelian threefolds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .conventions import MetricConvention
from .eisenstein_cm import EisensteinHermitianForm
from .hermitian_cm import GaussianHermitianForm
from .metric import Metric
from .polarization import Polarization
from .systole import compute_relative_systole


FloatMatrix = tuple[tuple[float, ...], ...]


def _as_array(matrix: Sequence[Sequence[int | float]]) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("matrix must be square")
    return result


def _cartan_tangent_basis(
    alternating: np.ndarray,
    metric: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, ...]:
    """Return a canonical Frobenius-orthonormal Cartan basis.

    In metric-orthonormal coordinates the desired matrices are symmetric and
    anticommute with the compatible complex structure.  Projecting the
    lexicographically ordered standard symmetric matrices onto that subspace,
    followed by ordered Gram--Schmidt, avoids the arbitrary null-space rotation
    produced by an SVD.
    """

    size = metric.shape[0]
    expected = (size // 2) * (size // 2 + 1)

    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    complex_structure = -np.linalg.inv(alternating) @ metric
    orthogonal_complex_structure = square_root @ complex_structure @ inverse_square_root

    orthonormal: list[np.ndarray] = []
    for row in range(size):
        for column in range(row, size):
            candidate = np.zeros((size, size), dtype=float)
            candidate[row, column] = 1.0
            candidate[column, row] = 1.0
            if row != column:
                candidate /= np.sqrt(2.0)
            projected = (
                candidate
                + orthogonal_complex_structure @ candidate @ orthogonal_complex_structure
            ) / 2.0
            for previous in orthonormal:
                projected -= np.sum(previous * projected) * previous
            norm = float(np.linalg.norm(projected))
            if norm > tolerance:
                orthonormal.append(projected / norm)
            if len(orthonormal) == expected:
                break
        if len(orthonormal) == expected:
            break
    if len(orthonormal) != expected:
        raise ArithmeticError(
            f"compatible tangent space has dimension {len(orthonormal)}, expected {expected}"
        )

    basis = []
    for k in orthonormal:
        h = inverse_square_root @ k @ square_root
        h[np.abs(h) < 1e-15] = 0.0
        basis.append(h)
    return tuple(basis)


def _self_adjoint_exponential(
    generator: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    """Exponentiate a metric-self-adjoint matrix via a symmetric eigensolve."""

    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    symmetric = square_root @ generator @ inverse_square_root
    symmetric = (symmetric + symmetric.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    exponential = (vectors * np.exp(values)) @ vectors.T
    return inverse_square_root @ exponential @ square_root


@dataclass(frozen=True)
class CompatibleMetricFamily:
    """Full-dimensional compatible-metric family at fixed polarization."""

    name: str
    alternating: tuple[tuple[int, ...], ...]
    reference_metric: FloatMatrix
    reference_exact_ell_squared: str
    reference_ell_squared: float
    reference_cm: str
    tangent_basis: tuple[np.ndarray, ...]

    @classmethod
    def from_reference(
        cls,
        *,
        name: str,
        alternating: Sequence[Sequence[int]],
        reference_metric: Sequence[Sequence[int | float]],
        reference_exact_ell_squared: str,
        reference_ell_squared: float,
        reference_cm: str,
    ) -> "CompatibleMetricFamily":
        polarization = Polarization(alternating)
        a = _as_array(polarization.matrix)
        g = _as_array(reference_metric)
        Metric(g)
        inverse_a = np.linalg.inv(a)
        complex_structure = -inverse_a @ g
        identity = np.eye(g.shape[0])
        if np.max(np.abs(complex_structure @ complex_structure + identity)) > 1e-9:
            raise ValueError("reference metric is not compatible with its polarization")
        expected_determinant = abs(float(np.linalg.det(a)))
        if abs(float(np.linalg.det(g)) - expected_determinant) > 1e-8:
            raise ValueError("reference metric has the wrong polarization covolume")
        basis = _cartan_tangent_basis(a, g)
        family = cls(
            name=name,
            alternating=polarization.matrix,
            reference_metric=tuple(tuple(float(value) for value in row) for row in g),
            reference_exact_ell_squared=reference_exact_ell_squared,
            reference_ell_squared=float(reference_ell_squared),
            reference_cm=reference_cm,
            tangent_basis=basis,
        )
        family.validate_coordinates((0.0,) * family.coordinate_dimension)
        return family

    @property
    def polarization(self) -> Polarization:
        return Polarization(self.alternating)

    @property
    def coordinate_dimension(self) -> int:
        return len(self.tangent_basis)

    def transformation(self, coordinates: Sequence[float]) -> np.ndarray:
        if len(coordinates) != self.coordinate_dimension:
            raise ValueError("coordinate vector has the wrong dimension")
        generator = sum(
            (float(value) * basis for value, basis in zip(coordinates, self.tangent_basis)),
            start=np.zeros_like(_as_array(self.reference_metric)),
        )
        return _self_adjoint_exponential(generator, _as_array(self.reference_metric))

    def metric(self, coordinates: Sequence[float]) -> FloatMatrix:
        transformation = self.transformation(coordinates)
        reference = _as_array(self.reference_metric)
        metric = transformation.T @ reference @ transformation
        metric = (metric + metric.T) / 2.0
        return tuple(tuple(float(value) for value in row) for row in metric)

    def validate_coordinates(self, coordinates: Sequence[float]) -> None:
        a = _as_array(self.alternating)
        transformation = self.transformation(coordinates)
        symplectic_error = np.max(np.abs(transformation.T @ a @ transformation - a))
        if symplectic_error > 2e-8:
            raise ArithmeticError(f"coordinate map is not symplectic: residual={symplectic_error}")
        metric = _as_array(self.metric(coordinates))
        Metric(metric)
        determinant_error = abs(np.linalg.det(metric) - abs(np.linalg.det(a)))
        if determinant_error > 2e-7:
            raise ArithmeticError(
                f"compatible metric has the wrong determinant: residual={determinant_error}"
            )
        complex_structure = -np.linalg.inv(a) @ metric
        complex_error = np.max(
            np.abs(complex_structure @ complex_structure + np.eye(metric.shape[0]))
        )
        if complex_error > 2e-7:
            raise ArithmeticError(
                f"deformed complex structure is invalid: residual={complex_error}"
            )

    def evaluate(self, coordinates: Sequence[float]) -> "ModuliSample":
        metric = self.metric(coordinates)
        result = compute_relative_systole(
            self.alternating,
            metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        return ModuliSample(
            coordinates=tuple(float(value) for value in coordinates),
            squared_systole=float(result.squared_systole),
            class_multiplicity=result.class_multiplicity,
            lift_multiplicity=result.lift_multiplicity,
            metric=metric,
            certified=result.certified,
        )


@dataclass(frozen=True)
class ModuliSample:
    coordinates: tuple[float, ...]
    squared_systole: float
    class_multiplicity: int
    lift_multiplicity: int
    metric: FloatMatrix
    certified: bool


@dataclass(frozen=True)
class CompatibleModuliSearch:
    family: CompatibleMetricFamily
    samples: tuple[ModuliSample, ...]
    refined_samples: tuple[ModuliSample, ...]
    radius: float
    local_radius: float
    seed: int

    @property
    def best_sample(self) -> ModuliSample:
        return max(self.samples + self.refined_samples, key=lambda item: item.squared_systole)

    @property
    def best_screen_sample(self) -> ModuliSample:
        return max(self.samples, key=lambda item: item.squared_systole)

    @property
    def number_beating_reference(self) -> int:
        threshold = self.family.reference_ell_squared + 1e-10
        return sum(
            sample.squared_systole > threshold
            for sample in self.samples + self.refined_samples
        )

    @property
    def evaluated_count(self) -> int:
        return len(self.samples) + len(self.refined_samples)

    def quantile(self, probability: float) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie in [0,1]")
        values = np.asarray([sample.squared_systole for sample in self.samples])
        return float(np.quantile(values, probability))


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _halton_coordinates(
    index: int,
    dimension: int,
    radius: float,
    shifts: Sequence[float],
) -> tuple[float, ...]:
    if dimension > len(_HALTON_PRIMES):
        raise ValueError("Halton table does not contain enough prime bases")
    return tuple(
        radius
        * (2.0 * ((_radical_inverse(index, _HALTON_PRIMES[axis]) + shifts[axis]) % 1.0) - 1.0)
        for axis in range(dimension)
    )


def _pattern_refine(
    family: CompatibleMetricFamily,
    start: ModuliSample,
    *,
    initial_step: float,
    minimum_step: float,
    maximum_rounds: int,
) -> ModuliSample:
    current = start
    step = initial_step
    rounds = 0
    while step >= minimum_step and rounds < maximum_rounds:
        rounds += 1
        best = current
        for axis in range(family.coordinate_dimension):
            for direction in (-1.0, 1.0):
                coordinates = list(current.coordinates)
                coordinates[axis] += direction * step
                candidate = family.evaluate(coordinates)
                if candidate.squared_systole > best.squared_systole + 1e-13:
                    best = candidate
        if best is current:
            step /= 2.0
        else:
            current = best
    return current


def refine_compatible_moduli_sample(
    family: CompatibleMetricFamily,
    start: ModuliSample | Sequence[float],
    *,
    seed: int,
    direction_count: int = 32,
    initial_step: float = 0.1,
    minimum_step: float = 1e-6,
    maximum_rounds: int = 160,
) -> ModuliSample:
    """Refine with coordinate and deterministic random tangent directions.

    The relative systole is a nonsmooth minimum of CVP objectives.  Pure
    coordinate pattern search can therefore stop even when a combined tangent
    direction improves the value.  This routine augments the axes by a fixed
    set of unit directions and is intended for final Phase-7 candidates.
    """

    if direction_count < 0:
        raise ValueError("direction_count must be nonnegative")
    if initial_step <= 0 or minimum_step <= 0 or maximum_rounds <= 0:
        raise ValueError("refinement scales and rounds must be positive")
    current = start if isinstance(start, ModuliSample) else family.evaluate(start)
    dimension = family.coordinate_dimension
    random = np.random.default_rng(seed)
    directions = [np.eye(dimension)[axis] for axis in range(dimension)]
    for _ in range(direction_count):
        direction = random.normal(size=dimension)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            directions.append(direction / norm)

    step = initial_step
    rounds = 0
    while step >= minimum_step and rounds < maximum_rounds:
        rounds += 1
        best = current
        for direction in directions:
            for sign in (-1.0, 1.0):
                coordinates = tuple(
                    value + sign * step * float(delta)
                    for value, delta in zip(current.coordinates, direction)
                )
                candidate = family.evaluate(coordinates)
                if candidate.squared_systole > best.squared_systole + 1e-13:
                    best = candidate
        if best is current:
            step /= 2.0
        else:
            current = best
    return current


def refine_compatible_moduli_simplex(
    family: CompatibleMetricFamily,
    start: ModuliSample | Sequence[float],
    *,
    initial_step: float = 0.01,
    coordinate_tolerance: float = 1e-9,
    value_tolerance: float = 1e-12,
    maximum_iterations: int = 2500,
) -> ModuliSample:
    """Dependency-free adaptive Nelder--Mead refinement for a final candidate."""

    if initial_step <= 0 or coordinate_tolerance <= 0 or value_tolerance <= 0:
        raise ValueError("simplex scales and tolerances must be positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    origin = start if isinstance(start, ModuliSample) else family.evaluate(start)
    dimension = family.coordinate_dimension
    vertices = [np.asarray(origin.coordinates, dtype=float)]
    for axis in range(dimension):
        vertex = np.asarray(origin.coordinates, dtype=float).copy()
        vertex[axis] += initial_step
        vertices.append(vertex)
    samples = [family.evaluate(vertex) for vertex in vertices]

    reflection = 1.0
    expansion = 1.0 + 2.0 / dimension
    contraction = 0.75 - 1.0 / (2.0 * dimension)
    shrink = 1.0 - 1.0 / dimension

    for _ in range(maximum_iterations):
        order = sorted(range(dimension + 1), key=lambda index: -samples[index].squared_systole)
        vertices = [vertices[index] for index in order]
        samples = [samples[index] for index in order]
        values = [sample.squared_systole for sample in samples]
        diameter = max(float(np.linalg.norm(vertex - vertices[0])) for vertex in vertices[1:])
        if diameter <= coordinate_tolerance and max(values) - min(values) <= value_tolerance:
            break

        centroid = sum(vertices[:-1]) / dimension
        reflected_vertex = centroid + reflection * (centroid - vertices[-1])
        reflected = family.evaluate(reflected_vertex)
        if samples[0].squared_systole >= reflected.squared_systole > samples[-2].squared_systole:
            vertices[-1], samples[-1] = reflected_vertex, reflected
            continue
        if reflected.squared_systole > samples[0].squared_systole:
            expanded_vertex = centroid + expansion * (reflected_vertex - centroid)
            expanded = family.evaluate(expanded_vertex)
            if expanded.squared_systole > reflected.squared_systole:
                vertices[-1], samples[-1] = expanded_vertex, expanded
            else:
                vertices[-1], samples[-1] = reflected_vertex, reflected
            continue

        if reflected.squared_systole > samples[-1].squared_systole:
            contracted_vertex = centroid + contraction * (reflected_vertex - centroid)
        else:
            contracted_vertex = centroid + contraction * (vertices[-1] - centroid)
        contracted = family.evaluate(contracted_vertex)
        threshold = max(reflected.squared_systole, samples[-1].squared_systole)
        if contracted.squared_systole > threshold:
            vertices[-1], samples[-1] = contracted_vertex, contracted
            continue

        best_vertex = vertices[0]
        for index in range(1, dimension + 1):
            vertices[index] = best_vertex + shrink * (vertices[index] - best_vertex)
            samples[index] = family.evaluate(vertices[index])

    return max(samples, key=lambda sample: sample.squared_systole)


def scan_compatible_moduli(
    family: CompatibleMetricFamily,
    *,
    sample_count: int,
    seed: int,
    radius: float = 1.5,
    local_fraction: float = 0.25,
    local_radius: float = 0.2,
    refinement_starts: int = 12,
    refinement_initial_step: float = 0.15,
    refinement_minimum_step: float = 1e-5,
    refinement_rounds: int = 100,
) -> CompatibleModuliSearch:
    """Screen a full-dimensional coordinate box and refine its strongest samples."""

    if sample_count <= 0 or radius <= 0 or local_radius <= 0:
        raise ValueError("sample_count and radii must be positive")
    if not 0 <= local_fraction <= 1:
        raise ValueError("local_fraction must lie in [0,1]")
    if refinement_starts < 0:
        raise ValueError("refinement_starts must be nonnegative")
    if refinement_initial_step <= 0 or refinement_minimum_step <= 0:
        raise ValueError("refinement step sizes must be positive")
    if refinement_rounds <= 0:
        raise ValueError("refinement_rounds must be positive")
    random = np.random.default_rng(seed)
    shifts = tuple(float(value) for value in random.random(family.coordinate_dimension))
    local_count = int(round(sample_count * local_fraction))
    samples = []
    for offset in range(sample_count):
        sample_radius = local_radius if offset < local_count else radius
        coordinates = _halton_coordinates(
            offset + 1,
            family.coordinate_dimension,
            sample_radius,
            shifts,
        )
        samples.append(family.evaluate(coordinates))

    strongest = sorted(samples, key=lambda item: -item.squared_systole)[
        : min(refinement_starts, len(samples))
    ]
    refined = tuple(
        _pattern_refine(
            family,
            sample,
            initial_step=refinement_initial_step,
            minimum_step=refinement_minimum_step,
            maximum_rounds=refinement_rounds,
        )
        for sample in strongest
    )
    return CompatibleModuliSearch(
        family=family,
        samples=tuple(samples),
        refined_samples=refined,
        radius=radius,
        local_radius=local_radius,
        seed=seed,
    )


def high_precision_metric_systole(alternating, metric, decimal_places: int = 70) -> str:
    """Solve the finite-kernel CVPs for an arbitrary high-precision metric."""

    import mpmath as mp

    from .kernel import KernelGroup

    mp.mp.dps = decimal_places
    polarization = Polarization(alternating)
    size = len(metric)
    g = mp.matrix([[mp.mpf(str(value)) for value in row] for row in metric])
    upper = mp.cholesky(g).T
    global_best = None
    for element in KernelGroup.from_polarization(polarization).nonzero_elements:
        coordinates = [
            mp.mpf(value.numerator) / value.denominator for value in element.coordinates
        ]
        current = [0 for _ in range(size)]
        initial = [int(mp.floor(-value + mp.mpf("0.5"))) for value in coordinates]
        vector = mp.matrix([[coordinates[i] + initial[i]] for i in range(size)])
        class_best = (vector.T * g * vector)[0]

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
            lower = int(mp.ceil(center - radius - mp.mpf("1e-70")))
            upper_integer = int(mp.floor(center + radius + mp.mpf("1e-70")))
            for integer in sorted(
                range(lower, upper_integer + 1),
                key=lambda value: abs(mp.mpf(value) - center),
            ):
                current[index] = integer
                row_value = diagonal * (coordinates[index] + integer) + tail
                new_partial = partial + row_value * row_value
                if new_partial <= class_best + mp.mpf("1e-70"):
                    recurse(index - 1, new_partial)

        recurse(size - 1, mp.mpf("0"))
        if global_best is None or class_best < global_best:
            global_best = class_best
    return mp.nstr(global_best, decimal_places)


def high_precision_coordinate_systole(
    family: CompatibleMetricFamily,
    coordinates: Sequence[float],
    *,
    decimal_places: int = 70,
) -> str:
    """Reconstruct ``exp(H)^T G0 exp(H)`` and solve CVP with mpmath."""

    import mpmath as mp

    mp.mp.dps = decimal_places
    size = len(family.reference_metric)
    generator = mp.zeros(size)
    for coefficient, basis in zip(coordinates, family.tangent_basis):
        scalar = mp.mpf(str(float(coefficient)))
        generator += scalar * mp.matrix(
            [[mp.mpf(str(float(value))) for value in row] for row in basis]
        )
    transformation = mp.expm(generator)
    reference = mp.matrix(
        [[mp.mpf(str(value)) for value in row] for row in family.reference_metric]
    )
    metric = transformation.T * reference * transformation
    metric_rows = tuple(
        tuple(mp.nstr(metric[row, column], decimal_places) for column in range(size))
        for row in range(size)
    )
    return high_precision_metric_systole(family.alternating, metric_rows, decimal_places)


def gaussian_type_12_family() -> CompatibleMetricFamily:
    form = GaussianHermitianForm(2, 2, 1, 1)
    return CompatibleMetricFamily.from_reference(
        name="type (1,2) around coupled Gaussian CM",
        alternating=form.alternating,
        reference_metric=form.metric,
        reference_exact_ell_squared="1",
        reference_ell_squared=1.0,
        reference_cm="E_i^2 with coupled Gaussian Hermitian polarization",
    )


def eisenstein_type_13_family() -> CompatibleMetricFamily:
    form = EisensteinHermitianForm(2, 2, 1, 1)
    physical = tuple(
        tuple(float(value) / np.sqrt(3.0) for value in row)
        for row in form.metric_core
    )
    return CompatibleMetricFamily.from_reference(
        name="type (1,3) around coupled Eisenstein CM",
        alternating=form.alternating,
        reference_metric=physical,
        reference_exact_ell_squared="4/(3*sqrt(3))",
        reference_ell_squared=4.0 / (3.0 * np.sqrt(3.0)),
        reference_cm="E_omega^2 with coupled Eisenstein Hermitian polarization",
    )
