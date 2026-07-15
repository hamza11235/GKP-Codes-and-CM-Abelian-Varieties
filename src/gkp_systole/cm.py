"""Exact CM elliptic-curve generation and low-dimensional survey helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt

from .models import PeriodModel, ScaledSystoleResult


@dataclass(frozen=True, order=True)
class ReducedQuadraticForm:
    """A primitive reduced positive-definite binary quadratic form [a,b,c]."""

    a: int
    b: int
    c: int

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be positive")
        if self.discriminant >= 0:
            raise ValueError("the discriminant must be negative")
        if gcd(gcd(abs(self.a), abs(self.b)), abs(self.c)) != 1:
            raise ValueError("the form must be primitive")
        if not self.is_reduced:
            raise ValueError("the form must satisfy the reduced-form conditions")

    @property
    def discriminant(self) -> int:
        return self.b * self.b - 4 * self.a * self.c

    @property
    def is_reduced(self) -> bool:
        if abs(self.b) > self.a or self.a > self.c:
            return False
        if (abs(self.b) == self.a or self.a == self.c) and self.b < 0:
            return False
        return True

    @property
    def tau_real(self) -> Fraction:
        return Fraction(-self.b, 2 * self.a)

    @property
    def tau_imaginary_core(self) -> Fraction:
        """Y_core for Im(tau)=Y_core/sqrt(abs(discriminant))."""

        return Fraction(abs(self.discriminant), 2 * self.a)

    def as_tuple(self) -> tuple[int, int, int]:
        return self.a, self.b, self.c


def reduced_primitive_forms(discriminant: int) -> tuple[ReducedQuadraticForm, ...]:
    """Enumerate reduced primitive forms of a fixed negative discriminant."""

    if discriminant >= 0 or discriminant % 4 not in (0, 1):
        raise ValueError("discriminant must be negative and congruent to 0 or 1 mod 4")
    absolute = abs(discriminant)
    # Reduced positive forms satisfy a <= sqrt(|Delta|/3).
    maximum_a = isqrt(absolute // 3) + 1
    forms: list[ReducedQuadraticForm] = []
    for a in range(1, maximum_a + 1):
        for b in range(-a, a + 1):
            numerator = b * b - discriminant
            denominator = 4 * a
            if numerator % denominator:
                continue
            c = numerator // denominator
            if a > c:
                continue
            if (abs(b) == a or a == c) and b < 0:
                continue
            if gcd(gcd(a, abs(b)), c) != 1:
                continue
            forms.append(ReducedQuadraticForm(a, b, c))
    return tuple(sorted(set(forms)))


def cm_elliptic_period_model(form: ReducedQuadraticForm) -> PeriodModel:
    """Construct the principally polarized CM elliptic curve associated to a form."""

    discriminant = form.discriminant
    return PeriodModel(
        name=f"CM elliptic curve Delta={discriminant}, form={form.as_tuple()}",
        real_part=((form.tau_real,),),
        imaginary_core=((form.tau_imaginary_core,),),
        scale_radicand=abs(discriminant),
        source="Generated from a reduced primitive positive-definite binary quadratic form.",
        cm_field=f"imaginary quadratic order of discriminant {discriminant}",
    )


@dataclass(frozen=True)
class CMEllipticResult:
    form: ReducedQuadraticForm
    model: PeriodModel
    systole_result: ScaledSystoleResult
    level: int

    @property
    def discriminant(self) -> int:
        return self.form.discriminant

    @property
    def squared_systole(self) -> float:
        return self.systole_result.squared_systole


def survey_cm_elliptic_curves(
    maximum_absolute_discriminant: int,
    *,
    level: int = 2,
) -> tuple[CMEllipticResult, ...]:
    """Generate and rank all reduced CM elliptic examples in a discriminant range."""

    if maximum_absolute_discriminant < 3:
        raise ValueError("maximum discriminant size must be at least 3")
    if level <= 1:
        raise ValueError("level must be greater than one")
    results: list[CMEllipticResult] = []
    for absolute in range(3, maximum_absolute_discriminant + 1):
        discriminant = -absolute
        if discriminant % 4 not in (0, 1):
            continue
        for form in reduced_primitive_forms(discriminant):
            model = cm_elliptic_period_model(form)
            result = model.compute_uniform_systole(level)
            results.append(CMEllipticResult(form, model, result, level))
    return tuple(
        sorted(
            results,
            key=lambda item: (
                -item.squared_systole,
                abs(item.discriminant),
                item.form.as_tuple(),
            ),
        )
    )


def scaled_squared_systole_greater(
    left: ScaledSystoleResult,
    right: ScaledSystoleResult,
) -> bool:
    """Compare positive values c/sqrt(r) without floating point."""

    left_coefficient = left.squared_systole_coefficient
    right_coefficient = right.squared_systole_coefficient
    return (
        left_coefficient * left_coefficient * right.model.scale_radicand
        > right_coefficient * right_coefficient * left.model.scale_radicand
    )
