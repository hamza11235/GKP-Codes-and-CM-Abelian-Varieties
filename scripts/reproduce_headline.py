#!/usr/bin/env python3
"""Recompute and print the repository's certified headline results."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isqrt

from gkp_systole import (
    D4_PERIOD_MODEL,
    E8_PPAV_MODEL,
    KLEIN_QUARTIC_PERIOD_MODEL,
    TYPE_15_EXACT_MODEL,
    TYPE_112_EXACT_MODEL,
)


def scaled_expression(coefficient: Fraction, radicand: int) -> str:
    """Format ``coefficient/sqrt(radicand)`` after removing square factors."""
    square_factor = 1
    for candidate in range(1, isqrt(radicand) + 1):
        if radicand % (candidate * candidate) == 0:
            square_factor = candidate
    reduced_coefficient = coefficient / square_factor
    reduced_radicand = radicand // (square_factor * square_factor)
    if reduced_radicand == 1:
        return str(reduced_coefficient)
    if reduced_coefficient == 1:
        return f"1/sqrt({reduced_radicand})"
    return f"({reduced_coefficient})/sqrt({reduced_radicand})"


def main() -> None:
    d4 = D4_PERIOD_MODEL.compute_uniform_systole_via_svp(2)
    klein = KLEIN_QUARTIC_PERIOD_MODEL.compute_uniform_systole_via_svp(2)
    e8 = E8_PPAV_MODEL.compute_qubit_systole()
    type15_core = TYPE_15_EXACT_MODEL.core_relative_systole()
    type112_core = TYPE_112_EXACT_MODEL.core_relative_systole()

    assert d4.certified and d4.class_multiplicity == 12 and d4.lift_multiplicity == 24
    assert klein.certified and klein.class_multiplicity == 21 and klein.lift_multiplicity == 42
    assert e8.certified and e8.squared_systole == Fraction(1, 2)
    assert e8.class_multiplicity == 120 and e8.lift_multiplicity == 240
    assert TYPE_15_EXACT_MODEL.validation_certificate().certified
    assert type15_core.certified and type15_core.squared_systole == Fraction(2)
    assert type15_core.class_multiplicity == type15_core.lift_multiplicity == 24
    assert TYPE_15_EXACT_MODEL.cm_certificate().is_cm
    assert TYPE_112_EXACT_MODEL.validation_certificate().certified
    assert type112_core.certified and type112_core.squared_systole == Fraction(2)
    assert type112_core.class_multiplicity == 3 and type112_core.lift_multiplicity == 36
    assert TYPE_112_EXACT_MODEL.cm_certificate().is_cm

    results = {
        "d4_type_2_2": {
            "ell_squared": scaled_expression(
                d4.squared_systole_coefficient,
                D4_PERIOD_MODEL.scale_radicand,
            ),
            "decimal": d4.squared_systole,
            "classes": d4.class_multiplicity,
            "lifts": d4.lift_multiplicity,
            "status": "exact benchmark",
        },
        "klein_type_2_2_2": {
            "ell_squared": scaled_expression(
                klein.squared_systole_coefficient,
                KLEIN_QUARTIC_PERIOD_MODEL.scale_radicand,
            ),
            "decimal": klein.squared_systole,
            "classes": klein.class_multiplicity,
            "lifts": klein.lift_multiplicity,
            "status": "exact benchmark",
        },
        "e8_type_2_2_2_2": {
            "ell_squared": "1/2",
            "decimal": float(e8.squared_systole),
            "classes": e8.class_multiplicity,
            "lifts": e8.lift_multiplicity,
            "status": "global uniform optimum",
        },
        "type_1_5": {
            "ell_squared": TYPE_15_EXACT_MODEL.exact_squared_systole,
            "decimal": TYPE_15_EXACT_MODEL.squared_systole,
            "classes": type15_core.class_multiplicity,
            "lifts": type15_core.lift_multiplicity,
            "status": "exact CM project record; global optimality open",
        },
        "type_1_1_2": {
            "ell_squared": TYPE_112_EXACT_MODEL.exact_squared_systole,
            "decimal": TYPE_112_EXACT_MODEL.squared_systole,
            "classes": type112_core.class_multiplicity,
            "lifts": type112_core.lift_multiplicity,
            "status": "exact CM project record; global optimality open",
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
