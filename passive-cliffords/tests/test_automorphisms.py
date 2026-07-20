from fractions import Fraction

from gkp_passive_cliffords import (
    GENERIC_RECTANGULAR,
    HEXAGONAL,
    SQUARE,
    elliptic_logical_action,
    integer_vectors_of_norm,
)


def test_integer_vectors_of_square_norm_one_are_coordinate_units() -> None:
    vectors = integer_vectors_of_norm(((1, 0), (0, 1)), Fraction(1))
    assert set(vectors) == {(-1, 0), (0, -1), (0, 1), (1, 0)}


def test_generic_rectangular_elliptic_curve_has_only_plus_minus_identity() -> None:
    action = elliptic_logical_action(GENERIC_RECTANGULAR, 3)
    assert action.automorphism_order == 2
    assert action.image_order == 2
    assert action.action_kernel_order == 1


def test_square_cm_units_and_level_two_image() -> None:
    action = elliptic_logical_action(SQUARE, 2)
    assert action.automorphism_order == 4
    assert action.image_order == 2
    assert action.action_kernel_order == 2


def test_hexagonal_cm_units_and_level_two_image() -> None:
    action = elliptic_logical_action(HEXAGONAL, 2)
    assert action.automorphism_order == 6
    assert action.image_order == 3
    assert action.action_kernel_order == 2


def test_odd_level_actions_are_faithful_for_square_and_hexagonal_units() -> None:
    for benchmark, expected in ((SQUARE, 4), (HEXAGONAL, 6)):
        action = elliptic_logical_action(benchmark, 3)
        assert action.automorphism_order == expected
        assert action.image_order == expected
        assert action.action_kernel_order == 1


def test_every_reported_action_preserves_the_finite_pairing() -> None:
    for benchmark in (GENERIC_RECTANGULAR, SQUARE, HEXAGONAL):
        assert elliptic_logical_action(benchmark, 5).as_dict()["pairing_verified"] is True
