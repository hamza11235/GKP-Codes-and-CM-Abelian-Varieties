from gkp_systole import D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL

from gkp_passive_cliffords import period_model_logical_action


def test_d4_bolza_passive_clifford_images() -> None:
    level_two = period_model_logical_action(D4_PERIOD_MODEL, 2)
    level_three = period_model_logical_action(D4_PERIOD_MODEL, 3)
    assert level_two.automorphism_order == 48
    assert level_two.image_order == 24
    assert level_two.action_kernel_order == 2
    assert level_three.automorphism_order == 48
    assert level_three.image_order == 48
    assert level_three.action_kernel_order == 1


def test_klein_quartic_reproduces_paper_benchmark() -> None:
    level_two = period_model_logical_action(KLEIN_QUARTIC_PERIOD_MODEL, 2)
    level_three = period_model_logical_action(KLEIN_QUARTIC_PERIOD_MODEL, 3)
    assert level_two.automorphism_order == 336
    assert level_two.image_order == 168
    assert level_two.action_kernel_order == 2
    assert level_three.automorphism_order == 336
    assert level_three.image_order == 336
    assert level_three.action_kernel_order == 1


def test_higher_dimensional_image_kernel_identity() -> None:
    for model in (D4_PERIOD_MODEL, KLEIN_QUARTIC_PERIOD_MODEL):
        for level in (2, 3):
            action = period_model_logical_action(model, level)
            assert action.automorphism_order == action.image_order * action.action_kernel_order
            assert action.as_dict()["pairing_verified"] is True
