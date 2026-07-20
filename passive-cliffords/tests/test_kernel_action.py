from gkp_systole.kernel import KernelGroup
from gkp_systole.polarization import Polarization

from gkp_passive_cliffords.exact import identity_matrix
from gkp_passive_cliffords.kernel_action import induced_permutation, preserves_kernel_pairing


def test_identity_action_on_nonuniform_kernel() -> None:
    polarization = Polarization(
        (
            (0, 0, 1, 0),
            (0, 0, 0, 3),
            (-1, 0, 0, 0),
            (0, -3, 0, 0),
        )
    )
    kernel = KernelGroup.from_polarization(polarization)
    permutation = induced_permutation(identity_matrix(4), kernel)
    assert kernel.order == 9
    assert permutation == tuple(range(9))
    assert preserves_kernel_pairing(permutation, kernel)
