"""Orders of the elementary finite symplectic targets used in the survey."""

from __future__ import annotations

from math import prod
from typing import Sequence


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    candidate = 2
    while candidate * candidate <= value:
        if value % candidate == 0:
            return False
        candidate += 1
    return True


def elementary_prime_symplectic_order(polarization_type: Sequence[int]) -> int:
    """Return ``|Sp(2k,F_p)|`` after discarding spectator entries equal to one.

    This covers all nonuniform Phase 3 records.  A later phase can add the
    general composite-level finite symplectic group without changing the
    logical-action engine.
    """

    nontrivial = tuple(int(value) for value in polarization_type if int(value) > 1)
    if not nontrivial:
        return 1
    if len(set(nontrivial)) != 1 or not _is_prime(nontrivial[0]):
        raise ValueError("type is not elementary at one prime")
    prime = nontrivial[0]
    rank = len(nontrivial)
    return prime ** (rank * rank) * prod(
        prime ** (2 * index) - 1 for index in range(1, rank + 1)
    )
