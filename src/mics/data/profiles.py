"""Pre-registered modality-availability profiles for MICS experiments."""

from __future__ import annotations

import itertools
import random
from collections import Counter

from .brats import BRATS_MODALITIES

BRATS_BALANCED_PROFILE_SEED = 1701


def balanced_brats_modalities(
    *,
    seed: int = BRATS_BALANCED_PROFILE_SEED,
) -> dict[str, tuple[str, ...]]:
    """Return a modality-symmetric eight-client profile.

    Two clients receive all modalities and the other six receive each possible
    two-modality combination exactly once. The templates are assigned to client
    IDs with a fixed random permutation independent of patient data.
    """

    if seed < 0:
        raise ValueError("Modality-profile seed must be non-negative.")
    templates = [
        BRATS_MODALITIES,
        BRATS_MODALITIES,
        *itertools.combinations(BRATS_MODALITIES, 2),
    ]
    random.Random(seed).shuffle(templates)
    return {
        f"C{index}": tuple(template)
        for index, template in enumerate(templates, start=1)
    }


def validate_balanced_brats_modalities(
    profile: dict[str, tuple[str, ...]],
) -> None:
    """Verify equal prevalence and symmetric pair co-occurrence."""

    if set(profile) != {f"C{index}" for index in range(1, 9)}:
        raise ValueError("Balanced BraTS profile must contain clients C1 through C8.")
    modality_counts = Counter(
        modality
        for modalities in profile.values()
        for modality in modalities
    )
    if modality_counts != Counter({modality: 5 for modality in BRATS_MODALITIES}):
        raise ValueError("Every modality must be available at exactly five clients.")

    complete_clients = sum(
        set(modalities) == set(BRATS_MODALITIES)
        for modalities in profile.values()
    )
    pair_templates = Counter(
        frozenset(modalities)
        for modalities in profile.values()
        if len(modalities) == 2
    )
    expected_pairs = Counter(
        frozenset(pair)
        for pair in itertools.combinations(BRATS_MODALITIES, 2)
    )
    if complete_clients != 2 or pair_templates != expected_pairs:
        raise ValueError(
            "Balanced profile requires two complete clients and every modality "
            "pair exactly once."
        )

    pair_cooccurrence = Counter()
    for modalities in profile.values():
        for pair in itertools.combinations(sorted(modalities), 2):
            pair_cooccurrence[frozenset(pair)] += 1
    if set(pair_cooccurrence.values()) != {3}:
        raise ValueError("Every modality pair must co-occur at exactly three clients.")
