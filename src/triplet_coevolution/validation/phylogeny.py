"""Exchangeability groups supplied by the user for matched-row randomization."""

import numpy as np


def exchangeability_groups(n: int, strata=None) -> tuple[np.ndarray, ...]:
    if strata is None:
        groups = (np.arange(n),)
    else:
        if len(strata) != n or any(not isinstance(key, str) or not key.strip() for key in strata):
            raise ValueError("permutation strata must give one nonempty group per observation")
        by_label = {}
        for index, label in enumerate(strata):
            by_label.setdefault(label, []).append(index)
        groups = tuple(np.asarray(indices) for indices in by_label.values())
    if not any(len(group) > 1 for group in groups):
        raise ValueError("the permutation null requires a group with at least two observations")
    return groups
