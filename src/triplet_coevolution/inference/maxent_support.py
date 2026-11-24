"""Fixed native-state maps for local three-site statistics."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL


@dataclass(frozen=True)
class FrozenSupport:
    """Native symbols mapped into fixed bins, indexed by retained component and site."""

    molecules: tuple[str, ...]
    maps: tuple[tuple[tuple[int, ...], ...], ...]
    include_gaps: bool = False
    min_state_count: int = 0
    training_observations: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "molecules", tuple(self.molecules))
        object.__setattr__(self, "training_observations", tuple(self.training_observations))
        object.__setattr__(
            self, "maps", tuple(tuple(tuple(m) for m in sites) for sites in self.maps)
        )
        if len(self.maps) != 3 or len(self.molecules) != 3 or type(self.include_gaps) is not bool:
            raise ValueError("support must describe A, B and C and an explicit gap policy")
        if type(self.min_state_count) is not int or self.min_state_count < 0:
            raise ValueError("training minimum state count must be a nonnegative integer")
        if bool(self.training_observations) != bool(self.min_state_count):
            raise ValueError("rare-state pooling requires explicit training observations")
        if any(not isinstance(key, str) or not key for key in self.training_observations) or len(
            set(self.training_observations)
        ) != len(self.training_observations):
            raise ValueError("training observations must have unique nonempty IDs")
        for molecule, sites in zip(self.molecules, self.maps):
            if molecule not in CANONICAL or not sites:
                raise ValueError("invalid component molecule or empty support site set")
            q = len(CANONICAL[molecule]) + self.include_gaps
            for mapping in sites:
                if len(mapping) != q or any(type(x) is not int or x < 0 for x in mapping):
                    raise ValueError("state mapping must cover every declared native symbol")
                if set(mapping) != set(range(len(set(mapping)))):
                    raise ValueError("state mapping bins must be contiguous starting at zero")

    def shape_for(self, candidate: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(candidate) != 3 or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(sites)
            for i, sites in zip(candidate, self.maps)
        ):
            raise ValueError(
                "candidate must contain valid zero-based retained site indices for A, B and C"
            )
        return tuple(max(sites[i]) + 1 for sites, i in zip(self.maps, candidate))

    def metadata(self) -> dict:
        identity = all(
            mapping == tuple(range(len(mapping))) for sites in self.maps for mapping in sites
        )
        policy = (
            "train_pooled"
            if self.min_state_count
            else ("declared" if identity else "prespecified_pooled")
        )
        return {
            "schema_version": 1,
            "policy": policy,
            "coordinate_system": "retained sites in A, B, C order, zero-based native codes and bins",
            "molecules": list(self.molecules),
            "include_gaps": self.include_gaps,
            "native_symbols": [
                CANONICAL[m] + ("-" if self.include_gaps else "") for m in self.molecules
            ],
            "min_state_count": self.min_state_count,
            "training_observations": list(self.training_observations),
            "maps": [[list(m) for m in sites] for sites in self.maps],
        }

    @classmethod
    def from_metadata(cls, data: dict) -> "FrozenSupport":
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported frozen-support schema")
        support = cls(
            tuple(data["molecules"]),
            tuple(tuple(tuple(m) for m in sites) for sites in data["maps"]),
            data["include_gaps"],
            data["min_state_count"],
            tuple(data["training_observations"]),
        )
        if support.metadata() != data:
            raise ValueError("frozen-support metadata disagrees with the encoded state map")
        return support


def freeze_support(
    msa: MatchedTripletMSA,
    *,
    include_gaps: bool = False,
    training_observations=None,
    min_state_count: int = 0,
) -> FrozenSupport:
    """Freeze canonical alphabets, or pool training-rare native states into one OTHER bin."""
    if type(min_state_count) is not int or min_state_count < 0 or type(include_gaps) is not bool:
        raise ValueError("invalid minimum state count or gap policy")
    if isinstance(training_observations, str):
        raise TypeError("training observations must be a sequence of observation IDs")
    names = tuple(training_observations) if training_observations is not None else ()
    if bool(names) != bool(min_state_count):
        raise ValueError(
            "a positive minimum state count requires an explicit training observation set"
        )
    ids = {m.observation_id: i for i, m in enumerate(msa.matches)}
    if len(set(names)) != len(names) or any(key not in ids for key in names):
        raise ValueError("unknown or duplicate training observation ID")
    training = np.array([ids[key] for key in names], dtype=int)
    components = []
    for comp in msa.components:
        q = len(CANONICAL[comp.molecule]) + include_gaps
        if not names:
            components.append((tuple(range(q)),) * comp.n_sites)
            continue
        maps = []
        for col in range(comp.n_sites):
            values = comp.matrix[training, col]
            freq = np.bincount(values[(values >= 0) & (values < q)], minlength=q)
            common = np.flatnonzero(freq >= min_state_count)
            recode = np.full(q, len(common), dtype=int)
            recode[common] = np.arange(len(common))
            maps.append(tuple(map(int, recode)))
        components.append(tuple(maps))
    return FrozenSupport(
        tuple(comp.molecule.value for comp in msa.components),
        tuple(components),
        include_gaps,
        min_state_count,
        names,
    )
