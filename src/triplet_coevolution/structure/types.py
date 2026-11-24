"""Deposited and assembly-transformed structural coordinates in ångströms."""

from dataclasses import dataclass

ALTLOC_POLICY = (
    "For each residue select the monomer/conformer with highest mean positive heavy-atom "
    "occupancy, breaking ties by monomer and alternate ID. Include its shared blank atoms. "
    "Exclude atoms with zero or unknown occupancy."
)
IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
Transform = tuple[tuple[float, ...], ...]


class ChoiceRequired(ValueError):
    """An input left a model or biological assembly underdetermined."""

    def __init__(self, message, models, assemblies):
        super().__init__(message)
        self.available_models = tuple(sorted(models))
        self.available_assemblies = tuple(sorted(assemblies))


def resolve_selection(assembly, model, models, assemblies):
    if model is not None and model not in models:
        raise ValueError(f"deposited model {model} unavailable (available models {sorted(models)})")
    if assembly is not None and assembly != "asu" and assembly not in assemblies:
        raise ValueError(
            f"assembly {assembly!r} unavailable (available {sorted(assemblies)} or asu)"
        )
    if not models:
        raise ValueError("structure contains no deposited atom models")
    if model is None and len(models) != 1:
        raise ChoiceRequired(f"choose a deposited model from {sorted(models)}", models, assemblies)
    if assembly is None and len(assemblies) != 1:
        raise ChoiceRequired(
            f"choose a declared assembly from {sorted(assemblies)} or explicitly select 'asu'",
            models,
            assemblies,
        )
    return (
        next(iter(assemblies)) if assembly is None else assembly,
        next(iter(models)) if model is None else model,
    )


@dataclass(frozen=True)
class Atom:
    name: str
    element: str
    xyz: tuple[float, float, float]
    occupancy: float
    altloc: str


@dataclass(frozen=True)
class Residue:
    seq_index: int
    mon_id: str
    letter: str
    auth_seq_id: str | None
    insertion_code: str
    altloc: str
    atoms: tuple[Atom, ...]
    alternatives: tuple[str, ...] = ()
    declared_monomers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chain:
    copy_id: str
    label_asym_id: str
    auth_asym_id: str
    entity_id: str
    molecule: str
    organism: str
    taxonomy_id: str
    operations: tuple[str, ...]
    transform: Transform
    residues: tuple[Residue, ...]

    @property
    def sequence(self) -> str:
        """Complete polymer sequence in entity order, including unobserved sites."""
        return "".join(res.letter for res in self.residues)


@dataclass(frozen=True)
class Structure:
    entry_id: str
    format: str
    model: int
    available_models: tuple[int, ...]
    assembly: str
    available_assemblies: tuple[str, ...]
    chains: tuple[Chain, ...]
    sha256: str
    method: str
    warnings: tuple[str, ...]
    altloc_policy: str = ALTLOC_POLICY
