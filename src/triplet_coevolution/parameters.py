"""Command-independent argument metadata shared by the typed native service and its CLI."""

from dataclasses import dataclass
from functools import cache
from inspect import signature
from types import MappingProxyType, UnionType
from typing import get_args, get_origin


@dataclass(frozen=True)
class Option:
    flag: str
    help: str
    min: int | float | None = None


@cache
def analysis_types():
    """Return the JSON setting types derived from the same annotations as native CLI options."""
    from triplet_coevolution.analysis_engine import execute

    reserved = {"a", "b", "c", "out", "matches", "dry_run"}
    types = {}
    for name, param in signature(execute).parameters.items():
        if name in reserved or name.startswith(("_", "a_", "b_", "c_")):
            continue
        dtype = get_args(param.annotation)[0]
        if get_origin(dtype) is UnionType:
            (dtype,) = (arg for arg in get_args(dtype) if arg is not type(None))
        types[name] = get_origin(dtype) or dtype
    return MappingProxyType(types)
