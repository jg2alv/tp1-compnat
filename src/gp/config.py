"""Validated configuration for one genetic programming run.

Tree depths count edges from the root, which has depth zero. Crossover and
mutation are exclusive alternatives; the remaining probability copies a parent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real
from pathlib import Path
import tomllib
from typing import Any

from .primitives import PRIMITIVES


@dataclass(frozen=True)
class GPConfig:
    """Parameters shared by initialization, variation, and evolution."""

    population_size: int = 100
    generations: int = 100
    min_initial_depth: int = 2
    max_depth: int = 7
    selection: str = "tournament"
    tournament_size: int = 2
    pc: float = 0.9
    pm: float = 0.05
    elite_size: int = 1
    functions: tuple[str, ...] = ("add", "sub", "mul", "div", "sin", "cos")
    constant_min: float = -5.0
    constant_max: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        # TOML arrays become lists; keep the public configuration immutable.
        if isinstance(self.functions, (list, tuple)):
            object.__setattr__(self, "functions", tuple(self.functions))
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when a parameter cannot define a valid run."""
        integer_fields = (
            "population_size",
            "generations",
            "min_initial_depth",
            "max_depth",
            "tournament_size",
            "elite_size",
            "seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")

        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 0:
            raise ValueError("generations must be nonnegative")
        if not 0 <= self.min_initial_depth <= self.max_depth:
            raise ValueError("depths must satisfy 0 <= min_initial_depth <= max_depth")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 1 and population_size")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be between 0 and population_size - 1")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.selection not in {"tournament", "roulette", "epsilon_lexicase"}:
            raise ValueError("selection must be tournament, roulette, or epsilon_lexicase")

        for name in ("pc", "pm", "constant_min", "constant_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.pc <= 1 or not 0 <= self.pm <= 1:
            raise ValueError("pc and pm must each be between 0 and 1")
        if self.pc + self.pm > 1:
            raise ValueError("pc + pm must not exceed 1; operators are exclusive alternatives")
        if self.constant_min > self.constant_max:
            raise ValueError("constant_min must not exceed constant_max")

        if not isinstance(self.functions, tuple) or not self.functions:
            raise ValueError("functions must be a nonempty list or tuple of primitive names")
        if any(not isinstance(name, str) for name in self.functions):
            raise ValueError("each function must be a primitive name")
        unknown = set(self.functions).difference(PRIMITIVES)
        if unknown:
            raise ValueError(f"unknown functions: {', '.join(sorted(unknown))}")
        if len(set(self.functions)) != len(self.functions):
            raise ValueError("functions must not contain duplicate primitive names")

    @property
    def constant_range(self) -> tuple[float, float]:
        """Inclusive bounds used to sample ephemeral numeric constants."""
        return (self.constant_min, self.constant_max)

    @property
    def reproduction_probability(self) -> float:
        """Probability left for unchanged reproduction."""
        return max(0.0, 1.0 - self.pc - self.pm)

    def to_dict(self) -> dict[str, Any]:
        """Return all effective parameters for result metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> GPConfig:
        """Build a configuration, rejecting misspelled or unsupported fields."""
        if not isinstance(mapping, Mapping):
            raise ValueError("configuration must be a mapping")
        known = {field.name for field in fields(cls)}
        unknown = set(mapping).difference(known)
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"unknown configuration fields: {names}")
        return cls(**dict(mapping))


def load_config(path: str | Path) -> GPConfig:
    """Read a TOML file containing either a ``[gp]`` section or root fields."""
    with Path(path).open("rb") as source:
        contents = tomllib.load(source)
    values = contents.get("gp", contents)
    return GPConfig.from_dict(values)
