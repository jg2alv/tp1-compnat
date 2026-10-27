"""Real-valued primitives used to build symbolic regression expressions."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


DIVISION_EPSILON = 1e-12


def protected_division(numerator, denominator):
    """Divide arrays or scalars, returning one for near-zero denominators.

    ``np.divide(where=...)`` avoids evaluating the invalid division, unlike a
    conditional applied after dividing.
    """
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.ones(numerator.shape, dtype=float),
        where=~(np.abs(denominator) <= DIVISION_EPSILON),
    )


@dataclass(frozen=True)
class Primitive:
    """A numeric function and the number of operands it accepts."""

    name: str
    arity: int
    function: Callable


PRIMITIVES = {
    "add": Primitive("add", 2, np.add),
    "sub": Primitive("sub", 2, np.subtract),
    "mul": Primitive("mul", 2, np.multiply),
    "div": Primitive("div", 2, protected_division),
    "sin": Primitive("sin", 1, np.sin),
    "cos": Primitive("cos", 1, np.cos),
    "neg": Primitive("neg", 1, np.negative),
}

DEFAULT_FUNCTIONS = ("add", "sub", "mul", "div", "sin", "cos")
