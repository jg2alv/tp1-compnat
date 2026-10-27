"""Immutable syntax trees, with depth measured in edges from the root."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Iterator

from .primitives import PRIMITIVES


Path = tuple[int, ...]


@dataclass(frozen=True)
class Node:
    """One expression node; terminal nodes have size one and depth zero."""

    kind: str
    value: str | int | float
    children: tuple["Node", ...] = ()

    def __post_init__(self):
        # A tuple also prevents mutation when the constructor received a list.
        object.__setattr__(self, "children", tuple(self.children))
        if any(not isinstance(child, Node) for child in self.children):
            raise TypeError("Children must be Node instances.")
        if self.kind == "function":
            if not isinstance(self.value, str) or self.value not in PRIMITIVES:
                raise ValueError(f"Unknown primitive: {self.value!r}.")
            expected = PRIMITIVES[self.value].arity
            if len(self.children) != expected:
                raise ValueError(f"{self.value} requires {expected} children.")
        elif self.kind == "variable":
            if isinstance(self.value, bool) or not isinstance(self.value, Integral):
                raise ValueError("Variable index must be an integer (booleans are not allowed).")
            if self.value < 0:
                raise ValueError("Variable index must be a nonnegative integer.")
            object.__setattr__(self, "value", int(self.value))
            if self.children:
                raise ValueError("Variables cannot have children.")
        elif self.kind == "constant":
            if isinstance(self.value, bool) or not isinstance(self.value, Real):
                raise ValueError("Constants must be finite real numbers.")
            value = float(self.value)
            if not isfinite(value):
                raise ValueError("Constants must be finite real numbers.")
            object.__setattr__(self, "value", value)
            if self.children:
                raise ValueError("Constants cannot have children.")
        else:
            raise ValueError(f"Unknown node kind: {self.kind!r}.")

    @classmethod
    def function(cls, name: str, *children: "Node") -> "Node":
        return cls("function", name, tuple(children))

    @classmethod
    def variable(cls, index: int) -> "Node":
        return cls("variable", index)

    @classmethod
    def constant(cls, value: float) -> "Node":
        return cls("constant", value)

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def depth(self) -> int:
        return 1 + max(child.depth for child in self.children) if self.children else 0

    def iter_paths(self) -> Iterator[Path]:
        """Yield every node path once in preorder, including the root ``()``."""
        yield ()
        for index, child in enumerate(self.children):
            for path in child.iter_paths():
                yield (index,) + path

    def subtree(self, path: Path) -> "Node":
        current = self
        for index in path:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise IndexError("Path components must be nonnegative integers.")
            if index < 0 or index >= len(current.children):
                raise IndexError(f"Invalid subtree path: {path!r}.")
            current = current.children[index]
        return current

    def replace_subtree(self, path: Path, replacement: "Node") -> "Node":
        """Return a new tree while sharing unchanged immutable subtrees."""
        if not isinstance(replacement, Node):
            raise TypeError("The replacement must be a Node.")
        path = tuple(path)
        self.subtree(path)  # Validate even when a negative index could be valid Python.
        if not path:
            return replacement
        index, *tail = path
        children = list(self.children)
        children[index] = children[index].replace_subtree(tuple(tail), replacement)
        return Node(self.kind, self.value, tuple(children))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(
            data["kind"],
            data["value"],
            tuple(cls.from_dict(child) for child in data.get("children", [])),
        )

    def expression(self) -> str:
        """Render an unambiguous expression, preserving protected division."""
        if self.kind == "variable":
            return f"x{self.value}"
        if self.kind == "constant":
            return repr(self.value)
        arguments = [child.expression() for child in self.children]
        symbols = {"add": "+", "sub": "-", "mul": "*"}
        if self.value in symbols:
            return f"({arguments[0]} {symbols[self.value]} {arguments[1]})"
        if self.value == "neg":
            return f"(-{arguments[0]})"
        name = "protected_div" if self.value == "div" else self.value
        return f"{name}({', '.join(arguments)})"

    def signature(self) -> tuple:
        """Return a hashable structural key for counting exact duplicates."""
        return (self.kind, self.value, tuple(child.signature() for child in self.children))
