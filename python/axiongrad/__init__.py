"""Python interface for the AxionGrad Rust engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from ._axiongrad import Tensor, _tensor

NestedData: TypeAlias = float | int | Sequence["NestedData"]


def _flatten(value: NestedData) -> tuple[list[float], list[int]]:
    if isinstance(value, (int, float)):
        return [float(value)], []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("tensor data must be numeric or a nested numeric sequence")
    children = [_flatten(child) for child in value]
    if not children:
        return [], [0]
    child_shape = children[0][1]
    if any(shape != child_shape for _, shape in children[1:]):
        raise ValueError("tensor data must have a regular, non-ragged shape")
    flat = [item for child, _ in children for item in child]
    return flat, [len(children), *child_shape]


def tensor(data: NestedData, *, requires_grad: bool = False) -> Tensor:
    """Create an owned contiguous f32 tensor from regular nested Python data."""
    flat, shape = _flatten(data)
    return _tensor(flat, shape, requires_grad)


__all__ = ["Tensor", "tensor"]

# Imported last so modules can reuse `tensor` without circular initialization.
from . import nn, optim

__all__ += ["nn", "optim"]

