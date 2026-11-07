"""Neural-network modules implemented on the AxionGrad Rust tensor engine."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ._axiongrad import Tensor
from . import tensor


class Module(ABC):
    @abstractmethod
    def __call__(self, value: Tensor) -> Tensor: ...

    def parameters(self) -> list[Tensor]:
        return []


class _SplitMix64:
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return value ^ (value >> 31)

    def uniform(self, low: float, high: float) -> float:
        unit = float(self.next_u64() >> 40) / float(1 << 24)
        return low + (high - low) * unit


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, seed: int = 0) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        generator = _SplitMix64(seed)
        bound = math.sqrt(6.0 / (in_features + out_features))
        values = [
            generator.uniform(-bound, bound)
            for _ in range(in_features * out_features)
        ]
        self.weight = tensor(
            [values[row * out_features : (row + 1) * out_features]
             for row in range(in_features)],
            requires_grad=True,
        )
        self.bias = tensor([0.0] * out_features, requires_grad=True)

    def __call__(self, value: Tensor) -> Tensor:
        return value.matmul(self.weight) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, value: Tensor) -> Tensor:
        return value.relu()


class Sigmoid(Module):
    def __call__(self, value: Tensor) -> Tensor:
        return value.sigmoid()


class Tanh(Module):
    def __call__(self, value: Tensor) -> Tensor:
        return value.tanh()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def __call__(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value)
        return value

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    difference = prediction - target
    return (difference * difference).mean()


def cross_entropy_loss(logits: Tensor, targets: Sequence[int]) -> Tensor:
    return logits.cross_entropy(list(targets))


__all__ = [
    "Linear",
    "Module",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "cross_entropy_loss",
    "mse_loss",
]
