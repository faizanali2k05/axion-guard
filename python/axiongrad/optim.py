"""Optimizers for Python AxionGrad modules."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ._axiongrad import Tensor


def _validate_learning_rate(learning_rate: float) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")


class SGD:
    def __init__(self, parameters: Iterable[Tensor], *, learning_rate: float) -> None:
        _validate_learning_rate(learning_rate)
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            parameter._replace_data([
                value - self.learning_rate * grad
                for value, grad in zip(parameter.data, gradient.data, strict=True)
            ])


class Adam:
    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
    ) -> None:
        _validate_learning_rate(learning_rate)
        beta1, beta2 = betas
        if not all(math.isfinite(beta) and 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must be finite and in [0, 1)")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first = [[0.0] * len(parameter.data) for parameter in self.parameters]
        self.second = [[0.0] * len(parameter.data) for parameter in self.parameters]
        self.timestep = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self.timestep += 1
        correction1 = 1.0 - self.beta1 ** self.timestep
        correction2 = 1.0 - self.beta2 ** self.timestep
        for index, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            if gradient is None:
                continue
            updated: list[float] = []
            for element, (value, grad) in enumerate(
                zip(parameter.data, gradient.data, strict=True)
            ):
                first = self.beta1 * self.first[index][element] + (1 - self.beta1) * grad
                second = self.beta2 * self.second[index][element] + (1 - self.beta2) * grad * grad
                self.first[index][element] = first
                self.second[index][element] = second
                updated.append(
                    value
                    - self.learning_rate * (first / correction1)
                    / (math.sqrt(second / correction2) + self.epsilon)
                )
            parameter._replace_data(updated)


__all__ = ["Adam", "SGD"]
