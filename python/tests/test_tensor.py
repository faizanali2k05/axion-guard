import math

import axiongrad as ag
import pytest


def test_square_sum_backward() -> None:
    x = ag.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x
    y.sum().backward()
    assert y.data == pytest.approx([1.0, 4.0, 9.0])
    assert x.grad is not None
    assert x.grad.data == pytest.approx([2.0, 4.0, 6.0])


def test_nested_shape_and_broadcast_gradient() -> None:
    matrix = ag.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = ag.tensor([0.5, -0.5], requires_grad=True)
    (matrix + bias).sum().backward()
    assert matrix.shape == [2, 2]
    assert bias.grad is not None
    assert bias.grad.data == pytest.approx([2.0, 2.0])


def test_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="ragged"):
        ag.tensor([[1.0], [2.0, 3.0]])


def test_stable_sigmoid() -> None:
    result = ag.tensor([-1000.0, 1000.0]).sigmoid()
    assert all(math.isfinite(value) for value in result.data)
    assert result.data == pytest.approx([0.0, 1.0])

