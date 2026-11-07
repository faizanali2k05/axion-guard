import axiongrad as ag


def test_linear_parameters_receive_gradients() -> None:
    layer = ag.nn.Linear(2, 1, seed=11)
    output = layer(ag.tensor([[1.0, -1.0]])).sum()
    output.backward()
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_cross_entropy_is_stable_for_large_logits() -> None:
    logits = ag.tensor([[1000.0, 999.0, 1001.0]], requires_grad=True)
    loss = ag.nn.cross_entropy_loss(logits, [2])
    loss.backward()
    assert 0.0 < loss.data[0] < 2.0
    assert logits.grad is not None


def test_python_api_learns_xor_reproducibly() -> None:
    inputs = ag.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = ag.tensor([[0.0], [1.0], [1.0], [0.0]])
    model = ag.nn.Sequential(
        ag.nn.Linear(2, 8, seed=7),
        ag.nn.Tanh(),
        ag.nn.Linear(8, 1, seed=8),
        ag.nn.Sigmoid(),
    )
    optimizer = ag.optim.Adam(model.parameters(), learning_rate=0.05)
    initial_loss = ag.nn.mse_loss(model(inputs), targets).data[0]
    for _ in range(1_500):
        optimizer.zero_grad()
        loss = ag.nn.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    predictions = model(inputs)
    final_loss = ag.nn.mse_loss(predictions, targets).data[0]
    classes = [int(value >= 0.5) for value in predictions.data]
    assert final_loss < initial_loss * 0.01
    assert classes == [0, 1, 1, 0]
