# Training examples

## Small supervised examples

Two compact binaries exercise the optimizer stack without external data:

```bash
cargo run --release -p axiongrad-examples --bin linear_regression
cargo run --release -p axiongrad-examples --bin binary_classification
```

Linear regression learns `y = 3x + 2` with SGD and MSE. Binary classification
learns the AND truth table with a two-logit linear model, stable cross-entropy,
and Adam. Both use fixed initialization seeds and print initial/final metrics.

## XOR — verified

The XOR network is `2 -> 8 -> 1`, with tanh hidden activation, sigmoid output,
MSE loss, Xavier-uniform initialization, and Adam. Every operation, gradient,
and parameter update uses AxionGrad.

```bash
cargo run --release -p axiongrad-examples --bin xor
```

The checked-in record at [`examples/metrics/xor-seed-7.json`](../examples/metrics/xor-seed-7.json)
was produced by that command, not copied from a reference framework. Seed 7,
1,500 steps, and learning rate 0.05 produced:

- initial loss: `0.253895432`
- final loss: `0.000003317`
- accuracy: `1.00`
- predicted classes: `[0, 1, 1, 0]`

The same convergence condition runs in `cargo test` and through the Python API.

## MNIST MLP — implemented, dataset run not recorded

The MNIST example implements a `784 -> 128 -> 10` MLP with ReLU, stable
softmax cross-entropy, Adam, normalized IDX pixels, mini-batches, and per-epoch
loss/accuracy output. Supply the official uncompressed training IDX files:

```bash
cargo run --release -p axiongrad-examples --bin mnist -- \
  train-images-idx3-ubyte train-labels-idx1-ubyte 5 60000
```

The final `limit` argument is optional and is useful for smoke runs. The parser
rejects wrong magic numbers, dimensions, truncated files, and inconsistent
lengths. No MNIST accuracy is claimed because the dataset is not vendored and
was not available during this completion run.

