# AxionGrad Roadmap

## Milestone 0 — contracts and vertical slice

- [x] Cargo workspace and architectural boundaries
- [x] Owned contiguous `f32` tensor contract
- [x] Dynamic graph lifecycle contract
- [x] Tensor arithmetic, broadcasting, reductions, and 2-D matmul
- [x] Reverse-mode gradients and numerical gradient tests
- [x] Minimal PyO3 tensor binding

Exit criterion: Rust tests cover forward edge cases, branching/shared graphs,
broadcast reductions, and finite-difference gradients; Python smoke tests pass.

## Milestone 1 — reliable tensor engine

- Axis-aware reductions, slicing, richer indexing, and explicit view semantics
- Kernel conformance tests for NaN, infinity, zero-sized dimensions, and errors
- Benchmarks with fixed seeds, warmups, hardware metadata, and matched `f32`
  workloads

## Milestone 2 — training primitives

- [x] `Parameter`, `Module`, `Linear`, activations, and `Sequential`
- [x] Deterministic seeded initialization
- [x] MSE and stable cross-entropy losses
- [x] SGD and Adam runtime state
- [x] AxionGrad-only XOR training with recorded loss and accuracy
- [ ] Optimizer state serialization

## Milestone 3 — Python usability

- [x] Core Python tensor/operator, layer, loss, and optimizer surface
- [x] Type hints, packaging, and exception mapping
- [ ] Python indexing and generated API documentation
- [ ] Optional PyTorch numerical-oracle tests

## Milestone 4 — end-to-end examples

- [x] Verified XOR MLP and implemented MNIST MLP
- [x] Reproducible XOR seed and metric record
- [x] Linear-regression and standalone binary-classification examples
- [ ] MNIST dataset run, environment capture, and checkpoints

## Milestone 5 — advanced CPU features

- Convolution, pooling, CNN modules, serialization, additional dtypes
- Profile-guided CPU kernels and memory profiling

GPU and distributed/federated work begins only after CPU correctness gates are
met and documented.

