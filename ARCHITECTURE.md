# AxionGrad Architecture

AxionGrad is a from-scratch, CPU-first deep-learning framework. PyTorch,
TensorFlow, JAX, Burn, and Candle are not runtime dependencies. PyTorch may be
used only by optional comparison tests.

## Workspace boundaries

- `axiongrad-core` owns dense tensor storage, shapes, strides, indexing,
  broadcasting, elementwise kernels, reductions, and matrix multiplication.
- `axiongrad-autograd` owns differentiable values, dynamic graphs, gradient
  rules, topological reverse traversal, and gradient accumulation.
- `axiongrad-nn` owns shared `Parameter` handles, `Module`, dense layers,
  activations, losses, deterministic initialization, and model composition.
- `axiongrad-optim` owns SGD and Adam state and parameter updates.
- `axiongrad-python` exposes the Rust implementation through PyO3. It contains
  bindings and conversions, not numerical kernels.
- `examples`, `benchmarks`, and `docs` contain executable learning material,
  reproducible measurements, and design/math documentation respectively.

Dependencies point inward: Python -> optim -> nn -> autograd -> core. Core does
not know about graphs, modules, Python, or training.

## Tensor ownership model

The first storage implementation is an owned `Vec<f32>`. A core `Tensor` owns
its allocation and records a shape plus row-major strides. Cloning a core
tensor is an explicit deep copy. This is intentionally simple and safe; a
future storage/view layer may introduce shared buffers only after aliasing and
mutation semantics are specified and benchmarked.

Tensors are contiguous in this milestone. `reshape` validates equal element
counts and preserves logical element order. `transpose` materializes a new
contiguous tensor instead of creating a strided view. The API still exposes
strides so a future view implementation can evolve without changing shape
semantics.

An empty shape (`[]`) denotes a scalar and therefore has one element. Shapes
containing a zero dimension have zero elements. Invalid shapes, indices,
broadcasts, reductions, and non-finite division outcomes are surfaced as typed
errors where applicable; elementwise floating-point inputs may themselves be
NaN or infinite and propagate according to IEEE-754.

## Autograd graph lifecycle

An autograd `Variable` is a cheap clone of an `Arc`-owned immutable graph node.
Each node owns its tensor value, `requires_grad` flag, optional accumulated
gradient behind a mutex, and an enum describing its local gradient rule and
strong references to its parents. Graph edges point only from result to
parents, so graphs are acyclic by construction through the public API and are
freed when all result/leaf handles are dropped.

Every operation creates a graph node only when at least one parent requires a
gradient. `backward()` builds a post-order topological list keyed by node
identity, seeds the output gradient, and visits each node once in reverse
topological order. Contributions are summed before a node's rule runs, which
handles branches, repeated operands, and shared subgraphs correctly.
Broadcasted operands use `sum_to_shape` in their backward rules. Gradients
accumulate across separate `backward()` calls until `zero_grad()` is called.
This first implementation retains its graph for repeatable backward passes; a
future `retain_graph` option can make release behavior explicit.

## Safety and concurrency

The implementation forbids unsafe Rust. Tensor values are immutable through
the current autograd API. Gradient mutation is synchronized, making node
handles safe to move between threads, although kernels are single-threaded in
this milestone.

## Parameters and optimizer steps

Autograd variables remain immutable. A `Parameter` is an
`Arc<Mutex<Variable>>` shared by its module and optimizer. Forward passes clone
the current leaf handle. After backward, an optimizer reads that leaf's
gradient, computes a new owned tensor, and replaces the parameter with a fresh
leaf. This clears the old graph and gradient without introducing in-place
aliases into the autograd DAG. Adam moment tensors remain optimizer-owned.

## Current scope

The CPU slice supports contiguous `f32` tensors, broadcasting, basic
arithmetic, 2-D matrix multiplication, full-tensor `sum`/`mean`, reverse-mode
gradients, dense networks, MSE, stable cross-entropy, SGD, and Adam. Views,
in-place operations, axis reductions, devices, convolution, serialization,
additional dtypes, and GPU execution are deliberately unsupported.

