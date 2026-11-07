# Supported operations

## Initial vertical slice

Core tensors are CPU-resident, owned, contiguous, row-major `f32` arrays.

- Metadata: shape, rank, strides, element count
- Shape operations: reshape and materializing axis transpose
- Access: checked multidimensional indexing
- Elementwise: add, subtract, multiply, divide with NumPy-style broadcasting
- Linear algebra: matrix multiplication for rank-2 tensors
- Reductions: full-tensor sum and mean
- Autograd: arithmetic, matmul, sum, mean, ReLU, sigmoid, and tanh
- Losses: MSE and mean stable softmax cross-entropy for class-index targets
- Modules: `Linear`, ReLU, sigmoid, tanh, and `Sequential`
- Optimizers: dense SGD and Adam
- Interfaces: native Rust plus PyO3 tensors and Python training utilities

Division by zero follows IEEE-754 (`inf`/`NaN`). Empty-tensor `mean` is rejected.
NaN and infinity inputs are preserved through ordinary floating-point kernels.

## Not yet supported

Axis-selective reductions, slicing, non-contiguous views, in-place mutation,
higher-order gradients, graph release controls, batched matmul, convolution,
pooling, sparse tensors, serialization, optimizer checkpointing, dtypes other
than `f32`, accelerators, and distributed execution are not implemented.

