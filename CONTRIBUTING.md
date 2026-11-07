# Contributing to AxionGrad

Changes should be small, testable vertical slices. A numerical operator is not
complete until its forward behavior, error behavior, and gradient (when
applicable) are tested.

## Required checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pytest
```

Use safe, idiomatic Rust and explain any proposed unsafe code before adding it.
Do not add a machine-learning framework as a dependency. `ndarray` requires a
written justification showing why a local kernel is inappropriate. PyTorch is
allowed only in optional tests as an oracle.

Tests should include invalid inputs, zero-sized dimensions, scalar tensors,
broadcasting, shared graph nodes, and finite-difference checks where relevant.
Bug fixes require a regression test. Benchmarks must be reproducible and must
not imply broad performance conclusions from a narrow workload.

