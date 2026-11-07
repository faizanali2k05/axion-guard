# Security

AxionGrad is experimental research software, not a hardened inference or
training service. Do not load untrusted model or tensor files; serialization is
not implemented yet. The MNIST example reads local IDX files and validates
magic numbers, dimensions, checked sizes, and exact lengths before allocating
tensor data.

The Rust workspace forbids unsafe code. Python wheels contain native code and
should be built or installed only from a trusted source. Dependency changes
should be reviewed through `Cargo.lock`; CI uses locked Rust dependencies.

Please report suspected vulnerabilities privately to the repository owner. A
public security contact and response SLA must be added before any public stable
release. Avoid including secrets, proprietary datasets, or personal data in a
report.

