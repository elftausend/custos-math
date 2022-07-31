# custos-math

[![Crates.io version](https://img.shields.io/crates/v/custos-math.svg)](https://crates.io/crates/custos-math)
[![Docs](https://docs.rs/custos-math/badge.svg?version=0.2.0)](https://docs.rs/custos-math/0.2.0/custos-math/)

This crate provides CUDA, OpenCL and CPU based matrix operations using [custos].

[custos]: https://github.com/elftausend/custos

## Installation

Add "custos-math" as a dependency:
You will also need [custos], if you want to run an example.
```toml
[dependencies]
custos-math = "0.2.0"
custos = "0.4.0"

# to disable the default features (cuda, opencl) and use an own set of features:
#custos-math = {version = "0.2.0", default-features=false, features=["opencl", "safe"]}
#custos = { version="0.4.0", default-features = false, features=["opencl", "safe"]}
```

## Example

```rust
use custos::{CPU, AsDev};
use custos_math::Matrix;

fn main() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));

    let c = a.gemm(&b);

    assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
}
```

Many more examples can be found in the tests and examples folder.