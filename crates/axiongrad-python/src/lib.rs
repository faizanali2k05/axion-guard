#![doc = "PyO3 bindings for AxionGrad. Numerical work remains in the Rust crates."]

use axiongrad_autograd::{AutogradError, Variable};
use axiongrad_core::Tensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "Tensor", module = "axiongrad._axiongrad")]
#[derive(Clone)]
struct PyTensor {
    inner: Variable,
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.value().shape().to_vec()
    }

    #[getter]
    fn data(&self) -> Vec<f32> {
        self.inner.value().data().to_vec()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Self>> {
        self.inner
            .grad()
            .map(|gradient| {
                gradient.map(|value| Self {
                    inner: Variable::new(value, false),
                })
            })
            .map_err(autograd_error)
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .add(&other.inner)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .sub(&other.inner)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .mul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .div(&other.inner)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn matmul(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .matmul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn sum(&self) -> Self {
        Self {
            inner: self.inner.sum(),
        }
    }

    fn mean(&self) -> PyResult<Self> {
        self.inner
            .mean()
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn relu(&self) -> PyResult<Self> {
        self.inner
            .relu()
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn sigmoid(&self) -> PyResult<Self> {
        self.inner
            .sigmoid()
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.inner
            .tanh()
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.inner
            .reshape(shape)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    #[pyo3(signature = (axis_a=0, axis_b=1))]
    fn transpose(&self, axis_a: usize, axis_b: usize) -> PyResult<Self> {
        self.inner
            .transpose(axis_a, axis_b)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn cross_entropy(&self, targets: Vec<usize>) -> PyResult<Self> {
        self.inner
            .cross_entropy(&targets)
            .map(|inner| Self { inner })
            .map_err(autograd_error)
    }

    fn backward(&self) -> PyResult<()> {
        self.inner.backward().map_err(autograd_error)
    }

    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad().map_err(autograd_error)
    }

    /// Internal optimizer hook. Replaces a leaf value and clears its graph/grad.
    fn _replace_data(&mut self, data: Vec<f32>) -> PyResult<()> {
        let value = Tensor::new(data, self.inner.value().shape().to_vec())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner = Variable::new(value, self.inner.requires_grad());
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(data={:?}, shape={:?}, requires_grad={})",
            self.inner.value().data(),
            self.inner.value().shape(),
            self.inner.requires_grad()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, shape, requires_grad=false))]
fn _tensor(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> PyResult<PyTensor> {
    Tensor::new(data, shape)
        .map(|value| PyTensor {
            inner: Variable::new(value, requires_grad),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn autograd_error(error: AutogradError) -> PyErr {
    match error {
        AutogradError::Tensor(error) => PyValueError::new_err(error.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pymodule]
fn _axiongrad(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_function(wrap_pyfunction!(_tensor, module)?)?;
    Ok(())
}
