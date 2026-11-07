#![doc = "Neural-network modules, parameters, initialization, and losses."]

use std::fmt;
use std::sync::{Arc, Mutex};

use axiongrad_autograd::{AutogradError, Variable};
use axiongrad_core::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NnError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("parameter mutex was poisoned")]
    PoisonedParameter,
    #[error("Linear dimensions must be non-zero, received in={input}, out={output}")]
    InvalidLinearDimensions { input: usize, output: usize },
    #[error("parameter update shape {actual:?} does not match {expected:?}")]
    ParameterShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// A shared, trainable leaf. Optimizers replace its immutable autograd leaf
/// after each step, while all module and optimizer handles continue to share it.
#[derive(Clone)]
pub struct Parameter(Arc<Mutex<Variable>>);

impl fmt::Debug for Parameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Parameter").finish_non_exhaustive()
    }
}

impl Parameter {
    #[must_use]
    pub fn new(variable: Variable) -> Self {
        Self(Arc::new(Mutex::new(variable)))
    }

    #[must_use]
    pub fn from_tensor(value: Tensor) -> Self {
        Self::new(Variable::new(value, true))
    }

    pub fn variable(&self) -> Result<Variable, NnError> {
        self.0
            .lock()
            .map(|variable| variable.clone())
            .map_err(|_| NnError::PoisonedParameter)
    }

    pub fn value(&self) -> Result<Tensor, NnError> {
        Ok(self.variable()?.value().clone())
    }

    pub fn grad(&self) -> Result<Option<Tensor>, NnError> {
        Ok(self.variable()?.grad()?)
    }

    pub fn zero_grad(&self) -> Result<(), NnError> {
        self.variable()?.zero_grad()?;
        Ok(())
    }

    pub fn set_value(&self, value: Tensor) -> Result<(), NnError> {
        let mut variable = self.0.lock().map_err(|_| NnError::PoisonedParameter)?;
        if value.shape() != variable.value().shape() {
            return Err(NnError::ParameterShape {
                expected: variable.value().shape().to_vec(),
                actual: value.shape().to_vec(),
            });
        }
        *variable = Variable::new(value, true);
        Ok(())
    }
}

pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError>;

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Parameter,
    bias: Parameter,
    input_features: usize,
    output_features: usize,
}

impl Linear {
    /// Creates a dense layer using deterministic Xavier-uniform initialization.
    pub fn new(input: usize, output: usize, seed: u64) -> Result<Self, NnError> {
        if input == 0 || output == 0 {
            return Err(NnError::InvalidLinearDimensions { input, output });
        }
        let mut generator = SplitMix64::new(seed);
        let bound = (6.0 / (input + output) as f32).sqrt();
        let weights = (0..input * output)
            .map(|_| generator.uniform(-bound, bound))
            .collect();
        Ok(Self {
            weight: Parameter::from_tensor(Tensor::new(weights, vec![input, output])?),
            bias: Parameter::from_tensor(Tensor::zeros(vec![output])?),
            input_features: input,
            output_features: output,
        })
    }

    #[must_use]
    pub fn input_features(&self) -> usize {
        self.input_features
    }

    #[must_use]
    pub fn output_features(&self) -> usize {
        self.output_features
    }

    #[must_use]
    pub fn weight(&self) -> Parameter {
        self.weight.clone()
    }

    #[must_use]
    pub fn bias(&self) -> Parameter {
        self.bias.clone()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError> {
        Ok(input
            .matmul(&self.weight.variable()?)?
            .add(&self.bias.variable()?)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError> {
        Ok(input.relu()?)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError> {
        Ok(input.sigmoid()?)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError> {
        Ok(input.tanh()?)
    }
}

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    #[must_use]
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn push(&mut self, layer: impl Module + 'static) {
        self.layers.push(Box::new(layer));
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Result<Variable, NnError> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

pub fn mse_loss(prediction: &Variable, target: &Variable) -> Result<Variable, NnError> {
    let difference = prediction.sub(target)?;
    Ok(difference.mul(&difference)?.mean()?)
}

pub fn cross_entropy_loss(logits: &Variable, targets: &[usize]) -> Result<Variable, NnError> {
    Ok(logits.cross_entropy(targets)?)
}

/// Tiny deterministic generator used only for initialization, avoiding a large
/// random-number dependency in the runtime.
#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u32 << 24) as f32;
        low + (high - low) * unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Variable {
        Variable::new(
            Tensor::new(data.to_vec(), shape.to_vec()).unwrap(),
            requires_grad,
        )
    }

    #[test]
    fn linear_initialization_is_deterministic_and_discovers_parameters() {
        let first = Linear::new(3, 2, 42).unwrap();
        let second = Linear::new(3, 2, 42).unwrap();
        assert_eq!(
            first.weight().value().unwrap(),
            second.weight().value().unwrap()
        );
        assert_eq!(first.parameters().len(), 2);
        let output = first
            .forward(&tensor(&[1.0, 2.0, 3.0], &[1, 3], false))
            .unwrap();
        assert_eq!(output.value().shape(), &[1, 2]);
    }

    #[test]
    fn sequential_discovers_nested_parameters() {
        let network = Sequential::new(vec![
            Box::new(Linear::new(2, 4, 1).unwrap()),
            Box::new(Tanh),
            Box::new(Linear::new(4, 1, 2).unwrap()),
            Box::new(Sigmoid),
        ]);
        assert_eq!(network.len(), 4);
        assert_eq!(network.parameters().len(), 4);
    }

    #[test]
    fn mse_loss_gradient_is_correct() {
        let prediction = tensor(&[1.0, 3.0], &[2], true);
        let target = tensor(&[0.0, 1.0], &[2], false);
        mse_loss(&prediction, &target).unwrap().backward().unwrap();
        assert_eq!(prediction.grad().unwrap().unwrap().data(), &[1.0, 2.0]);
    }

    #[test]
    fn parameter_replacement_preserves_shared_handle() {
        let parameter = Parameter::from_tensor(Tensor::scalar(1.0));
        let shared = parameter.clone();
        parameter.set_value(Tensor::scalar(2.0)).unwrap();
        assert_eq!(shared.value().unwrap().data(), &[2.0]);
    }
}
