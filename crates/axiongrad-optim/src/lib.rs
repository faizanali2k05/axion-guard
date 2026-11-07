#![doc = "Stateful optimizers for AxionGrad parameters."]

use axiongrad_core::{Tensor, TensorError};
use axiongrad_nn::{NnError, Parameter};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error(transparent)]
    NeuralNetwork(#[from] NnError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("learning rate must be finite and positive, found {0}")]
    InvalidLearningRate(f32),
    #[error("Adam beta values must be finite and in [0, 1), found ({0}, {1})")]
    InvalidBetas(f32, f32),
    #[error("Adam epsilon must be finite and positive, found {0}")]
    InvalidEpsilon(f32),
}

pub trait Optimizer {
    fn zero_grad(&mut self) -> Result<(), OptimizerError>;
    fn step(&mut self) -> Result<(), OptimizerError>;
}

#[derive(Clone, Debug)]
pub struct Sgd {
    parameters: Vec<Parameter>,
    learning_rate: f32,
}

impl Sgd {
    pub fn new(parameters: Vec<Parameter>, learning_rate: f32) -> Result<Self, OptimizerError> {
        validate_learning_rate(learning_rate)?;
        Ok(Self {
            parameters,
            learning_rate,
        })
    }
}

impl Optimizer for Sgd {
    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for parameter in &self.parameters {
            parameter.zero_grad()?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), OptimizerError> {
        for parameter in &self.parameters {
            let Some(gradient) = parameter.grad()? else {
                continue;
            };
            let value = parameter.value()?;
            let updated: Vec<f32> = value
                .data()
                .iter()
                .zip(gradient.data())
                .map(|(&value, &gradient)| value - self.learning_rate * gradient)
                .collect();
            parameter.set_value(Tensor::new(updated, value.shape().to_vec())?)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Adam {
    parameters: Vec<Parameter>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    first_moments: Vec<Option<Tensor>>,
    second_moments: Vec<Option<Tensor>>,
    step: u64,
}

impl Adam {
    pub fn new(parameters: Vec<Parameter>, learning_rate: f32) -> Result<Self, OptimizerError> {
        Self::with_config(parameters, learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with_config(
        parameters: Vec<Parameter>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, OptimizerError> {
        validate_learning_rate(learning_rate)?;
        if !beta1.is_finite()
            || !beta2.is_finite()
            || !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
        {
            return Err(OptimizerError::InvalidBetas(beta1, beta2));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(OptimizerError::InvalidEpsilon(epsilon));
        }
        let count = parameters.len();
        Ok(Self {
            parameters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            first_moments: vec![None; count],
            second_moments: vec![None; count],
            step: 0,
        })
    }
}

impl Optimizer for Adam {
    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for parameter in &self.parameters {
            parameter.zero_grad()?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), OptimizerError> {
        self.step += 1;
        let first_correction = 1.0 - self.beta1.powf(self.step as f32);
        let second_correction = 1.0 - self.beta2.powf(self.step as f32);

        for (index, parameter) in self.parameters.iter().enumerate() {
            let Some(gradient) = parameter.grad()? else {
                continue;
            };
            let value = parameter.value()?;
            let mut first = self.first_moments[index]
                .clone()
                .unwrap_or(Tensor::zeros(value.shape().to_vec())?);
            let mut second = self.second_moments[index]
                .clone()
                .unwrap_or(Tensor::zeros(value.shape().to_vec())?);
            let mut updated = Vec::with_capacity(value.numel());
            let mut first_data = Vec::with_capacity(value.numel());
            let mut second_data = Vec::with_capacity(value.numel());

            for ((&value, &gradient), (&old_first, &old_second)) in value
                .data()
                .iter()
                .zip(gradient.data())
                .zip(first.data().iter().zip(second.data()))
            {
                let new_first = self.beta1 * old_first + (1.0 - self.beta1) * gradient;
                let new_second = self.beta2 * old_second + (1.0 - self.beta2) * gradient * gradient;
                let corrected_first = new_first / first_correction;
                let corrected_second = new_second / second_correction;
                updated.push(
                    value
                        - self.learning_rate * corrected_first
                            / (corrected_second.sqrt() + self.epsilon),
                );
                first_data.push(new_first);
                second_data.push(new_second);
            }
            first = Tensor::new(first_data, value.shape().to_vec())?;
            second = Tensor::new(second_data, value.shape().to_vec())?;
            self.first_moments[index] = Some(first);
            self.second_moments[index] = Some(second);
            parameter.set_value(Tensor::new(updated, value.shape().to_vec())?)?;
        }
        Ok(())
    }
}

fn validate_learning_rate(learning_rate: f32) -> Result<(), OptimizerError> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(OptimizerError::InvalidLearningRate(learning_rate));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use axiongrad_autograd::Variable;
    use axiongrad_nn::{Linear, Module, Sequential, Sigmoid, Tanh, mse_loss};

    use super::*;

    #[test]
    fn sgd_updates_and_clears_a_parameter() {
        let parameter = Parameter::from_tensor(Tensor::scalar(2.0));
        let variable = parameter.variable().unwrap();
        variable.mul(&variable).unwrap().backward().unwrap();
        let mut optimizer = Sgd::new(vec![parameter.clone()], 0.1).unwrap();
        optimizer.step().unwrap();
        assert!((parameter.value().unwrap().data()[0] - 1.6).abs() < 1e-6);
        assert!(parameter.grad().unwrap().is_none());
        optimizer.zero_grad().unwrap();
    }

    #[test]
    fn adam_learns_xor_reproducibly() {
        let inputs = Variable::new(
            Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]).unwrap(),
            false,
        );
        let targets = Variable::new(
            Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]).unwrap(),
            false,
        );
        let network = Sequential::new(vec![
            Box::new(Linear::new(2, 8, 7).unwrap()),
            Box::new(Tanh),
            Box::new(Linear::new(8, 1, 8).unwrap()),
            Box::new(Sigmoid),
        ]);
        let mut optimizer = Adam::new(network.parameters(), 0.05).unwrap();
        let initial_loss = mse_loss(&network.forward(&inputs).unwrap(), &targets)
            .unwrap()
            .value()
            .data()[0];
        for _ in 0..1_500 {
            optimizer.zero_grad().unwrap();
            let prediction = network.forward(&inputs).unwrap();
            let loss = mse_loss(&prediction, &targets).unwrap();
            loss.backward().unwrap();
            optimizer.step().unwrap();
        }
        let predictions = network.forward(&inputs).unwrap();
        let final_loss = mse_loss(&predictions, &targets).unwrap().value().data()[0];
        let classes: Vec<u8> = predictions
            .value()
            .data()
            .iter()
            .map(|&value| u8::from(value >= 0.5))
            .collect();
        assert!(
            final_loss < initial_loss * 0.01,
            "{initial_loss} -> {final_loss}"
        );
        assert_eq!(classes, vec![0, 1, 1, 0]);
    }
}
