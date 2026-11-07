#![doc = "Dynamic reverse-mode automatic differentiation for AxionGrad tensors."]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use axiongrad_core::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AutogradError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("backward() without an explicit gradient requires a scalar output, found {0:?}")]
    NonScalarBackward(Vec<usize>),
    #[error("backward gradient shape {gradient:?} does not match output shape {output:?}")]
    GradientShape {
        gradient: Vec<usize>,
        output: Vec<usize>,
    },
    #[error("an autograd gradient mutex was poisoned")]
    PoisonedGradient,
    #[error("cross_entropy expects rank-2 [batch, classes] logits, found {0:?}")]
    CrossEntropyRank(Vec<usize>),
    #[error("cross_entropy requires a non-empty batch and at least one class")]
    EmptyCrossEntropy,
    #[error("cross_entropy received {targets} targets for a batch of {batch}")]
    CrossEntropyTargetCount { targets: usize, batch: usize },
    #[error("cross_entropy target {target} at row {row} is outside 0..{classes}")]
    CrossEntropyTarget {
        row: usize,
        target: usize,
        classes: usize,
    },
    #[error("cross_entropy requires finite logits")]
    NonFiniteCrossEntropy,
}

#[derive(Clone)]
pub struct Variable(Arc<Node>);

struct Node {
    value: Tensor,
    requires_grad: bool,
    grad: Mutex<Option<Tensor>>,
    grad_fn: Option<GradFn>,
}

#[derive(Clone)]
enum GradFn {
    Add {
        left: Variable,
        right: Variable,
    },
    Sub {
        left: Variable,
        right: Variable,
    },
    Mul {
        left: Variable,
        right: Variable,
    },
    Div {
        left: Variable,
        right: Variable,
    },
    Matmul {
        left: Variable,
        right: Variable,
    },
    Sum {
        input: Variable,
    },
    Mean {
        input: Variable,
    },
    Relu {
        input: Variable,
    },
    Sigmoid {
        input: Variable,
        output: Tensor,
    },
    Tanh {
        input: Variable,
        output: Tensor,
    },
    Reshape {
        input: Variable,
    },
    Transpose {
        input: Variable,
        axis_a: usize,
        axis_b: usize,
    },
    CrossEntropy {
        logits: Variable,
        probabilities: Tensor,
        targets: Vec<usize>,
    },
}

impl fmt::Debug for Variable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Variable")
            .field("value", &self.0.value)
            .field("requires_grad", &self.0.requires_grad)
            .field("has_grad_fn", &self.0.grad_fn.is_some())
            .finish()
    }
}

impl Variable {
    #[must_use]
    pub fn new(value: Tensor, requires_grad: bool) -> Self {
        Self(Arc::new(Node {
            value,
            requires_grad,
            grad: Mutex::new(None),
            grad_fn: None,
        }))
    }

    #[must_use]
    pub fn value(&self) -> &Tensor {
        &self.0.value
    }

    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    pub fn grad(&self) -> Result<Option<Tensor>, AutogradError> {
        Ok(self.lock_grad()?.clone())
    }

    pub fn zero_grad(&self) -> Result<(), AutogradError> {
        *self.lock_grad()? = None;
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self, AutogradError> {
        let value = self.value().add(other.value())?;
        Ok(Self::operation(
            value,
            self.requires_grad() || other.requires_grad(),
            GradFn::Add {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    pub fn sub(&self, other: &Self) -> Result<Self, AutogradError> {
        let value = self.value().sub(other.value())?;
        Ok(Self::operation(
            value,
            self.requires_grad() || other.requires_grad(),
            GradFn::Sub {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, AutogradError> {
        let value = self.value().mul(other.value())?;
        Ok(Self::operation(
            value,
            self.requires_grad() || other.requires_grad(),
            GradFn::Mul {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    pub fn div(&self, other: &Self) -> Result<Self, AutogradError> {
        let value = self.value().div(other.value())?;
        Ok(Self::operation(
            value,
            self.requires_grad() || other.requires_grad(),
            GradFn::Div {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, AutogradError> {
        let value = self.value().matmul(other.value())?;
        Ok(Self::operation(
            value,
            self.requires_grad() || other.requires_grad(),
            GradFn::Matmul {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        Self::operation(
            self.value().sum(),
            self.requires_grad(),
            GradFn::Sum {
                input: self.clone(),
            },
        )
    }

    pub fn mean(&self) -> Result<Self, AutogradError> {
        Ok(Self::operation(
            self.value().mean()?,
            self.requires_grad(),
            GradFn::Mean {
                input: self.clone(),
            },
        ))
    }

    pub fn relu(&self) -> Result<Self, AutogradError> {
        let value = self.value().map(|element| {
            if element > 0.0 {
                element
            } else if element <= 0.0 {
                0.0
            } else {
                f32::NAN
            }
        })?;
        Ok(Self::operation(
            value,
            self.requires_grad(),
            GradFn::Relu {
                input: self.clone(),
            },
        ))
    }

    pub fn sigmoid(&self) -> Result<Self, AutogradError> {
        let value = self.value().map(stable_sigmoid)?;
        Ok(Self::operation(
            value.clone(),
            self.requires_grad(),
            GradFn::Sigmoid {
                input: self.clone(),
                output: value,
            },
        ))
    }

    pub fn tanh(&self) -> Result<Self, AutogradError> {
        let value = self.value().map(f32::tanh)?;
        Ok(Self::operation(
            value.clone(),
            self.requires_grad(),
            GradFn::Tanh {
                input: self.clone(),
                output: value,
            },
        ))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, AutogradError> {
        Ok(Self::operation(
            self.value().reshape(shape)?,
            self.requires_grad(),
            GradFn::Reshape {
                input: self.clone(),
            },
        ))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self, AutogradError> {
        Ok(Self::operation(
            self.value().transpose(axis_a, axis_b)?,
            self.requires_grad(),
            GradFn::Transpose {
                input: self.clone(),
                axis_a,
                axis_b,
            },
        ))
    }

    /// Numerically stable mean softmax cross-entropy for `[batch, classes]` logits.
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<Self, AutogradError> {
        if self.value().rank() != 2 {
            return Err(AutogradError::CrossEntropyRank(
                self.value().shape().to_vec(),
            ));
        }
        let batch = self.value().shape()[0];
        let classes = self.value().shape()[1];
        if batch == 0 || classes == 0 {
            return Err(AutogradError::EmptyCrossEntropy);
        }
        if targets.len() != batch {
            return Err(AutogradError::CrossEntropyTargetCount {
                targets: targets.len(),
                batch,
            });
        }
        if self.value().data().iter().any(|value| !value.is_finite()) {
            return Err(AutogradError::NonFiniteCrossEntropy);
        }

        let mut probabilities = vec![0.0; self.value().numel()];
        let mut total_loss = 0.0;
        for (row, &target) in targets.iter().enumerate() {
            if target >= classes {
                return Err(AutogradError::CrossEntropyTarget {
                    row,
                    target,
                    classes,
                });
            }
            let start = row * classes;
            let logits = &self.value().data()[start..start + classes];
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = logits.iter().map(|value| (value - maximum).exp()).sum();
            for (class, &logit) in logits.iter().enumerate() {
                probabilities[start + class] = (logit - maximum).exp() / denominator;
            }
            total_loss += denominator.ln() + maximum - logits[target];
        }
        let probabilities = Tensor::new(probabilities, vec![batch, classes])?;
        let value = Tensor::scalar(total_loss / batch as f32);
        Ok(Self::operation(
            value,
            self.requires_grad(),
            GradFn::CrossEntropy {
                logits: self.clone(),
                probabilities,
                targets: targets.to_vec(),
            },
        ))
    }

    pub fn backward(&self) -> Result<(), AutogradError> {
        if self.value().rank() != 0 {
            return Err(AutogradError::NonScalarBackward(
                self.value().shape().to_vec(),
            ));
        }
        self.backward_with_grad(Tensor::scalar(1.0))
    }

    pub fn backward_with_grad(&self, gradient: Tensor) -> Result<(), AutogradError> {
        if gradient.shape() != self.value().shape() {
            return Err(AutogradError::GradientShape {
                gradient: gradient.shape().to_vec(),
                output: self.value().shape().to_vec(),
            });
        }

        let mut topological = Vec::new();
        let mut visited = HashSet::new();
        build_topology(self, &mut visited, &mut topological);

        let mut pass_gradients = HashMap::new();
        pass_gradients.insert(self.id(), gradient);

        for variable in topological.iter().rev() {
            let Some(upstream) = pass_gradients.get(&variable.id()).cloned() else {
                continue;
            };
            if variable.requires_grad() {
                variable.accumulate_persistent(&upstream)?;
            }
            if let Some(grad_fn) = &variable.0.grad_fn {
                for (parent, contribution) in grad_fn.backward(&upstream)? {
                    if parent.requires_grad() {
                        accumulate_pass(&mut pass_gradients, parent.id(), contribution)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn operation(value: Tensor, requires_grad: bool, grad_fn: GradFn) -> Self {
        Self(Arc::new(Node {
            value,
            requires_grad,
            grad: Mutex::new(None),
            grad_fn: requires_grad.then_some(grad_fn),
        }))
    }

    fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    fn lock_grad(&self) -> Result<MutexGuard<'_, Option<Tensor>>, AutogradError> {
        self.0
            .grad
            .lock()
            .map_err(|_| AutogradError::PoisonedGradient)
    }

    fn accumulate_persistent(&self, contribution: &Tensor) -> Result<(), AutogradError> {
        let mut gradient = self.lock_grad()?;
        *gradient = Some(match gradient.take() {
            Some(existing) => existing.add(contribution)?,
            None => contribution.clone(),
        });
        Ok(())
    }
}

impl GradFn {
    fn parents(&self) -> Vec<&Variable> {
        match self {
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Matmul { left, right } => vec![left, right],
            Self::Sum { input }
            | Self::Mean { input }
            | Self::Relu { input }
            | Self::Sigmoid { input, .. }
            | Self::Tanh { input, .. }
            | Self::Reshape { input }
            | Self::Transpose { input, .. } => vec![input],
            Self::CrossEntropy { logits, .. } => vec![logits],
        }
    }

    fn backward(&self, upstream: &Tensor) -> Result<Vec<(Variable, Tensor)>, AutogradError> {
        match self {
            Self::Add { left, right } => Ok(vec![
                (left.clone(), upstream.sum_to_shape(left.value().shape())?),
                (right.clone(), upstream.sum_to_shape(right.value().shape())?),
            ]),
            Self::Sub { left, right } => Ok(vec![
                (left.clone(), upstream.sum_to_shape(left.value().shape())?),
                (
                    right.clone(),
                    upstream.neg()?.sum_to_shape(right.value().shape())?,
                ),
            ]),
            Self::Mul { left, right } => Ok(vec![
                (
                    left.clone(),
                    upstream
                        .mul(right.value())?
                        .sum_to_shape(left.value().shape())?,
                ),
                (
                    right.clone(),
                    upstream
                        .mul(left.value())?
                        .sum_to_shape(right.value().shape())?,
                ),
            ]),
            Self::Div { left, right } => {
                let left_gradient = upstream
                    .div(right.value())?
                    .sum_to_shape(left.value().shape())?;
                let denominator = right.value().mul(right.value())?;
                let right_gradient = upstream
                    .mul(left.value())?
                    .div(&denominator)?
                    .neg()?
                    .sum_to_shape(right.value().shape())?;
                Ok(vec![
                    (left.clone(), left_gradient),
                    (right.clone(), right_gradient),
                ])
            }
            Self::Matmul { left, right } => Ok(vec![
                (
                    left.clone(),
                    upstream.matmul(&right.value().transpose(0, 1)?)?,
                ),
                (
                    right.clone(),
                    left.value().transpose(0, 1)?.matmul(upstream)?,
                ),
            ]),
            Self::Sum { input } => Ok(vec![(
                input.clone(),
                Tensor::ones(input.value().shape().to_vec())?.mul(upstream)?,
            )]),
            Self::Mean { input } => {
                let scale = Tensor::scalar(input.value().numel() as f32);
                Ok(vec![(
                    input.clone(),
                    Tensor::ones(input.value().shape().to_vec())?
                        .mul(upstream)?
                        .div(&scale)?,
                )])
            }
            Self::Relu { input } => {
                let mask = input.value().map(|element| {
                    if element > 0.0 {
                        1.0
                    } else if element <= 0.0 {
                        0.0
                    } else {
                        f32::NAN
                    }
                })?;
                Ok(vec![(input.clone(), upstream.mul(&mask)?)])
            }
            Self::Sigmoid { input, output } => {
                let one_minus = Tensor::ones(output.shape().to_vec())?.sub(output)?;
                Ok(vec![(
                    input.clone(),
                    upstream.mul(&output.mul(&one_minus)?)?,
                )])
            }
            Self::Tanh { input, output } => {
                let one_minus_square =
                    Tensor::ones(output.shape().to_vec())?.sub(&output.mul(output)?)?;
                Ok(vec![(input.clone(), upstream.mul(&one_minus_square)?)])
            }
            Self::Reshape { input } => Ok(vec![(
                input.clone(),
                upstream.reshape(input.value().shape().to_vec())?,
            )]),
            Self::Transpose {
                input,
                axis_a,
                axis_b,
            } => Ok(vec![(input.clone(), upstream.transpose(*axis_a, *axis_b)?)]),
            Self::CrossEntropy {
                logits,
                probabilities,
                targets,
            } => {
                let batch = probabilities.shape()[0];
                let classes = probabilities.shape()[1];
                let scale = upstream.data()[0] / batch as f32;
                let mut gradient = probabilities.data().to_vec();
                for (row, &target) in targets.iter().enumerate() {
                    gradient[row * classes + target] -= 1.0;
                }
                for value in &mut gradient {
                    *value *= scale;
                }
                Ok(vec![(
                    logits.clone(),
                    Tensor::new(gradient, probabilities.shape().to_vec())?,
                )])
            }
        }
    }
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn build_topology(variable: &Variable, visited: &mut HashSet<usize>, output: &mut Vec<Variable>) {
    if !visited.insert(variable.id()) {
        return;
    }
    if let Some(grad_fn) = &variable.0.grad_fn {
        for parent in grad_fn.parents() {
            build_topology(parent, visited, output);
        }
    }
    output.push(variable.clone());
}

fn accumulate_pass(
    gradients: &mut HashMap<usize, Tensor>,
    id: usize,
    contribution: Tensor,
) -> Result<(), TensorError> {
    if let Some(existing) = gradients.get_mut(&id) {
        *existing = existing.add(&contribution)?;
    } else {
        gradients.insert(id, contribution);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: expected {expected}, found {actual}"
            );
        }
    }

    #[test]
    fn scalar_polynomial_matches_analytic_gradient() {
        let x = Variable::new(Tensor::scalar(3.0), true);
        let y = x.mul(&x).unwrap().add(&x).unwrap();
        y.backward().unwrap();
        assert_close(x.grad().unwrap().unwrap().data(), &[7.0], 1e-6);
    }

    #[test]
    fn branching_and_repeated_nodes_accumulate_once_per_path() {
        let x = Variable::new(Tensor::scalar(2.0), true);
        let square = x.mul(&x).unwrap();
        let output = square.add(&square).unwrap().add(&x).unwrap();
        output.backward().unwrap();
        assert_close(x.grad().unwrap().unwrap().data(), &[9.0], 1e-6);
    }

    #[test]
    fn backward_calls_accumulate_but_zero_grad_clears() {
        let x = Variable::new(Tensor::scalar(4.0), true);
        let output = x.mul(&x).unwrap();
        output.backward().unwrap();
        output.backward().unwrap();
        assert_close(x.grad().unwrap().unwrap().data(), &[16.0], 1e-6);
        x.zero_grad().unwrap();
        assert!(x.grad().unwrap().is_none());
    }

    #[test]
    fn broadcast_gradient_reduces_to_original_shape() {
        let matrix = Variable::new(tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let bias = Variable::new(tensor(&[0.1, 0.2, 0.3], &[3]), true);
        matrix.add(&bias).unwrap().sum().backward().unwrap();
        assert_close(
            matrix.grad().unwrap().unwrap().data(),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            1e-6,
        );
        assert_close(bias.grad().unwrap().unwrap().data(), &[2.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn matmul_gradient_has_expected_values() {
        let left = Variable::new(tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let right = Variable::new(tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]), true);
        left.matmul(&right).unwrap().sum().backward().unwrap();
        assert_close(
            left.grad().unwrap().unwrap().data(),
            &[11.0, 15.0, 11.0, 15.0],
            1e-6,
        );
        assert_close(
            right.grad().unwrap().unwrap().data(),
            &[4.0, 4.0, 6.0, 6.0],
            1e-6,
        );
    }

    #[test]
    fn activations_have_stable_and_correct_gradients() {
        let x = Variable::new(tensor(&[-100.0, -0.5, 0.0, 0.5, 100.0], &[5]), true);
        let output = x.sigmoid().unwrap().sum();
        assert!(output.value().data()[0].is_finite());
        output.backward().unwrap();
        let gradient = x.grad().unwrap().unwrap();
        assert!(gradient.data().iter().all(|value| value.is_finite()));
        assert_close(
            &gradient.data()[1..4],
            &[0.235_003_71, 0.25, 0.235_003_71],
            1e-6,
        );
    }

    #[test]
    fn relu_explicitly_propagates_nan() {
        let x = Variable::new(tensor(&[f32::NAN], &[1]), true);
        let output = x.relu().unwrap();
        assert!(output.value().data()[0].is_nan());
        output.sum().backward().unwrap();
        assert!(x.grad().unwrap().unwrap().data()[0].is_nan());
    }

    #[test]
    fn finite_difference_matches_composed_autograd_gradient() {
        fn evaluate(values: &[f32]) -> f32 {
            let x = tensor(values, &[3]);
            let sigmoid = x.map(stable_sigmoid).unwrap();
            x.mul(&x).unwrap().mul(&sigmoid).unwrap().sum().data()[0]
        }

        let values = vec![-0.8, 0.2, 1.1];
        let x = Variable::new(tensor(&values, &[3]), true);
        x.mul(&x)
            .unwrap()
            .mul(&x.sigmoid().unwrap())
            .unwrap()
            .sum()
            .backward()
            .unwrap();

        let epsilon = 1e-3;
        let numerical: Vec<f32> = (0..values.len())
            .map(|index| {
                let mut plus = values.clone();
                let mut minus = values.clone();
                plus[index] += epsilon;
                minus[index] -= epsilon;
                (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
            })
            .collect();
        assert_close(x.grad().unwrap().unwrap().data(), &numerical, 2e-4);
    }

    #[test]
    fn nonscalar_backward_requires_explicit_seed() {
        let x = Variable::new(tensor(&[1.0, 2.0], &[2]), true);
        assert!(matches!(
            x.backward(),
            Err(AutogradError::NonScalarBackward(_))
        ));
        x.backward_with_grad(tensor(&[2.0, 3.0], &[2])).unwrap();
        assert_close(x.grad().unwrap().unwrap().data(), &[2.0, 3.0], 1e-6);
    }

    #[test]
    fn stable_cross_entropy_matches_finite_differences() {
        fn evaluate(values: &[f32]) -> f32 {
            Variable::new(tensor(values, &[2, 3]), false)
                .cross_entropy(&[2, 0])
                .unwrap()
                .value()
                .data()[0]
        }

        let values = vec![1000.0, 999.0, 1001.0, -3.0, 2.0, 0.5];
        let logits = Variable::new(tensor(&values, &[2, 3]), true);
        logits.cross_entropy(&[2, 0]).unwrap().backward().unwrap();
        let epsilon = 0.01;
        let numerical: Vec<f32> = (0..values.len())
            .map(|index| {
                let mut plus = values.clone();
                let mut minus = values.clone();
                plus[index] += epsilon;
                minus[index] -= epsilon;
                (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
            })
            .collect();
        assert_close(logits.grad().unwrap().unwrap().data(), &numerical, 2e-3);
    }

    #[test]
    fn cross_entropy_rejects_invalid_targets_and_non_finite_logits() {
        let logits = Variable::new(tensor(&[1.0, 2.0], &[1, 2]), true);
        assert!(matches!(
            logits.cross_entropy(&[2]),
            Err(AutogradError::CrossEntropyTarget { .. })
        ));
        let non_finite = Variable::new(tensor(&[f32::NAN, 2.0], &[1, 2]), true);
        assert!(matches!(
            non_finite.cross_entropy(&[0]),
            Err(AutogradError::NonFiniteCrossEntropy)
        ));
    }
}
