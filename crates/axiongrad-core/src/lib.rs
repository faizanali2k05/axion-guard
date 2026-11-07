#![doc = "Owned CPU tensor storage and from-scratch dense f32 kernels."]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TensorError {
    #[error("shape {shape:?} requires {expected} values, but received {actual}")]
    DataLength {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    #[error("shape element count overflow for {0:?}")]
    ShapeOverflow(Vec<usize>),
    #[error("cannot reshape {from:?} ({from_numel} elements) to {to:?} ({to_numel} elements)")]
    InvalidReshape {
        from: Vec<usize>,
        from_numel: usize,
        to: Vec<usize>,
        to_numel: usize,
    },
    #[error("axis {axis} is out of bounds for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },
    #[error("index {index:?} is out of bounds for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("shapes {left:?} and {right:?} cannot be broadcast together")]
    IncompatibleBroadcast { left: Vec<usize>, right: Vec<usize> },
    #[error("matmul expects rank-2 tensors, received {left:?} and {right:?}")]
    MatmulRank { left: Vec<usize>, right: Vec<usize> },
    #[error("matmul inner dimensions differ: {left:?} and {right:?}")]
    MatmulShape { left: Vec<usize>, right: Vec<usize> },
    #[error("mean is undefined for an empty tensor")]
    EmptyMean,
    #[error("cannot reduce shape {from:?} to broadcast ancestor {to:?}")]
    InvalidSumToShape { from: Vec<usize>, to: Vec<usize> },
}

/// An owned, contiguous, row-major CPU tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = checked_numel(&shape)?;
        if data.len() != expected {
            return Err(TensorError::DataLength {
                shape,
                expected,
                actual: data.len(),
            });
        }
        let strides = contiguous_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self {
            data: vec![value],
            shape: Vec::new(),
            strides: Vec::new(),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let numel = checked_numel(&shape)?;
        Self::new(vec![0.0; numel], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let numel = checked_numel(&shape)?;
        Self::new(vec![1.0; numel], shape)
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let offset = self.offset(index)?;
        Ok(self.data[offset])
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let to_numel = checked_numel(&shape)?;
        if to_numel != self.numel() {
            return Err(TensorError::InvalidReshape {
                from: self.shape.clone(),
                from_numel: self.numel(),
                to: shape,
                to_numel,
            });
        }
        Self::new(self.data.clone(), shape)
    }

    /// Swaps two axes and materializes the result in contiguous row-major order.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self, TensorError> {
        let rank = self.rank();
        if axis_a >= rank {
            return Err(TensorError::InvalidAxis { axis: axis_a, rank });
        }
        if axis_b >= rank {
            return Err(TensorError::InvalidAxis { axis: axis_b, rank });
        }
        let mut out_shape = self.shape.clone();
        out_shape.swap(axis_a, axis_b);
        let mut out = vec![0.0; self.numel()];
        let out_strides = contiguous_strides(&out_shape);
        for (source_offset, value) in self.data.iter().copied().enumerate() {
            let mut coord = unravel_index(source_offset, &self.shape, &self.strides);
            coord.swap(axis_a, axis_b);
            let target_offset = dot_index(&coord, &out_strides);
            out[target_offset] = value;
        }
        Self::new(out, out_shape)
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_map(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_map(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_map(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_map(other, |a, b| a / b)
    }

    pub fn neg(&self) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().map(|value| -value).collect(),
            self.shape.clone(),
        )
    }

    pub fn map(&self, function: impl Fn(f32) -> f32) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().copied().map(function).collect(),
            self.shape.clone(),
        )
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::MatmulRank {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        let (rows, inner) = (self.shape[0], self.shape[1]);
        if inner != other.shape[0] {
            return Err(TensorError::MatmulShape {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        let cols = other.shape[1];
        let mut out = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let mut value = 0.0;
                for k in 0..inner {
                    value += self.data[row * inner + k] * other.data[k * cols + col];
                }
                out[row * cols + col] = value;
            }
        }
        Self::new(out, vec![rows, cols])
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        Self::scalar(self.data.iter().sum())
    }

    pub fn mean(&self) -> Result<Self, TensorError> {
        if self.is_empty() {
            return Err(TensorError::EmptyMean);
        }
        Ok(Self::scalar(
            self.data.iter().sum::<f32>() / self.numel() as f32,
        ))
    }

    /// Sums broadcast dimensions so the result has `target_shape`.
    pub fn sum_to_shape(&self, target_shape: &[usize]) -> Result<Self, TensorError> {
        if target_shape.len() > self.rank() || !is_broadcast_ancestor(target_shape, &self.shape) {
            return Err(TensorError::InvalidSumToShape {
                from: self.shape.clone(),
                to: target_shape.to_vec(),
            });
        }
        let target_numel = checked_numel(target_shape)?;
        let target_strides = contiguous_strides(target_shape);
        let rank_offset = self.rank() - target_shape.len();
        let mut output = vec![0.0; target_numel];
        for (source_offset, value) in self.data.iter().copied().enumerate() {
            let source_coord = unravel_index(source_offset, &self.shape, &self.strides);
            let mut target_offset = 0;
            for (axis, &dim) in target_shape.iter().enumerate() {
                let coord = if dim == 1 {
                    0
                } else {
                    source_coord[rank_offset + axis]
                };
                target_offset += coord * target_strides[axis];
            }
            output[target_offset] += value;
        }
        Self::new(output, target_shape.to_vec())
    }

    fn binary_map(
        &self,
        other: &Self,
        function: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let output_shape = broadcast_shape(&self.shape, &other.shape)?;
        let output_numel = checked_numel(&output_shape)?;
        let output_strides = contiguous_strides(&output_shape);
        let mut output = Vec::with_capacity(output_numel);
        for offset in 0..output_numel {
            let coord = unravel_index(offset, &output_shape, &output_strides);
            let left = self.data[broadcast_offset(&coord, &output_shape, self)];
            let right = other.data[broadcast_offset(&coord, &output_shape, other)];
            output.push(function(left, right));
        }
        Self::new(output, output_shape)
    }

    fn offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.rank()
            || index
                .iter()
                .zip(&self.shape)
                .any(|(&coordinate, &dimension)| coordinate >= dimension)
        {
            return Err(TensorError::IndexOutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        }
        Ok(dot_index(index, &self.strides))
    }
}

pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank = left.len().max(right.len());
    let mut output = vec![1; rank];
    for reverse_axis in 0..rank {
        let left_dim = left
            .len()
            .checked_sub(reverse_axis + 1)
            .map_or(1, |axis| left[axis]);
        let right_dim = right
            .len()
            .checked_sub(reverse_axis + 1)
            .map_or(1, |axis| right[axis]);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return Err(TensorError::IncompatibleBroadcast {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        }
        output[rank - reverse_axis - 1] = if left_dim == right_dim {
            left_dim
        } else if left_dim == 1 {
            right_dim
        } else {
            left_dim
        };
    }
    Ok(output)
}

fn checked_numel(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| TensorError::ShapeOverflow(shape.to_vec()))
    })
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride.saturating_mul(shape[axis]);
    }
    strides
}

fn unravel_index(mut offset: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coordinate = vec![0; shape.len()];
    for axis in 0..shape.len() {
        if shape[axis] != 0 {
            coordinate[axis] = offset / strides[axis];
            offset %= strides[axis];
        }
    }
    coordinate
}

fn dot_index(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides)
        .map(|(&coordinate, &stride)| coordinate * stride)
        .sum()
}

fn broadcast_offset(output_coord: &[usize], output_shape: &[usize], input: &Tensor) -> usize {
    let rank_offset = output_shape.len() - input.rank();
    input
        .shape
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            let coordinate = if dimension == 1 {
                0
            } else {
                output_coord[rank_offset + axis]
            };
            coordinate * input.strides[axis]
        })
        .sum()
}

fn is_broadcast_ancestor(ancestor: &[usize], descendant: &[usize]) -> bool {
    if ancestor.len() > descendant.len() {
        return false;
    }
    let offset = descendant.len() - ancestor.len();
    ancestor
        .iter()
        .zip(&descendant[offset..])
        .all(|(&source, &target)| source == target || source == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[test]
    fn scalar_and_zero_sized_shapes_have_distinct_element_counts() {
        assert_eq!(Tensor::scalar(3.0).numel(), 1);
        assert_eq!(Tensor::zeros(vec![2, 0, 3]).unwrap().numel(), 0);
    }

    #[test]
    fn validates_data_length_and_indices() {
        assert!(matches!(
            Tensor::new(vec![1.0], vec![2]),
            Err(TensorError::DataLength { .. })
        ));
        let value = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(value.get(&[1, 0]).unwrap(), 3.0);
        assert!(matches!(
            value.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn reshape_and_transpose_preserve_logical_values() {
        let value = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(value.reshape(vec![3, 2]).unwrap().shape(), &[3, 2]);
        let transposed = value.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transposed.strides(), &[2, 1]);
    }

    #[test]
    fn broadcasts_elementwise_operations() {
        let matrix = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row = tensor(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!(
            matrix.add(&row).unwrap().data(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
        assert!(matches!(
            matrix.add(&tensor(&[1.0, 2.0], &[2])),
            Err(TensorError::IncompatibleBroadcast { .. })
        ));
    }

    #[test]
    fn broadcasting_a_zero_dimension_with_one_stays_empty() {
        let empty = Tensor::zeros(vec![0, 3]).unwrap();
        let singleton = Tensor::ones(vec![1, 3]).unwrap();
        let result = empty.add(&singleton).unwrap();
        assert_eq!(result.shape(), &[0, 3]);
        assert!(result.is_empty());
    }

    #[test]
    fn division_uses_ieee_754_results() {
        let result = tensor(&[1.0, 0.0], &[2])
            .div(&tensor(&[0.0, 0.0], &[2]))
            .unwrap();
        assert!(result.data()[0].is_infinite());
        assert!(result.data()[1].is_nan());
    }

    #[test]
    fn matrix_multiplication_and_reductions_work() {
        let left = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let right = tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        assert_eq!(
            left.matmul(&right).unwrap().data(),
            &[58.0, 64.0, 139.0, 154.0]
        );
        assert_eq!(left.sum().data(), &[21.0]);
        assert_eq!(left.mean().unwrap().data(), &[3.5]);
        assert!(matches!(
            Tensor::zeros(vec![0]).unwrap().mean(),
            Err(TensorError::EmptyMean)
        ));
    }

    #[test]
    fn sums_broadcast_axes_back_to_parent_shape() {
        let gradient = Tensor::ones(vec![2, 3]).unwrap();
        assert_eq!(
            gradient.sum_to_shape(&[3]).unwrap().data(),
            &[2.0, 2.0, 2.0]
        );
        assert_eq!(gradient.sum_to_shape(&[]).unwrap().data(), &[6.0]);
    }
}
