use std::env;
use std::fs;
use std::io;

use axiongrad_autograd::Variable;
use axiongrad_core::Tensor;
use axiongrad_nn::{Linear, Module, ReLU, Sequential, cross_entropy_loss};
use axiongrad_optim::{Adam, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() < 3 {
        eprintln!(
            "usage: mnist <train-images-idx3-ubyte> <train-labels-idx1-ubyte> [epochs] [limit]"
        );
        return Ok(());
    }
    let epochs = argument_or(&arguments, 3, 5)?;
    let requested_limit = argument_or(&arguments, 4, 60_000)?;
    let (images, image_count) = read_images(&arguments[1])?;
    let labels = read_labels(&arguments[2])?;
    let count = image_count.min(labels.len()).min(requested_limit);
    if count == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "MNIST dataset is empty").into());
    }

    const SEED: u64 = 2026;
    const FEATURES: usize = 28 * 28;
    const BATCH_SIZE: usize = 64;
    let model = Sequential::new(vec![
        Box::new(Linear::new(FEATURES, 128, SEED)?),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, SEED + 1)?),
    ]);
    let mut optimizer = Adam::new(model.parameters(), 1e-3)?;

    println!("seed={SEED} samples={count} batch_size={BATCH_SIZE}");
    for epoch in 0..epochs {
        let mut weighted_loss = 0.0;
        let mut correct = 0usize;
        for start in (0..count).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(count);
            let batch = end - start;
            let input = Variable::new(
                Tensor::new(
                    images[start * FEATURES..end * FEATURES].to_vec(),
                    vec![batch, FEATURES],
                )?,
                false,
            );
            let targets = &labels[start..end];
            optimizer.zero_grad()?;
            let logits = model.forward(&input)?;
            let loss = cross_entropy_loss(&logits, targets)?;
            weighted_loss += loss.value().data()[0] * batch as f32;
            correct += count_correct(logits.value(), targets);
            loss.backward()?;
            optimizer.step()?;
        }
        println!(
            "epoch={} loss={:.6} accuracy={:.4}",
            epoch + 1,
            weighted_loss / count as f32,
            correct as f32 / count as f32
        );
    }
    Ok(())
}

fn argument_or(
    arguments: &[String],
    index: usize,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(arguments
        .get(index)
        .map_or(Ok(default), |value| value.parse())?)
}

fn read_images(path: &str) -> Result<(Vec<f32>, usize), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 || u32_at(&bytes, 0)? != 2051 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "invalid MNIST image IDX file").into(),
        );
    }
    let count = u32_at(&bytes, 4)? as usize;
    let rows = u32_at(&bytes, 8)? as usize;
    let columns = u32_at(&bytes, 12)? as usize;
    let expected = 16usize
        .checked_add(
            count
                .checked_mul(rows * columns)
                .ok_or("image size overflow")?,
        )
        .ok_or("image size overflow")?;
    if rows != 28 || columns != 28 || bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid MNIST image dimensions or length",
        )
        .into());
    }
    Ok((
        bytes[16..]
            .iter()
            .map(|&pixel| f32::from(pixel) / 255.0)
            .collect(),
        count,
    ))
}

fn read_labels(path: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 || u32_at(&bytes, 0)? != 2049 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "invalid MNIST label IDX file").into(),
        );
    }
    let count = u32_at(&bytes, 4)? as usize;
    if bytes.len() != count + 8 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "invalid MNIST label length").into(),
        );
    }
    Ok(bytes[8..].iter().map(|&label| usize::from(label)).collect())
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, io::Error> {
    let value: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated IDX header"))?
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid IDX header"))?;
    Ok(u32::from_be_bytes(value))
}

fn count_correct(logits: &Tensor, targets: &[usize]) -> usize {
    let classes = logits.shape()[1];
    logits
        .data()
        .chunks_exact(classes)
        .zip(targets)
        .filter(|(row, target)| {
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map_or(0, |(index, _)| index);
            prediction == **target
        })
        .count()
}
