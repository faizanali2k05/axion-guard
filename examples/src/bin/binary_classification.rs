use axiongrad_autograd::Variable;
use axiongrad_core::Tensor;
use axiongrad_nn::{Linear, Module, cross_entropy_loss};
use axiongrad_optim::{Adam, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = Variable::new(
        Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2])?,
        false,
    );
    let targets = [0, 0, 0, 1]; // AND
    let model = Linear::new(2, 2, 202)?;
    let mut optimizer = Adam::new(model.parameters(), 0.05)?;
    let initial_loss = cross_entropy_loss(&model.forward(&inputs)?, &targets)?
        .value()
        .data()[0];

    for _ in 0..500 {
        optimizer.zero_grad()?;
        let loss = cross_entropy_loss(&model.forward(&inputs)?, &targets)?;
        loss.backward()?;
        optimizer.step()?;
    }

    let logits = model.forward(&inputs)?;
    let final_loss = cross_entropy_loss(&logits, &targets)?.value().data()[0];
    let classes: Vec<usize> = logits
        .value()
        .data()
        .chunks_exact(2)
        .map(|row| usize::from(row[1] > row[0]))
        .collect();
    let accuracy = classes
        .iter()
        .zip(targets)
        .filter(|(prediction, target)| **prediction == *target)
        .count() as f32
        / targets.len() as f32;
    println!(
        "seed=202 initial_loss={initial_loss:.9} final_loss={final_loss:.9} accuracy={accuracy:.2}"
    );
    println!("classes={classes:?}");
    Ok(())
}
