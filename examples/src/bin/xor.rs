use axiongrad_autograd::Variable;
use axiongrad_core::Tensor;
use axiongrad_nn::{Linear, Module, Sequential, Sigmoid, Tanh, mse_loss};
use axiongrad_optim::{Adam, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const SEED: u64 = 7;
    const EPOCHS: usize = 1_500;
    let inputs = Variable::new(
        Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2])?,
        false,
    );
    let targets = Variable::new(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?, false);
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, SEED)?),
        Box::new(Tanh),
        Box::new(Linear::new(8, 1, SEED + 1)?),
        Box::new(Sigmoid),
    ]);
    let mut optimizer = Adam::new(model.parameters(), 0.05)?;
    let initial_loss = mse_loss(&model.forward(&inputs)?, &targets)?.value().data()[0];

    for _ in 0..EPOCHS {
        optimizer.zero_grad()?;
        let loss = mse_loss(&model.forward(&inputs)?, &targets)?;
        loss.backward()?;
        optimizer.step()?;
    }

    let predictions = model.forward(&inputs)?;
    let final_loss = mse_loss(&predictions, &targets)?.value().data()[0];
    let classes: Vec<u8> = predictions
        .value()
        .data()
        .iter()
        .map(|&value| u8::from(value >= 0.5))
        .collect();
    let accuracy = classes
        .iter()
        .zip([0, 1, 1, 0])
        .filter(|(prediction, target)| **prediction == *target)
        .count() as f32
        / 4.0;

    println!("{{");
    println!("  \"example\": \"xor\",");
    println!("  \"seed\": {SEED},");
    println!("  \"epochs\": {EPOCHS},");
    println!("  \"optimizer\": \"Adam\",");
    println!("  \"learning_rate\": 0.05,");
    println!("  \"initial_loss\": {initial_loss:.9},");
    println!("  \"final_loss\": {final_loss:.9},");
    println!("  \"accuracy\": {accuracy:.2},");
    println!("  \"predictions\": {:?},", predictions.value().data());
    println!("  \"classes\": {classes:?}");
    println!("}}");
    Ok(())
}
