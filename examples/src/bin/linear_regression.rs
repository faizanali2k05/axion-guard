use axiongrad_autograd::Variable;
use axiongrad_core::Tensor;
use axiongrad_nn::{Linear, Module, mse_loss};
use axiongrad_optim::{Optimizer, Sgd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = Variable::new(
        Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5, 1])?,
        false,
    );
    let targets = Variable::new(
        Tensor::new(vec![-4.0, -1.0, 2.0, 5.0, 8.0], vec![5, 1])?,
        false,
    );
    let model = Linear::new(1, 1, 101)?;
    let mut optimizer = Sgd::new(model.parameters(), 0.05)?;
    let initial_loss = mse_loss(&model.forward(&inputs)?, &targets)?.value().data()[0];

    for _ in 0..500 {
        optimizer.zero_grad()?;
        let loss = mse_loss(&model.forward(&inputs)?, &targets)?;
        loss.backward()?;
        optimizer.step()?;
    }

    let final_loss = mse_loss(&model.forward(&inputs)?, &targets)?.value().data()[0];
    println!("seed=101 initial_loss={initial_loss:.9} final_loss={final_loss:.9}");
    println!(
        "weight={:.6} bias={:.6}",
        model.weight().value()?.data()[0],
        model.bias().value()?.data()[0]
    );
    Ok(())
}
