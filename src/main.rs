mod engine;
mod nn;

use crate::engine::Value;
use crate::nn::{Module, MLP};
use plotters::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn generate_moons(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Use rand 0.9 API!
    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    let normal = Normal::new(0.0, noise).unwrap();

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    let n_per_moon = n_samples / 2;

    // First moon: points along a half-circle from 0 to π
    for _ in 0..n_per_moon {
        let angle = rng.random_range(0.0..std::f64::consts::PI);
        let x1 = angle.cos() + normal.sample(&mut rng);
        let y1 = angle.sin() + normal.sample(&mut rng);
        x.push(vec![x1, y1]);
        y.push(1.0);
    }
    // Second moon: positioned to interleave with the first moon
    for _ in 0..(n_samples - n_per_moon) {
        let angle = rng.random_range(0.0..std::f64::consts::PI);
        let x2 = 1.0 - angle.cos() + normal.sample(&mut rng); // Slight horizontal offset
        let y2 = 0.25 - angle.sin() + normal.sample(&mut rng); // Reduced vertical offset to increase overlap
        x.push(vec![x2, y2]);
        y.push(-1.0);
    }
    // Shuffle
    let mut idx: Vec<_> = (0..x.len()).collect();
    idx.shuffle(&mut rng);
    let x_shuffled = idx.iter().map(|&i| x[i].clone()).collect();
    let y_shuffled = idx.iter().map(|&i| y[i]).collect();
    (x_shuffled, y_shuffled)
}

fn loss(
    model: &MLP,
    x: &[Vec<f64>],
    y: &[f64],
    batch_size: Option<usize>,
) -> (std::rc::Rc<Value>, f64) {
    let (xb, yb) = if let Some(bs) = batch_size {
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.shuffle(&mut rand::rng());
        let xb: Vec<Vec<f64>> = indices.iter().take(bs).map(|&i| x[i].clone()).collect();
        let yb: Vec<f64> = indices.iter().take(bs).map(|&i| y[i]).collect();
        (xb, yb)
    } else {
        (x.to_vec(), y.to_vec())
    };

    // Rest of the function remains unchanged
    let inputs: Vec<Vec<std::rc::Rc<Value>>> = xb
        .iter()
        .map(|row| row.iter().map(|&v| Value::new(v)).collect())
        .collect();
    let scores: Vec<std::rc::Rc<Value>> = inputs
        .iter()
        .map(|input| model.call(input.clone())[0].clone())
        .collect();

    // SVM "max-margin" loss
    let losses: Vec<std::rc::Rc<Value>> = yb
        .iter()
        .zip(scores.iter())
        .map(|(&yi, scorei)| {
            let neg_score = Value::new(-1.0).mul(scorei);
            let prod = Value::new(yi).mul(&neg_score);
            let one_plus = Value::new(1.0).add(&prod);
            one_plus.relu()
        })
        .collect();

    let data_loss = losses
        .iter()
        .fold(Value::new(0.0), |acc, loss| acc.add(loss))
        .mul(&Value::new(1.0 / losses.len() as f64));

    // L2 regularization
    let alpha = Value::new(1e-4);
    let reg_loss = model
        .parameters()
        .iter()
        .fold(Value::new(0.0), |acc, p| {
            let p_sq = p.mul(p);
            acc.add(&p_sq)
        })
        .mul(&alpha);

    let total_loss = data_loss.add(&reg_loss);

    // Compute accuracy
    let accuracy = yb
        .iter()
        .zip(scores.iter())
        .map(|(&yi, scorei)| (yi > 0.0) == (*scorei.data.borrow() > 0.0))
        .filter(|&b| b)
        .count() as f64
        / yb.len() as f64;

    (total_loss, accuracy)
}

fn main() {
    // Generate a moons-like dataset
    let n_samples = 100;
    let (x, y) = generate_moons(n_samples, 0.2, Some(3141592653));

    // Initialize model: 2 input dimensions, two hidden layers of 16 nodes each, 1 output
    let model = MLP::new(2, vec![16, 32, 1]);
    println!("Number of parameters: {}", model.parameters().len());

    // Training loop
    for k in 0..100 {
        // Forward pass and compute loss
        let (total_loss, acc) = loss(&model, &x, &y, None);

        // Backward pass
        model.zero_grad();
        total_loss.backward();

        // Update weights with SGD
        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;
        for p in model.parameters() {
            let update = learning_rate * *p.grad.borrow();
            *p.data.borrow_mut() -= update;
        }

        if k % 10 == 0 {
            println!(
                "Step {} loss {}, accuracy {}%",
                k,
                *total_loss.data.borrow(),
                acc * 100.0
            );
        }
    }

    // Visualize decision boundary (simplified)
    visualize_decision_boundary(&model, &x, &y);
}

fn visualize_decision_boundary(model: &MLP, x: &[Vec<f64>], y: &[f64]) {
    let root = BitMapBackend::new("decision_boundary.png", (600, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Grid
    let x_min = x.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min) - 0.5;
    let x_max = x.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max) + 0.5;
    let y_min = x.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min) - 0.5;
    let y_max = x.iter().map(|v| v[1]).fold(f64::NEG_INFINITY, f64::max) + 0.5;

    let h = 0.025;
    let nx = ((x_max - x_min) / h) as usize;
    let ny = ((y_max - y_min) / h) as usize;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // Decision boundary as filled grid
    chart
        .draw_series((0..nx).flat_map(|ix| {
            (0..ny).map(move |iy| {
                let x0 = x_min + ix as f64 * h;
                let y0 = y_min + iy as f64 * h;
                let input = vec![Value::new(x0), Value::new(y0)];
                let score = *model.call(input)[0].data.borrow();
                let color = if score > 0.0 {
                    RGBColor(252, 141, 89).mix(0.6)
                } else {
                    RGBColor(145, 191, 219).mix(0.6)
                };
                Rectangle::new(
                    [(x0, y0), (x0 + h, y0 + h)],
                    ShapeStyle::from(&color).filled(),
                )
            })
        }))
        .unwrap();

    // Overlay data points
    for (point, &label) in x.iter().zip(y.iter()) {
        let color = if label > 0.0 {
            &RGBColor(215, 48, 39)
        } else {
            &RGBColor(69, 117, 180)
        };
        chart
            .draw_series(std::iter::once(Circle::new(
                (point[0], point[1]),
                5,
                ShapeStyle::from(color).filled().stroke_width(1),
            )))
            .unwrap();
    }

    root.present().unwrap();
    println!("Decision boundary plot saved as 'decision_boundary.png'");
}
