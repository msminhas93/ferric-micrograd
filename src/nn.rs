use crate::engine::Value;
use rand::Rng;
use std::rc::Rc;

pub trait Module {
    fn zero_grad(&self);
    fn parameters(&self) -> Vec<Rc<Value>>;
}

pub struct Neuron {
    w: Vec<Rc<Value>>,
    b: Rc<Value>,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.random_range(-1.0..1.0)))
            .collect();
        let b = Value::new(0.0);
        Neuron { w, b, nonlin }
    }

    pub fn call(&self, x: &[Rc<Value>]) -> Rc<Value> {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = wi.mul(xi).add(&act);
        }
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn zero_grad(&self) {
        for p in self.parameters() {
            *p.grad.borrow_mut() = 0.0;
        }
    }

    fn parameters(&self) -> Vec<Rc<Value>> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin)).collect();
        Layer { neurons }
    }

    pub fn call(&self, x: &[Rc<Value>]) -> Vec<Rc<Value>> {
        let out: Vec<_> = self.neurons.iter().map(|n| n.call(x)).collect();
        out
    }
}

impl Module for Layer {
    fn zero_grad(&self) {
        for n in &self.neurons {
            n.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Rc<Value>> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut sz = vec![nin];
        sz.extend(nouts.iter());
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sz[i], sz[i + 1], i != nouts.len() - 1))
            .collect();
        MLP { layers }
    }

    pub fn call(&self, mut x: Vec<Rc<Value>>) -> Vec<Rc<Value>> {
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
}

impl Module for MLP {
    fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Rc<Value>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
