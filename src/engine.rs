use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub struct Value {
    pub data: RefCell<f64>, // Wrapped in RefCell for mutability
    pub grad: RefCell<f64>,
    _prev: Vec<Rc<Value>>,
    _op: String,
    _backward: RefCell<Option<Box<dyn Fn()>>>,
}

impl Value {
    pub fn new(data: f64) -> Rc<Self> {
        Rc::new(Value {
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            _prev: Vec::new(),
            _op: String::new(),
            _backward: RefCell::new(None),
        })
    }

    pub fn from(data: f64, children: Vec<Rc<Value>>, op: &str) -> Rc<Self> {
        Rc::new(Value {
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            _prev: children,
            _op: op.to_string(),
            _backward: RefCell::new(None),
        })
    }

    pub fn add(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Self> {
        let out_data = *self.data.borrow() + *other.data.borrow();
        let out = Value::from(out_data, vec![self.clone(), other.clone()], "+");
        let self_rc = self.clone();
        let other_rc = other.clone();
        let out_clone = out.clone();
        *out._backward.borrow_mut() = Some(Box::new(move || {
            *self_rc.grad.borrow_mut() += *out_clone.grad.borrow();
            *other_rc.grad.borrow_mut() += *out_clone.grad.borrow();
        }));
        out
    }

    pub fn mul(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Self> {
        let out_data = *self.data.borrow() * *other.data.borrow();
        let out = Value::from(out_data, vec![self.clone(), other.clone()], "*");
        let self_rc = self.clone();
        let other_rc = other.clone();
        let out_clone = out.clone();
        *out._backward.borrow_mut() = Some(Box::new(move || {
            *self_rc.grad.borrow_mut() += *other_rc.data.borrow() * *out_clone.grad.borrow();
            *other_rc.grad.borrow_mut() += *self_rc.data.borrow() * *out_clone.grad.borrow();
        }));
        out
    }

    pub fn pow(self: &Rc<Self>, exponent: f64) -> Rc<Self> {
        let out_data = self.data.borrow().powf(exponent);
        let out = Value::from(out_data, vec![self.clone()], &format!("**{}", exponent));
        let self_rc = self.clone();
        let out_clone = out.clone();
        *out._backward.borrow_mut() = Some(Box::new(move || {
            *self_rc.grad.borrow_mut() +=
                (exponent * self_rc.data.borrow().powf(exponent - 1.0)) * *out_clone.grad.borrow();
        }));
        out
    }

    pub fn relu(self: &Rc<Self>) -> Rc<Self> {
        let out_data = if *self.data.borrow() < 0.0 {
            0.0
        } else {
            *self.data.borrow()
        };
        let out = Value::from(out_data, vec![self.clone()], "ReLU");
        let self_rc = self.clone();
        let out_clone = out.clone();
        *out._backward.borrow_mut() = Some(Box::new(move || {
            *self_rc.grad.borrow_mut() += if *out_clone.data.borrow() > 0.0 {
                *out_clone.grad.borrow()
            } else {
                0.0
            };
        }));
        out
    }

    pub fn sub(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Self> {
        let neg_other = Value::new(-1.0).mul(other);
        self.add(&neg_other)
    }

    pub fn div(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Self> {
        let inv_other = other.pow(-1.0);
        self.mul(&inv_other)
    }

    pub fn backward(self: &Rc<Self>) {
        let mut topo: Vec<Rc<Value>> = Vec::new();
        let mut visited: HashSet<*const Value> = HashSet::new();

        fn build_topo(
            v: &Rc<Value>,
            visited: &mut HashSet<*const Value>,
            topo: &mut Vec<Rc<Value>>,
        ) {
            let ptr = Rc::as_ptr(v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v._prev {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);
        *self.grad.borrow_mut() = 1.0;
        for v in topo.iter().rev() {
            if let Some(backward) = v._backward.borrow().as_ref() {
                backward();
            }
        }
    }
}

// Manual Debug implementation to avoid issues with dyn Fn()
impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data.borrow())
            .field("grad", &self.grad.borrow())
            .field("_op", &self._op)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanity_check() {
        // Equivalent to Python: x = Value(-4.0); z = 2 * x + 2 + x;
        let x = Value::new(-4.0);
        let two = Value::new(2.0);
        let z = two.mul(&x).add(&two).add(&x); // 2 * x + 2 + x = 3 * x + 2 = -10
        let q = z.relu().add(&z.mul(&x)); // z.relu() (since z=-10, relu=0) + z * x = 0 + (-10)*(-4) = 40
        let h = z.mul(&z).relu(); // (z * z).relu() = (-10)^2 = 100
        let y = h.add(&q).add(&q.mul(&x)); // h + q + q * x = 100 + 40 + 40*(-4) = 140 - 160 = -20
        y.backward();

        let expected_y_data = -20.0; // Forward pass result for y when x = -4.0
        let expected_x_grad = 46.0; // Backward pass gradient for x

        // Check forward pass
        assert_eq!(
            *y.data.borrow(),
            expected_y_data,
            "Forward pass result mismatch in sanity check"
        );

        // Check backward pass
        assert_eq!(
            *x.grad.borrow(),
            expected_x_grad,
            "Backward pass gradient mismatch for x in sanity check"
        );
    }

    #[test]
    fn test_more_ops() {
        // Equivalent to Python: a = Value(-4.0); b = Value(2.0);
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let mut c = a.add(&b); // c = a + b = -2
        let mut d = a.mul(&b).add(&b.pow(3.0)); // d = a * b + b**3 = (-4)*2 + 8 = -8 + 8 = 0
        c = c.add(&c).add(&Value::new(1.0)); // c += c + 1 = (-2)*2 + 1 = -4 + 1 = -3
        let neg_a = Value::new(-1.0).mul(&a);
        c = c.add(&Value::new(1.0)).add(&c).add(&neg_a); // c += 1 + c + (-a) = -3 + 1 + (-3) + 4 = -1
        d = d.add(&d.mul(&Value::new(2.0))).add(&b.add(&a).relu()); // d += d * 2 + (b + a).relu() = 0 + 0 + (-2).relu() = 0
        d = d.add(&Value::new(3.0).mul(&d)).add(&b.sub(&a).relu()); // d += 3 * d + (b - a).relu() = 0 + 0 + (6).relu() = 6
        let e = c.sub(&d); // e = c - d = -1 - 6 = -7
        let f = e.pow(2.0); // f = e**2 = 49
        let g = f.div(&Value::new(2.0)); // g = f / 2.0 = 24.5
        let g_final = g.add(&Value::new(10.0).div(&f)); // g += 10.0 / f = 24.5 + 10/49 ≈ 24.704081632653057
        g_final.backward();

        // Expected values from PyTorch
        let tol = 1e-6;
        let expected_g_data = 24.70408163265306; // Forward pass result for g
        let expected_a_grad = 138.83381924198252; // Backward pass gradient for a
        let expected_b_grad = 645.5772594752186; // Backward pass gradient for b

        // Check forward pass
        assert!(
            (*g_final.data.borrow() - expected_g_data).abs() < tol,
            "Forward pass result mismatch in more ops"
        );

        // Check backward pass
        assert!(
            (*a.grad.borrow() - expected_a_grad).abs() < tol,
            "Backward pass gradient mismatch for a in more ops"
        );
        assert!(
            (*b.grad.borrow() - expected_b_grad).abs() < tol,
            "Backward pass gradient mismatch for b in more ops"
        );
    }
}
