use crate::json_loader::{LayerJson, ModelJson};
use crate::types::Activation;
use granite::{Matrix, Vector};

// DENSE LAYER
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: Matrix<OUT, IN>,
    bias: Vector<OUT>,
    activation: Activation,
}

// Here not only generate a deterministically sized Matrix from vectors, but we also transpose from
// our Torch matrix notation to regular linalg matrices: [IN][OUT] -> [OUT][IN]
fn vecs_to_matrix<const OUT: usize, const IN: usize>(w: &Vec<Vec<f32>>) -> Matrix<OUT, IN> {
    let mut m = Matrix::<OUT, IN>::new();
    for i in 0..IN {
        for o in 0..OUT {
            m[o][i] = w[i][o];
        }
    }
    m
}

fn vec_to_vector<const N: usize>(v: &Vec<f32>) -> Vector<N> {
    let mut vector = Vector::<N>::new();
    for i in 0..N {
        vector[i] = v[i];
    }
    vector
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn new(weights: Matrix<OUT, IN>, bias: Vector<OUT>, activation: Activation) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(&self, input: Vector<IN>) -> Vector<OUT> {
        let logit = self.weights * input + self.bias;
        self.activation.process(logit)
    }
}

pub struct Model {
    model_json: ModelJson,
    layer_0: DenseLayer<2, 16>,
    layer_1: DenseLayer<16, 16>,
    layer_2: DenseLayer<16, 2>,
    model_latency: usize,
}

impl Model {
    pub fn new_from_json(json_str: &str) -> Self {
        let model_json = ModelJson::new_from_str(json_str);
        let counted = model_json.count_parameters();
        let expected = model_json.expected_parameters();
        println!(
            "Loaded model: {} parameters (expected {})",
            counted, expected
        );
        assert_eq!(counted, expected, "Parameter count mismatch in model JSON");

        let layer_0 = DenseLayer::new(
            vecs_to_matrix::<16, 2>(&model_json.weights(0)),
            vec_to_vector::<16>(&model_json.bias(0)),
            model_json.activation(0),
        );

        let layer_1 = DenseLayer::new(
            vecs_to_matrix::<16, 16>(&model_json.weights(1)),
            vec_to_vector::<16>(&model_json.bias(1)),
            model_json.activation(1),
        );

        let layer_2 = DenseLayer::new(
            vecs_to_matrix::<2, 16>(&model_json.weights(2)),
            vec_to_vector::<2>(&model_json.bias(2)),
            model_json.activation(2),
        );

        let model_latency = 0;

        Self {
            model_json,
            layer_0,
            layer_1,
            layer_2,
            model_latency,
        }
    }

    pub fn forward(&self, input: Vector<2>) -> Vector<2> {
        let hidden_1 = self.layer_0.forward(input);
        let hidden_2 = self.layer_1.forward(hidden_1);
        self.layer_2.forward(hidden_2)
    }

    pub fn process(&self, buffer: &mut [&mut [f32]]) {
        let buffer_size = buffer[0].len();

        for s in 0..buffer_size {
            let input = Vector::from([buffer[0][s], buffer[1][s]]);
            let output = self.forward(input);

            buffer[0][s] = output[0];
            buffer[1][s] = output[1];
        }
    }

    pub fn measure_latency(&mut self) -> usize {
        let mut ch0 = [0.0f32; 512];
        let mut ch1 = [0.0f32; 512];
        ch0[64] = 1.0;
        ch1[64] = 1.0;

        let input = ch0; // copy before processing overwrites it
        let mut buffer: [&mut [f32]; 2] = [&mut ch0, &mut ch1];
        self.process(&mut buffer);

        let max_lag = 128;
        let (best_lag, _) = (0..max_lag)
            .map(|lag| {
                let corr: f32 = (lag..512).map(|i| input[i - lag] * buffer[0][i]).sum();
                (lag, corr)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.model_latency = best_lag;
        best_lag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency() {
        let json = include_str!("../models/model.json");
        let mut model = Model::new_from_json(json);
        let latency = model.measure_latency();
        println!("Latency: {}", latency);
        assert_eq!(latency, 0);
    }
}
