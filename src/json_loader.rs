use crate::types::{Activation, LayerType, LossFunction, Optimizer, WeightLayout};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct TrainingJson {
    pub epochs: usize,
    pub final_loss: f64,
    pub loss_function: LossFunction,
    pub optimizer: Optimizer,
    pub learning_rate: f32,
}

#[derive(Deserialize, Debug)]
pub struct ModelJson {
    name: String,
    sample_rate: f32,
    num_parameters: usize,
    in_shape: usize,
    out_size: usize,
    weight_layout: WeightLayout,
    training: TrainingJson,
    pub layers: Vec<LayerJson>,
}

#[derive(Deserialize, Debug)]
pub struct LayerJson {
    #[serde(rename = "type")]
    pub layer_type: LayerType,
    pub in_size: usize,
    pub out_size: usize,
    pub activation: Activation,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl ModelJson {
    pub fn new_from_file(json_file_path: &str) -> Self {
        let json_str = std::fs::read_to_string(json_file_path).expect("Failed to read file");
        serde_json::from_str(&json_str).expect("Failed to parse JSON")
    }

    pub fn new_from_str(json_str: &str) -> Self {
        serde_json::from_str(json_str).expect("Failed to parse JSON")
    }

    pub fn weights(&self, index: usize) -> Vec<Vec<f32>> {
        self.layers[index].weights.clone()
    }

    pub fn bias(&self, index: usize) -> Vec<f32> {
        self.layers[index].bias.clone()
    }

    pub fn activation(&self, index: usize) -> Activation {
        self.layers[index].activation
    }

    pub fn count_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.bias.len())
            .sum()
    }

    pub fn expected_parameters(&self) -> usize {
        self.num_parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_json() {
        let path = "/Users/ursbollhalder/Desktop/Plugins_2026/glod/src/torchline/models/model.json";

        let model = ModelJson::new_from_file(path);
        println!("Number of Layers: {:?}", model.layers.len());

        println!("Bias Layer 2: {:?}", model.bias(1));
        println!("Weights Layer 2: {:?}", model.weights(1));
        println!("Activation Layer 2: {:?}", model.activation(1));

        println!("Num parameters: {:?}", model.num_parameters);
    }
}
