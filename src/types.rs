use granite::Vector;
use serde::Deserialize;

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[serde(alias = "")]
    None,
    ReLU,
    Tanh,
}

impl Activation {
    pub fn get(&self) -> &'static str {
        match self {
            Activation::None => "none",
            Activation::ReLU => "relu",
            Activation::Tanh => "tanh",
        }
    }

    pub fn process<const N: usize>(&self, mut logits: Vector<N>) -> Vector<N> {
        match self {
            Activation::None => logits,
            Activation::ReLU => {
                for i in 0..N {
                    if logits[i] < 0.0 {
                        logits[i] = 0.0;
                    }
                }
                logits
            }
            Activation::Tanh => {
                for i in 0..N {
                    logits[i] = logits[i].tanh();
                }
                logits
            }
        }
    }
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum LayerType {
    Dense,
    Convolutional,
    GRU,
    LSTM,
}

impl LayerType {
    pub fn get(&self) -> &'static str {
        match self {
            LayerType::Dense => "dense",
            LayerType::Convolutional => "convolutional",
            LayerType::GRU => "gru",
            LayerType::LSTM => "lstm",
        }
    }
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum WeightLayout {
    InOut,
}

impl WeightLayout {
    pub fn get(&self) -> &'static str {
        match self {
            WeightLayout::InOut => "in_out",
        }
    }
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum LossFunction {
    #[serde(rename = "mse")]
    MeanSquaredError,
    CrossEntropy,
}

impl LossFunction {
    pub fn get(&self) -> &'static str {
        match self {
            LossFunction::MeanSquaredError => "mse",
            LossFunction::CrossEntropy => "cross_entropy",
        }
    }
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    Adam,
}

impl Optimizer {
    pub fn get(&self) -> &'static str {
        match self {
            Optimizer::Adam => "adam",
        }
    }
}
