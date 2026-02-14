mod gain_table;
mod model_legacy;

use crate::model_legacy::SaturationModel;

mod model;
use model::Model;

mod json_loader;
use json_loader::{LayerJson, ModelJson};

mod envelope_follower;
use envelope_follower::EnvelopeFollower;

mod types;
use types::{Activation, LayerType};

pub mod neural_engine;
pub use neural_engine::NeuralEngine;
