mod gain_table;
mod model;
use crate::model::SaturationModel;
use nih_plug::prelude::*;

/// Per-channel envelope follower for real-time gain compensation.
struct EnvelopeFollower {
    input_env: f32,
    output_env: f32,
    coeff: f32, // smoothing coefficient (0..1), higher = slower
}

impl EnvelopeFollower {
    fn new(coeff: f32) -> Self {
        Self {
            input_env: 0.0,
            output_env: 0.0,
            coeff,
        }
    }

    /// Track input and output envelopes, return compensation gain.
    fn process(&mut self, input: f32, output: f32) -> f32 {
        let c = self.coeff;
        self.input_env = c * self.input_env + (1.0 - c) * input.abs();
        self.output_env = c * self.output_env + (1.0 - c) * output.abs();

        if self.output_env > 1e-7 {
            self.input_env / self.output_env
        } else {
            1.0
        }
    }
}

pub struct NeuralEngine {
    model_l: SaturationModel,
    model_r: SaturationModel,
    envelope_l: EnvelopeFollower,
    envelope_r: EnvelopeFollower,
    latency: f32,
}

impl NeuralEngine {
    pub fn new() -> Self {
        Self {
            model_l: SaturationModel::new(),
            model_r: SaturationModel::new(),
            envelope_l: EnvelopeFollower::new(0.995),
            envelope_r: EnvelopeFollower::new(0.995),
            latency: 0.0,
        }
    }

    pub fn initialize(&mut self) -> bool {
        let weights: Vec<f32> =
            serde_json::from_str(include_str!("../models/weights_causal.json")).unwrap();
        let consumed = self.model_l.load_weights(&weights);
        assert_eq!(consumed, 2065);

        self.model_r.load_weights(&weights);

        // Quick test
        self.model_l.reset();
        self.model_r.reset();
        let out = self.model_l.process_sample(0.8);
        println!("Model test: 0.8 -> {out}");
        true
    }

    pub fn process(&mut self, buffer: &mut [&mut [f32]], drive: f32) {
        for (ch_idx, channel) in buffer.iter_mut().enumerate() {
            let (model, envelope) = if ch_idx == 0 {
                (&mut self.model_l, &mut self.envelope_l)
            } else {
                (&mut self.model_r, &mut self.envelope_r)
            };

            for sample in channel.iter_mut() {
                let input = *sample;
                let saturated = model.process_sample(input * drive);
                let compensation = envelope.process(input, saturated);
                *sample = saturated * compensation;
            }
        }
    }

    pub fn measure_latency(&mut self) -> usize {
        let latency = self.model_l.measure_latency();
        println!("Neural Engine has a latency of {:?} samples.", latency);
        latency
    }
}

#[cfg(test)]
mod tests {
    use lib_audio::signal::Signal;

    use super::*;

    fn rms(slice: &[f32]) -> f32 {
        (slice.iter().map(|x| x * x).sum::<f32>() / slice.len() as f32).sqrt()
    }

    #[test]
    fn test_gain() {
        let mut engine = NeuralEngine::new();
        engine.initialize();
        // Test across the full parameter range: 1.0 to 10.0
        let mut drive_values: [f32; 13] = [
            0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ];
        for (_i, drive) in drive_values.iter().enumerate() {
            engine.model_l.reset();
            engine.model_r.reset();

            let sig_l = Signal::rand_sig(1024, (-0.2, 0.2));
            let sig_r = Signal::rand_sig(1024, (-0.2, 0.2));
            let mut ch_l: Vec<f32> = sig_l.get().to_vec();
            let mut ch_r: Vec<f32> = sig_r.get().to_vec();
            let mut buffer: [&mut [f32]; 2] = [&mut ch_l, &mut ch_r];

            engine.process(&mut buffer, *drive);

            println!(
                "Drive: {:.1} | L clean: {:.4}  L proc: {:.4}  ratio: {:.2}x | R clean: {:.4}  R proc: {:.4}  ratio: {:.2}x",
                drive,
                sig_l.rms(),
                rms(buffer[0]),
                rms(buffer[0]) / sig_l.rms(),
                sig_r.rms(),
                rms(buffer[1]),
                rms(buffer[1]) / sig_r.rms(),
            );
        }
    }
}
