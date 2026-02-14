use crate::envelope_follower::EnvelopeFollower;
use crate::model::Model;

pub struct NeuralEngine {
    model: Model,
    envelope_l: EnvelopeFollower,
    envelope_r: EnvelopeFollower,
    latency: f32,
}

const MODEL_JSON: &str = include_str!("../models/model.json");

impl NeuralEngine {
    pub fn new() -> Self {
        Self {
            model: Model::new_from_json(MODEL_JSON),
            envelope_l: EnvelopeFollower::new(0.995),
            envelope_r: EnvelopeFollower::new(0.995),
            latency: 0.0,
        }
    }

    pub fn initialize(&mut self) -> bool {
        let weights: Vec<f32> =
            serde_json::from_str(include_str!("../models/weights_causal.json")).unwrap();
        // let consumed = self.model_l.load_weights(&weights);
        // assert_eq!(consumed, 2065);

        // self.model_r.load_weights(&weights);

        // // Quick test
        // self.model_l.reset();
        // self.model_r.reset();
        // let out = self.model_l.process_sample(0.8);
        // println!("Model test: 0.8 -> {out}");
        true
    }

    pub fn process(&mut self, buffer: &mut [&mut [f32]], drive: f32) {
        // for (ch_idx, channel) in buffer.iter_mut().enumerate() {
        //     let (model, envelope) = if ch_idx == 0 {
        //         (&mut self.model_l, &mut self.envelope_l)
        //     } else {
        //         (&mut self.model_r, &mut self.envelope_r)
        //     };

        //     for sample in channel.iter_mut() {
        //         let input = *sample;
        //         let saturated = model.process(input * drive);
        //         let compensation = envelope.process(input, saturated);
        //         *sample = saturated * compensation;
        //     }
        // }
        for channel in buffer.iter_mut() {
            for sample in channel.iter_mut() {
                let input = *sample;
                *sample = input * drive;
            }
        }

        self.model.process(buffer);
    }

    pub fn measure_latency(&mut self) -> usize {
        let latency = self.model.measure_latency();
        println!("Neural Engine has a latency of {:?} samples.", latency);
        latency
    }
}

#[cfg(test)]
mod tests {
    use keystone::signal::Signal;

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
            // engine.model_l.reset();
            // engine.model_r.reset();

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

    #[test]
    fn test_latency() {
        let mut engine = NeuralEngine::new();
        // engine.initialize();
        let latency = engine.measure_latency();
        println!("Neural Engine has a latency of {:?} samples.", latency);
    }
}
