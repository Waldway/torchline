/// Per-channel envelope follower for real-time gain compensation.
pub struct EnvelopeFollower {
    input_env: f32,
    output_env: f32,
    coeff: f32, // smoothing coefficient (0..1), higher = slower
}

impl EnvelopeFollower {
    pub fn new(coeff: f32) -> Self {
        Self {
            input_env: 0.0,
            output_env: 0.0,
            coeff,
        }
    }

    /// Track input and output envelopes, return compensation gain.
    pub fn process(&mut self, input: f32, output: f32) -> f32 {
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
