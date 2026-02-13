// ============================================================
// Real-time safe neural network layers — zero heap allocation
// All sizes known at compile time via const generics.
// ============================================================

use std::f32;

#[inline(always)]
fn tanh_fast(x: f32) -> f32 {
    x.tanh()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================
// Dense layer: y = W * x + b
// ============================================================

pub struct Dense<const IN: usize, const OUT: usize> {
    pub weights: [[f32; IN]; OUT],
    pub bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Dense<IN, OUT> {
    pub fn zeros() -> Self {
        Self {
            weights: [[0.0; IN]; OUT],
            bias: [0.0; OUT],
        }
    }

    #[inline]
    pub fn forward(&self, input: &[f32; IN], output: &mut [f32; OUT]) {
        for i in 0..OUT {
            let mut sum = self.bias[i];
            for j in 0..IN {
                sum += self.weights[i][j] * input[j];
            }
            output[i] = sum;
        }
    }
}

// ============================================================
// Conv1D (causal, no dilation) with ring buffer
// ============================================================

pub struct Conv1D<const IN_CH: usize, const OUT_CH: usize, const KERNEL: usize> {
    pub weights: [[[f32; KERNEL]; IN_CH]; OUT_CH],
    pub bias: [f32; OUT_CH],
    history: [[f32; IN_CH]; KERNEL],
    write_pos: usize,
}

impl<const IN_CH: usize, const OUT_CH: usize, const KERNEL: usize> Conv1D<IN_CH, OUT_CH, KERNEL> {
    pub fn zeros() -> Self {
        Self {
            weights: [[[0.0; KERNEL]; IN_CH]; OUT_CH],
            bias: [0.0; OUT_CH],
            history: [[0.0; IN_CH]; KERNEL],
            write_pos: 0,
        }
    }

    #[inline]
    pub fn forward(&mut self, input: &[f32; IN_CH], output: &mut [f32; OUT_CH]) {
        self.history[self.write_pos] = *input;
        self.write_pos = (self.write_pos + 1) % KERNEL;

        for out_ch in 0..OUT_CH {
            let mut sum = self.bias[out_ch];
            for k in 0..KERNEL {
                // PyTorch Conv1d is cross-correlation: weight[k=0] matches newest sample
                let hist_idx = (self.write_pos + KERNEL - 1 - k) % KERNEL;
                for in_ch in 0..IN_CH {
                    sum += self.weights[out_ch][in_ch][k] * self.history[hist_idx][in_ch];
                }
            }
            output[out_ch] = sum;
        }
    }

    pub fn reset(&mut self) {
        self.history = [[0.0; IN_CH]; KERNEL];
        self.write_pos = 0;
    }
}

// ============================================================
// GRU (single time-step, stateful)
// ============================================================

pub struct GRU<const IN: usize, const HIDDEN: usize> {
    pub w_z: [[f32; IN]; HIDDEN],
    pub w_r: [[f32; IN]; HIDDEN],
    pub w_n: [[f32; IN]; HIDDEN],
    pub u_z: [[f32; HIDDEN]; HIDDEN],
    pub u_r: [[f32; HIDDEN]; HIDDEN],
    pub u_n: [[f32; HIDDEN]; HIDDEN],
    pub b_iz: [f32; HIDDEN],
    pub b_ir: [f32; HIDDEN],
    pub b_in: [f32; HIDDEN],
    pub b_hz: [f32; HIDDEN],
    pub b_hr: [f32; HIDDEN],
    pub b_hn: [f32; HIDDEN],
    pub h: [f32; HIDDEN],
}

impl<const IN: usize, const HIDDEN: usize> GRU<IN, HIDDEN> {
    pub fn zeros() -> Self {
        Self {
            w_z: [[0.0; IN]; HIDDEN],
            w_r: [[0.0; IN]; HIDDEN],
            w_n: [[0.0; IN]; HIDDEN],
            u_z: [[0.0; HIDDEN]; HIDDEN],
            u_r: [[0.0; HIDDEN]; HIDDEN],
            u_n: [[0.0; HIDDEN]; HIDDEN],
            b_iz: [0.0; HIDDEN],
            b_ir: [0.0; HIDDEN],
            b_in: [0.0; HIDDEN],
            b_hz: [0.0; HIDDEN],
            b_hr: [0.0; HIDDEN],
            b_hn: [0.0; HIDDEN],
            h: [0.0; HIDDEN],
        }
    }

    #[inline]
    pub fn forward(&mut self, input: &[f32; IN], output: &mut [f32; HIDDEN]) {
        let mut z = [0.0f32; HIDDEN];
        let mut r = [0.0f32; HIDDEN];
        let mut n = [0.0f32; HIDDEN];

        for i in 0..HIDDEN {
            let mut zv = self.b_iz[i] + self.b_hz[i];
            for j in 0..IN {
                zv += self.w_z[i][j] * input[j];
            }
            for j in 0..HIDDEN {
                zv += self.u_z[i][j] * self.h[j];
            }
            z[i] = sigmoid(zv);

            let mut rv = self.b_ir[i] + self.b_hr[i];
            for j in 0..IN {
                rv += self.w_r[i][j] * input[j];
            }
            for j in 0..HIDDEN {
                rv += self.u_r[i][j] * self.h[j];
            }
            r[i] = sigmoid(rv);
        }

        for i in 0..HIDDEN {
            let mut nv_input = self.b_in[i];
            for j in 0..IN {
                nv_input += self.w_n[i][j] * input[j];
            }

            let mut nv_hidden = self.b_hn[i];
            for j in 0..HIDDEN {
                nv_hidden += self.u_n[i][j] * self.h[j];
            }

            n[i] = tanh_fast(nv_input + r[i] * nv_hidden);
        }

        for i in 0..HIDDEN {
            self.h[i] = (1.0 - z[i]) * n[i] + z[i] * self.h[i];
            output[i] = self.h[i];
        }
    }

    pub fn reset(&mut self) {
        self.h = [0.0; HIDDEN];
    }
}

// ============================================================
// Actual model: conv_pre(1,8,65) → conv_mid(8,8,17) → GRU(8,8) → Dense(8,1)
// ============================================================

pub struct SaturationModel {
    conv_pre: Conv1D<1, 8, 65>,
    conv_mid: Conv1D<8, 8, 17>,
    gru: GRU<8, 8>,
    output: Dense<8, 1>,
    model_latency: usize,
}

impl SaturationModel {
    pub fn new() -> Self {
        Self {
            conv_pre: Conv1D::zeros(),
            conv_mid: Conv1D::zeros(),
            gru: GRU::zeros(),
            output: Dense::zeros(),
            model_latency: 0,
        }
    }

    pub fn measure_latency(&mut self) -> usize {
        self.reset();

        // Feed a short burst of noise, compare input/output timing
        let len = 512;
        let mut input = vec![0.0f32; len];
        // Put a click at sample 64 (away from the edge)
        input[64] = 1.0;

        let mut output = vec![0.0f32; len];
        for i in 0..len {
            output[i] = self.process_sample(input[i]);
        }

        // Cross-correlate to find the delay
        let max_lag = 128;
        let mut best_lag = 0;
        let mut best_corr = 0.0f32;

        for lag in 0..max_lag {
            let mut corr = 0.0;
            for i in lag..len {
                corr += input[i - lag] * output[i];
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        self.model_latency = best_lag;
        best_lag
    }

    pub fn latency(&self) -> usize {
        self.model_latency
    }

    /// Process one sample. Zero allocations.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut conv1_out = [0.0f32; 8];
        self.conv_pre.forward(&[input], &mut conv1_out);
        for v in &mut conv1_out {
            *v = v.tanh();
        }

        let mut conv2_out = [0.0f32; 8];
        self.conv_mid.forward(&conv1_out, &mut conv2_out);
        for v in &mut conv2_out {
            *v = v.tanh();
        }

        let mut gru_out = [0.0f32; 8];
        self.gru.forward(&conv2_out, &mut gru_out);

        let mut out = [0.0f32; 1];
        self.output.forward(&gru_out, &mut out);

        out[0]
    }

    pub fn reset(&mut self) {
        self.conv_pre.reset();
        self.conv_mid.reset();
        self.gru.reset();
    }

    /// Load weights from flat f32 slice. Returns number of floats consumed.
    pub fn load_weights(&mut self, data: &[f32]) -> usize {
        let mut o = 0;

        // conv_pre: [8, 1, 65] + [8]
        for oc in 0..8 {
            for ic in 0..1 {
                for k in 0..65 {
                    self.conv_pre.weights[oc][ic][k] = data[o];
                    o += 1;
                }
            }
        }
        for i in 0..8 {
            self.conv_pre.bias[i] = data[o];
            o += 1;
        }

        // conv_mid: [8, 8, 17] + [8]
        for oc in 0..8 {
            for ic in 0..8 {
                for k in 0..17 {
                    self.conv_mid.weights[oc][ic][k] = data[o];
                    o += 1;
                }
            }
        }
        for i in 0..8 {
            self.conv_mid.bias[i] = data[o];
            o += 1;
        }

        // GRU: w_ih [r,z,n], w_hh [r,z,n], b_ih [r,z,n], b_hh [r,z,n]
        for i in 0..8 {
            for j in 0..8 {
                self.gru.w_r[i][j] = data[o];
                o += 1;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                self.gru.w_z[i][j] = data[o];
                o += 1;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                self.gru.w_n[i][j] = data[o];
                o += 1;
            }
        }

        for i in 0..8 {
            for j in 0..8 {
                self.gru.u_r[i][j] = data[o];
                o += 1;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                self.gru.u_z[i][j] = data[o];
                o += 1;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                self.gru.u_n[i][j] = data[o];
                o += 1;
            }
        }

        for i in 0..8 {
            self.gru.b_ir[i] = data[o];
            o += 1;
        }
        for i in 0..8 {
            self.gru.b_iz[i] = data[o];
            o += 1;
        }
        for i in 0..8 {
            self.gru.b_in[i] = data[o];
            o += 1;
        }

        for i in 0..8 {
            self.gru.b_hr[i] = data[o];
            o += 1;
        }
        for i in 0..8 {
            self.gru.b_hz[i] = data[o];
            o += 1;
        }
        for i in 0..8 {
            self.gru.b_hn[i] = data[o];
            o += 1;
        }

        // output: [1, 8] + [1]
        for i in 0..1 {
            for j in 0..8 {
                self.output.weights[i][j] = data[o];
                o += 1;
            }
        }
        for i in 0..1 {
            self.output.bias[i] = data[o];
            o += 1;
        }

        o // should be 2065
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency() {
        let mut model = SaturationModel::new();
        let weights: Vec<f32> =
            serde_json::from_str(include_str!("../models/weights_causal.json")).unwrap();

        model.load_weights(&weights);
        let latency = model.measure_latency();
        println!("Model latency: {}", latency);
    }

    #[test]
    fn test_polarity() {
        let mut model = SaturationModel::new();
        let weights: Vec<f32> =
            serde_json::from_str(include_str!("../models/weights_causal.json")).unwrap();
        model.load_weights(&weights);

        // Feed a positive DC-ish signal
        model.reset();
        for _ in 0..100 {
            model.process_sample(0.5);
        } // warm up
        let out = model.process_sample(0.5);
        println!("Input: 0.5, Output: {}", out);
        // If output is negative, the model inverts polarity
    }
}
