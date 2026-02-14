use crate::model_legacy::SaturationModel;

pub struct GainTable {
    points: Vec<(f32, f32)>, // (drive, net_gain) sorted by drive
}

impl GainTable {
    pub fn new(points: Vec<(f32, f32)>) -> Self {
        Self { points }
    }

    pub fn lookup(&self, drive: f32) -> f32 {
        let pts = &self.points;
        if drive <= pts[0].0 {
            return pts[0].1;
        }
        if drive >= pts.last().unwrap().0 {
            return pts.last().unwrap().1;
        }

        // Find surrounding points
        let i = pts.iter().position(|p| p.0 > drive).unwrap();
        let (d0, g0) = pts[i - 1];
        let (d1, g1) = pts[i];
        let t = (drive - d0) / (d1 - d0);
        g0 + t * (g1 - g0)
    }

    /// Measures gain at each drive level, then returns pre-computed compensation factors.
    /// Compensation is normalized so that at `ref_drive` the factor is 1.0 (no change).
    /// At higher drives the factor < 1.0, taming volume while preserving harmonic character.
    pub fn measure_gain_curve(
        model: &mut SaturationModel,
        drives: &[f32],
        ref_drive: f32,
    ) -> Vec<(f32, f32)> {
        use keystone::signal::Signal;

        let test_len = 1024;
        let num_trials = 20;

        // First, measure raw gain at each drive
        let raw_gains: Vec<(f32, f32)> = drives
            .iter()
            .map(|&d| {
                let mut gain_sum = 0.0;

                for _trial in 0..num_trials {
                    let test_signal = Signal::rand_sig(test_len, (-0.2, 0.2));
                    let original_input_rms = Self::rms(test_signal.get());

                    model.reset();
                    let mut buf: Vec<f32> = test_signal.get().to_vec();
                    for s in buf.iter_mut() {
                        *s = model.process_sample(*s * d);
                    }
                    let model_output_rms = Self::rms(&buf);
                    gain_sum += model_output_rms / original_input_rms;
                }

                let avg_gain = gain_sum / num_trials as f32;
                (d, avg_gain)
            })
            .collect();

        // Find gain at reference drive by interpolation
        let ref_gain = Self::interp(&raw_gains, ref_drive);

        // Store compensation = ref_gain / gain_at_drive
        // At ref_drive this is 1.0; at higher drives this attenuates
        raw_gains
            .into_iter()
            .map(|(d, g)| (d, ref_gain / (g + 1e-7)))
            .collect()
    }

    fn interp(points: &[(f32, f32)], x: f32) -> f32 {
        if x <= points[0].0 {
            return points[0].1;
        }
        if x >= points.last().unwrap().0 {
            return points.last().unwrap().1;
        }
        let i = points.iter().position(|p| p.0 > x).unwrap();
        let (x0, y0) = points[i - 1];
        let (x1, y1) = points[i];
        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }

    fn rms(slice: &[f32]) -> f32 {
        (slice.iter().map(|x| x * x).sum::<f32>() / slice.len() as f32).sqrt()
    }
}
