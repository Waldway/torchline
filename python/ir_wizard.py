#!/usr/bin/env python3
"""
IR Wizard — Analyzes audio files and extracts meaningful segments
as stereo impulse responses for creative convolution-based sound design.

Usage:
    python ir_wizard.py <input_folder> [--output <output_folder>] [--min-length 0.05] [--max-length 5.0]

Dependencies:
    pip install numpy scipy librosa soundfile
"""

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class IRSegment:
    """A detected segment suitable for use as an impulse response."""

    start_sample: int
    end_sample: int
    category: str  # "transient", "resonance", "texture"
    score: float  # quality/interest score 0-1
    description: str


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def safe_n_fft(y: np.ndarray, desired: int = 1024) -> int:
    """Return an n_fft that fits the signal length."""
    return min(desired, len(y))


def analyze_spectral_complexity(y: np.ndarray, sr: int) -> float:
    """Rate how spectrally interesting a segment is (0-1)."""
    if len(y) < 512:
        return 0.0
    n_fft = safe_n_fft(y, 1024)
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=S))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)) / (
        sr / 2
    )
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)) / (
        sr / 2
    )
    complexity = 1.0 - abs(spectral_flatness - 0.3) * 2
    complexity = np.clip(
        complexity + spectral_centroid * 0.3 + spectral_bandwidth * 0.2, 0, 1
    )
    return float(complexity)


def compute_decay_rate(y: np.ndarray) -> float:
    """Estimate how cleanly a segment decays (higher = better IR candidate)."""
    env = np.abs(y)
    env_smooth = uniform_filter1d(env, size=min(256, len(env) // 2 + 1))
    if env_smooth[0] < 1e-8:
        return 0.0
    quarter = len(env_smooth) // 4
    if quarter == 0:
        return 0.0
    ratios = []
    for i in range(1, 4):
        seg = env_smooth[i * quarter : (i + 1) * quarter]
        prev = env_smooth[(i - 1) * quarter : i * quarter]
        if np.mean(prev) > 1e-8:
            ratios.append(np.mean(seg) / np.mean(prev))
    if not ratios:
        return 0.0
    avg_ratio = np.mean(ratios)
    if avg_ratio >= 1.0:
        return 0.1
    return float(np.clip(1.0 - abs(avg_ratio - 0.5), 0, 1))


# ---------------------------------------------------------------------------
# Segment detectors
# ---------------------------------------------------------------------------


def detect_transients(
    y: np.ndarray, sr: int, min_len: int, max_len: int
) -> list[IRSegment]:
    """Find transient events (hits, plucks, impacts)."""
    segments = []
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    peaks, _ = find_peaks(
        onset_env, height=np.percentile(onset_env, 85), distance=sr // 256
    )

    for peak in peaks:
        start = max(0, peak * 256 - int(0.005 * sr))
        env = np.abs(y[start : min(start + max_len, len(y))])
        if len(env) == 0:
            continue
        env_smooth = uniform_filter1d(env, size=min(128, len(env) // 2 + 1))
        peak_val = np.max(env_smooth[: min(1024, len(env_smooth))])
        if peak_val < 1e-6:
            continue
        threshold = peak_val * 0.05
        below = np.where(env_smooth < threshold)[0]
        end_offset = below[0] if len(below) > 0 else len(env_smooth)
        end_offset = max(end_offset, min_len)
        end_offset = min(end_offset, max_len)
        end = start + end_offset
        if end > len(y):
            continue

        segment = y[start:end]
        score = (
            analyze_spectral_complexity(segment, sr) * 0.5
            + compute_decay_rate(segment) * 0.5
        )
        segments.append(
            IRSegment(
                start, end, "transient", score, "Impact/transient with natural decay"
            )
        )

    return segments


def detect_resonances(
    y: np.ndarray, sr: int, min_len: int, max_len: int
) -> list[IRSegment]:
    """Find resonant/ringing segments."""
    segments = []
    hop = 512
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    chroma_energy = np.max(chroma, axis=0)
    chroma_smooth = uniform_filter1d(chroma_energy, size=20)
    threshold = np.percentile(chroma_smooth, 70)
    active = chroma_smooth > threshold

    in_region = False
    region_start = 0
    for i, val in enumerate(active):
        if val and not in_region:
            region_start = i
            in_region = True
        elif not val and in_region:
            start_sample = region_start * hop
            end_sample = min(i * hop, len(y))
            length = end_sample - start_sample
            if min_len <= length <= max_len:
                segment = y[start_sample:end_sample]
                decay = compute_decay_rate(segment)
                complexity = analyze_spectral_complexity(segment, sr)
                score = decay * 0.6 + complexity * 0.4
                if score > 0.3:
                    segments.append(
                        IRSegment(
                            start_sample,
                            end_sample,
                            "resonance",
                            score,
                            "Resonant/tonal segment with sustained character",
                        )
                    )
            in_region = False

    return segments


def detect_textures(
    y: np.ndarray, sr: int, min_len: int, max_len: int
) -> list[IRSegment]:
    """Find interesting textural segments."""
    segments = []
    frame_length = max_len
    step = frame_length // 2

    for start in range(0, len(y) - frame_length, step):
        segment = y[start : start + frame_length]
        rms = np.sqrt(np.mean(segment**2))
        if rms < 1e-5:
            continue

        complexity = analyze_spectral_complexity(segment, sr)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=segment)))

        if 0.1 < flatness < 0.8 and complexity > 0.4:
            score = complexity * 0.7 + flatness * 0.3
            segments.append(
                IRSegment(
                    start,
                    start + frame_length,
                    "texture",
                    score,
                    "Textural/ambient segment for creative convolution",
                )
            )

    return segments


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def generate_ir_name(
    category: str, spectral_centroid: float, flatness: float, duration_ms: float
) -> str:
    """Generate an evocative two-word name based on acoustic properties."""

    adjectives = {
        (0.0, 0.15): ["dark", "deep", "murky", "hollow", "submerged"],
        (0.15, 0.3): ["warm", "mellow", "wooden", "amber", "dusky"],
        (0.3, 0.5): ["clear", "copper", "focused", "analog", "dense"],
        (0.5, 0.7): ["bright", "crisp", "glassy", "silver", "electric"],
        (0.7, 1.0): ["airy", "shimmering", "crystalline", "icy", "spectral"],
    }

    nouns = {
        (0.0, 0.15): ["bell", "chime", "ring", "drone", "vessel"],
        (0.15, 0.3): ["bloom", "shimmer", "tone", "resonance", "harmonic"],
        (0.3, 0.5): ["whisper", "breath", "flutter", "ripple", "grain"],
        (0.5, 0.7): ["wash", "static", "sand", "haze", "cloud"],
        (0.7, 1.0): ["grit", "dust", "erosion", "storm", "noise"],
    }

    seed = f"{spectral_centroid:.4f}_{flatness:.4f}_{duration_ms:.1f}"

    def pick(word_map, value, salt):
        for (lo, hi), words in word_map.items():
            if lo <= value < hi:
                h = int(hashlib.md5((seed + salt).encode()).hexdigest(), 16)
                return words[h % len(words)]
        return list(list(word_map.values())[-1])[0]

    adj = pick(adjectives, spectral_centroid, "adj")
    noun = pick(nouns, flatness, "noun")
    return f"{adj}_{noun}"


# ---------------------------------------------------------------------------
# IR processing (fade, gain compensation, stereo)
# ---------------------------------------------------------------------------


def normalize_and_fade(
    y: np.ndarray,
    sr: int = 44100,
    category: str = "transient",
    fade_in_samples: int = 64,
) -> np.ndarray:
    """Normalize for energy-neutral convolution, apply category-aware fades."""

    # --- Fade-in (short, anti-click) ---
    fade_in = min(fade_in_samples, len(y) // 4)
    y[:fade_in] *= np.linspace(0, 1, fade_in)

    # --- Fade-out (proportional to length and category) ---
    fade_ratios = {
        "transient": 0.15,
        "resonance": 0.35,
        "texture": 0.40,
    }
    ratio = fade_ratios.get(category, 0.25)
    fade_out_samples = int(len(y) * ratio)
    fade_out_samples = max(fade_out_samples, min(int(0.01 * sr), len(y) // 4))

    if category == "transient":
        curve = np.linspace(1, 0, fade_out_samples)
    elif category == "resonance":
        curve = np.exp(-5 * np.linspace(0, 1, fade_out_samples))
    else:
        curve = np.cos(np.linspace(0, np.pi / 2, fade_out_samples)) ** 2

    y[-fade_out_samples:] *= curve

    # --- Energy normalization (L2 norm) for gain-neutral convolution ---
    l2_norm = np.sqrt(np.sum(y**2))
    if l2_norm > 1e-8:
        y = y / l2_norm

    # Safety limiter
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak * 0.99

    return y


def create_stereo_ir(
    mono_ir: np.ndarray, sr: int, category: str = "transient"
) -> np.ndarray:
    """Create a stereo IR from mono with natural decorrelation. Returns (2, n)."""
    n = len(mono_ir)

    # L/R time offset for natural stereo spread
    offsets = {"transient": 0.0002, "resonance": 0.0006, "texture": 0.0008}
    offset_samples = int(offsets.get(category, 0.0004) * sr)

    # Decorrelation noise shaped by IR envelope
    envelope = uniform_filter1d(np.abs(mono_ir), size=min(512, n // 2 + 1))
    rng = np.random.default_rng(seed=hash(mono_ir[: min(64, n)].tobytes()) & 0xFFFFFFFF)
    noise = uniform_filter1d(rng.normal(0, 1, n), size=32)
    noise *= envelope

    decorr_amount = {"transient": 0.03, "resonance": 0.08, "texture": 0.12}
    amount = decorr_amount.get(category, 0.05)

    left = mono_ir.copy()
    right = np.zeros(n)
    if 0 < offset_samples < n:
        right[offset_samples:] = mono_ir[:-offset_samples]
    else:
        right = mono_ir.copy()

    left += noise * amount
    right -= noise * amount

    stereo = np.stack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 1.0:
        stereo /= peak * 1.01

    return stereo


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_segments(
    segments: list[IRSegment], sr: int, min_gap_sec: float = 0.1
) -> list[IRSegment]:
    """Remove overlapping segments, keeping highest scoring ones."""
    if not segments:
        return []
    segments.sort(key=lambda s: s.score, reverse=True)
    min_gap = int(min_gap_sec * sr)
    kept = []
    for seg in segments:
        overlap = False
        for k in kept:
            if not (
                seg.end_sample + min_gap < k.start_sample
                or seg.start_sample > k.end_sample + min_gap
            ):
                overlap = True
                break
        if not overlap:
            kept.append(seg)
    return kept


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_file(
    filepath: Path,
    output_dir: Path,
    min_length: float,
    max_length: float,
    max_irs_per_file: int = 10,
):
    """Analyze a single audio file and extract stereo IRs."""
    print(f"\n📂 Analyzing: {filepath.name}")
    try:
        y, sr = librosa.load(str(filepath), sr=None, mono=True)
    except Exception as e:
        print(f"  ⚠️  Could not load: {e}")
        return 0

    min_samples = int(min_length * sr)
    max_samples = int(max_length * sr)

    print(f"  Duration: {len(y) / sr:.1f}s | SR: {sr} Hz")

    # Run all detectors
    all_segments = []
    all_segments.extend(detect_transients(y, sr, min_samples, max_samples))
    all_segments.extend(detect_resonances(y, sr, min_samples, max_samples))
    all_segments.extend(detect_textures(y, sr, min_samples, max_samples))

    # Deduplicate and take the best
    segments = deduplicate_segments(all_segments, sr)
    segments = segments[:max_irs_per_file]

    if not segments:
        print("  No suitable IR segments found.")
        return 0

    # Export
    stem = filepath.stem
    count = 0
    for i, seg in enumerate(segments):
        ir = y[seg.start_sample : seg.end_sample].copy()
        ir = normalize_and_fade(ir, sr=sr, category=seg.category)

        duration_ms = len(ir) / sr * 1000

        # Compute acoustic features for naming
        n_fft = safe_n_fft(ir, 1024)
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=ir, sr=sr, n_fft=n_fft))
        ) / (sr / 2)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=ir, n_fft=n_fft)))
        ir_name = generate_ir_name(seg.category, centroid, flatness, duration_ms)
        filename = f"{stem}_{ir_name}.wav"

        # Create stereo IR
        ir_stereo = create_stereo_ir(ir, sr, category=seg.category)

        out_path = output_dir / filename
        sf.write(str(out_path), ir_stereo.T, sr, subtype="FLOAT")
        count += 1
        print(
            f"  ✅ {filename} | {seg.category} | score={seg.score:.2f} | {duration_ms:.0f}ms | stereo"
        )

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract creative impulse responses from audio files"
    )
    parser.add_argument("input_folder", type=str, help="Folder containing audio files")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output folder (default: <input>/IRs)",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=0.02,
        help="Minimum IR length in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=5.0,
        help="Maximum IR length in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=10,
        help="Max IRs to extract per file (default: 10)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        print(f"❌ Not a valid directory: {input_dir}")
        return

    output_dir = Path(args.output) if args.output else input_dir / "IRs"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".aiff", ".aif", ".flac", ".ogg", ".mp3", ".m4a"}
    files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in audio_extensions]
    )

    if not files:
        print(f"❌ No audio files found in {input_dir}")
        return

    print(f"🎛️  IR Wizard — Creative Impulse Response Extractor")
    print(f"   Input:  {input_dir} ({len(files)} files)")
    print(f"   Output: {output_dir}")
    print(f"   IR length: {args.min_length}s – {args.max_length}s")

    total = 0
    for f in files:
        total += process_file(
            f, output_dir, args.min_length, args.max_length, args.max_per_file
        )

    print(f"\n🎉 Done! Extracted {total} stereo impulse responses → {output_dir}")


if __name__ == "__main__":
    main()
