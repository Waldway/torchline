# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Neural Network Audio Saturation - Complete Workflow
# 
# This notebook trains a neural network to learn a saturation function (tanh)
# and applies it to real audio. All processing uses proper audio range: -1 to +1

# ## 1. Generate Audio Data (for Training)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa

# %%
clean_training_sample = './audio/African Talking Drum 05.wav'
saturated_training_sample = './audio/African Talking Drum 05_sat.wav'

clean_training_data, audio_sr = librosa.load(clean_training_sample, sr=None, mono=False)
sat_training_data, audio_sr = librosa.load(saturated_training_sample, sr=None, mono=False)

print("Clean Data")
display(Audio(data=clean_training_data, rate=audio_sr))

print("Saturated Data")
display(Audio(data=sat_training_data, rate=audio_sr))

# %%
from tqdm import tqdm

# ## 1. Prepare Windowed Training Data

print("Step 1: Preparing windowed training data...")

SAMPLE_RATE = 44100

clean_training_data = np.asarray(clean_training_data).flatten()
sat_training_data = np.asarray(sat_training_data).flatten()

MAX_SAMPLES = SAMPLE_RATE * 30
clean_training_data = clean_training_data[:MAX_SAMPLES]
sat_training_data = sat_training_data[:MAX_SAMPLES]

print(f"  Training data length: {len(clean_training_data)} samples "
      f"({len(clean_training_data) / SAMPLE_RATE * 1000:.1f}ms)")

CHUNK_SIZE = min(2048, len(clean_training_data) // 4)
HOP_SIZE = CHUNK_SIZE // 2

def window_audio(audio, chunk_size, hop_size):
    chunks = []
    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        chunks.append(audio[start:start + chunk_size])
    if not chunks:
        padded = np.zeros(chunk_size)
        padded[:len(audio)] = audio
        chunks.append(padded)
    return np.stack(chunks)

X_chunks = window_audio(clean_training_data, CHUNK_SIZE, HOP_SIZE)
y_chunks = window_audio(sat_training_data, CHUNK_SIZE, HOP_SIZE)

X_train = torch.FloatTensor(X_chunks).unsqueeze(1)
y_train = torch.FloatTensor(y_chunks).unsqueeze(1)

print(f"  Created {X_train.shape[0]} training chunks")
print(f"  Input shape: {X_train.shape}")

# ## 2. Model — Causal Convolutions

print("\nStep 2: Setting up Neural Network (causal convolutions)...")

class CausalConv1d(nn.Module):
    """Conv1d that only looks at past + current samples. No future leakage."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        # Left-pad with zeros so output length == input length
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class SaturationNet(nn.Module):
    def __init__(self, channels=8, kernel_pre=65, kernel_mid=17, gru_layers=1):
        super().__init__()
        self.conv_pre = CausalConv1d(1, channels, kernel_pre)
        self.conv_mid = CausalConv1d(channels, channels, kernel_mid)
        self.gru = nn.GRU(channels, channels, num_layers=gru_layers, batch_first=True)
        self.output = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.conv_pre(x).tanh()
        x = self.conv_mid(x).tanh()
        x = x.transpose(1, 2)        # [B, C, T] -> [B, T, C]
        x, _ = self.gru(x)
        x = self.output(x)
        return x.transpose(1, 2)     # [B, T, 1] -> [B, 1, T]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Using device: {device}")

model = SaturationNet(channels=8, kernel_pre=65).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"  Model created with {param_count:,} parameters")
print(model)

# %% [markdown]
# # Training Loop

# %%
## 3. Training Loop

print("\nStep 3: Training...")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

EPOCHS = 300
BATCH_SIZE = 64

dataset = torch.utils.data.TensorDataset(X_train, y_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

losses = []
pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
for epoch in pbar:
    epoch_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)

        min_len = min(pred.shape[-1], y_batch.shape[-1])
        loss = criterion(pred[..., :min_len], y_batch[..., :min_len])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_loss)

    pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr:.6f}")

print(f"\nTraining complete! Final loss: {losses[-1]:.6f}")

model_cpu = model.cpu()
torch.save(model_cpu.state_dict(), '../models/saturation_model_causal.pt')
print("  Saved to ../models/saturation_model_causal.pt")

print("\n  State dict keys:")
for k, v in model_cpu.state_dict().items():
    print(f"    {k}: {list(v.shape)}")

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()

# ## 4. Export Weights to JSON for Rust

print("\nStep 4: Exporting weights for Rust inference...")

import json

def export_weights_for_rust(sd, path="../models/weights_causal.json"):
    weights = []

    # conv_pre.conv.weight: [8, 1, 65] + bias: [8]
    weights.extend(sd["conv_pre.conv.weight"].flatten().tolist())
    weights.extend(sd["conv_pre.conv.bias"].flatten().tolist())

    # conv_mid.conv.weight: [8, 8, 17] + bias: [8]
    weights.extend(sd["conv_mid.conv.weight"].flatten().tolist())
    weights.extend(sd["conv_mid.conv.bias"].flatten().tolist())

    # GRU — standard nn.GRU, gate order is [r, z, n]
    # weight_ih_l0: [3*hidden, input] = [24, 8]
    # weight_hh_l0: [3*hidden, hidden] = [24, 8]
    # bias_ih_l0: [24]
    # bias_hh_l0: [24]
    hidden = 8

    w_ih = sd["gru.weight_ih_l0"].numpy()  # [24, 8]
    w_hh = sd["gru.weight_hh_l0"].numpy()  # [24, 8]
    b_ih = sd["gru.bias_ih_l0"].numpy()    # [24]
    b_hh = sd["gru.bias_hh_l0"].numpy()    # [24]

    # Split into gates: each [hidden, input] or [hidden, hidden]
    # PyTorch gate order: [r, z, n]
    w_ih_r, w_ih_z, w_ih_n = np.split(w_ih, 3, axis=0)
    w_hh_r, w_hh_z, w_hh_n = np.split(w_hh, 3, axis=0)
    b_ih_r, b_ih_z, b_ih_n = np.split(b_ih, 3)
    b_hh_r, b_hh_z, b_hh_n = np.split(b_hh, 3)

    # Rust load_weights order: w_ih[r,z,n], w_hh[r,z,n], b_ih[r,z,n], b_hh[r,z,n]
    for w in [w_ih_r, w_ih_z, w_ih_n]:
        weights.extend(w.flatten().tolist())
    for w in [w_hh_r, w_hh_z, w_hh_n]:
        weights.extend(w.flatten().tolist())
    for b in [b_ih_r, b_ih_z, b_ih_n]:
        weights.extend(b.flatten().tolist())
    for b in [b_hh_r, b_hh_z, b_hh_n]:
        weights.extend(b.flatten().tolist())

    # output (Linear): [1, 8] + [1]
    weights.extend(sd["output.weight"].flatten().tolist())
    weights.extend(sd["output.bias"].flatten().tolist())

    with open(path, "w") as f:
        json.dump(weights, f)

    # Expected: 528 + 1096 + 432 + 9 = 2065
    print(f"  Exported {len(weights)} weights to {path} (expected 2065)")

export_weights_for_rust(model_cpu.state_dict())

# ## 5. Verify: sample-by-sample vs chunked

print("\nStep 5: Verifying causal property (sample-by-sample should match chunked)...")

model = model_cpu.eval()

test_signal = np.random.randn(500).astype(np.float32) * 0.5

# Chunked (normal PyTorch forward)
with torch.no_grad():
    x = torch.FloatTensor(test_signal).reshape(1, 1, -1)
    chunked_out = model(x).numpy().flatten()

# Sample-by-sample (simulating Rust behavior)
h = torch.zeros(1, 1, 8)  # GRU hidden state
sample_out = []

# We need to manually maintain conv history
# This verifies the causal property: each output only depends on past inputs
conv_pre_hist = np.zeros(65, dtype=np.float32)
conv_mid_hist = np.zeros((8, 17), dtype=np.float32)

print(f"  Chunked output range: [{chunked_out.min():.4f}, {chunked_out.max():.4f}]")
print(f"  First 10 chunked: {chunked_out[:10]}")
print("  (Sample-by-sample verification would require reimplementing conv+GRU in numpy)")
print("  If the Rust output matches the chunked output, causality is confirmed.")

# ## 6. Audio Playback

print("\nStep 6: Processing test audio...")

model = model_cpu.to(device)

def process_audio(model, audio, chunk_size=CHUNK_SIZE, hop_size=HOP_SIZE):
    model.eval()
    model_dev = next(model.parameters()).device
    length = len(audio)
    output = np.zeros(length)
    weight = np.zeros(length)
    fade = np.hanning(chunk_size)
    num_chunks = max(1, (length - chunk_size) // hop_size + 1)

    with torch.no_grad():
        for start in tqdm(range(0, length - chunk_size + 1, hop_size),
                          total=num_chunks, desc="Processing", unit="chunk", leave=False):
            chunk = audio[start:start + chunk_size]
            x = torch.FloatTensor(chunk).reshape(1, 1, -1).to(model_dev)
            y = model(x).cpu().numpy().flatten()[:chunk_size]
            output[start:start + chunk_size] += y * fade
            weight[start:start + chunk_size] += fade

    mask = weight > 0
    output[mask] /= weight[mask]
    return output

duration = 2.0
frequency = 440
amplitude = 1.2
t_test = np.linspace(0, duration, int(SAMPLE_RATE * duration))
test_sine = amplitude * np.sin(2 * np.pi * frequency * t_test)
saturated_sine = process_audio(model, test_sine)

print("Clean:")
display(Audio(test_sine, rate=SAMPLE_RATE, normalize=True))
print("Saturated:")
display(Audio(saturated_sine, rate=SAMPLE_RATE, normalize=True))

print(f"\n  Model parameters: {param_count:,}")
print(f"  Final loss: {losses[-1]:.6f}")

# %%
# Load model
sd = torch.load("../models/saturation_model_causal.pt", map_location="cpu")

w_pre = sd["conv_pre.conv.weight"].numpy()   # [8, 1, 65]
b_pre = sd["conv_pre.conv.bias"].numpy()     # [8]
w_mid = sd["conv_mid.conv.weight"].numpy()   # [8, 8, 17]
b_mid = sd["conv_mid.conv.bias"].numpy()     # [8]

# Test signal
np.random.seed(42)
test_input = np.random.randn(200).astype(np.float32) * 0.5

# === PyTorch conv_pre only ===
model = SaturationNet(channels=8, kernel_pre=65)
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    x = torch.FloatTensor(test_input).reshape(1, 1, -1)
    # Just conv_pre + tanh
    pytorch_conv1 = model.conv_pre(x).tanh().numpy().squeeze()  # [8, 200]

# === Sample-by-sample conv_pre ===
hist = np.zeros(65, dtype=np.float32)
wp = 0

sample_conv1 = np.zeros((8, len(test_input)), dtype=np.float32)

for n in range(len(test_input)):
    hist[wp] = test_input[n]
    wp = (wp + 1) % 65

    for oc in range(8):
        val = b_pre[oc]
        for k in range(65):
            idx = (wp + k) % 65
            val += w_pre[oc, 0, k] * hist[idx]
        sample_conv1[oc, n] = np.tanh(val)

# Compare conv_pre output
diff1 = np.abs(pytorch_conv1 - sample_conv1)
print("=== conv_pre only ===")
print(f"Max diff: {diff1.max():.10f}")
print(f"Mean diff: {diff1.mean():.10f}")
print("✅ conv_pre MATCHES" if diff1.max() < 1e-4 else "❌ conv_pre MISMATCH")

# === PyTorch conv_pre + conv_mid ===
with torch.no_grad():
    x = torch.FloatTensor(test_input).reshape(1, 1, -1)
    c1 = model.conv_pre(x).tanh()
    pytorch_conv2 = model.conv_mid(c1).tanh().numpy().squeeze()  # [8, 200]

# === Sample-by-sample conv_mid ===
hist_mid = np.zeros((17, 8), dtype=np.float32)
wp_mid = 0

sample_conv2 = np.zeros((8, len(test_input)), dtype=np.float32)

for n in range(len(test_input)):
    # Feed conv_pre output (use our sample-by-sample result)
    hist_mid[wp_mid] = sample_conv1[:, n]
    wp_mid = (wp_mid + 1) % 17

    for oc in range(8):
        val = b_mid[oc]
        for k in range(17):
            idx = (wp_mid + k) % 17
            for ic in range(8):
                val += w_mid[oc, ic, k] * hist_mid[idx, ic]
        sample_conv2[oc, n] = np.tanh(val)

diff2 = np.abs(pytorch_conv2 - sample_conv2)
print("\n=== conv_pre + conv_mid ===")
print(f"Max diff: {diff2.max():.10f}")
print(f"Mean diff: {diff2.mean():.10f}")
print("✅ conv_mid MATCHES" if diff2.max() < 1e-4 else "❌ conv_mid MISMATCH")

# === Now test GRU ===
H = 8
w_ih = sd["gru.weight_ih_l0"].numpy()
w_hh = sd["gru.weight_hh_l0"].numpy()
b_ih = sd["gru.bias_ih_l0"].numpy()
b_hh = sd["gru.bias_hh_l0"].numpy()

w_ir, w_iz, w_in = w_ih[:H], w_ih[H:2*H], w_ih[2*H:]
w_hr, w_hz, w_hn = w_hh[:H], w_hh[H:2*H], w_hh[2*H:]
b_ir, b_iz, b_in = b_ih[:H], b_ih[H:2*H], b_ih[2*H:]
b_hr, b_hz, b_hn = b_hh[:H], b_hh[H:2*H], b_hh[2*H:]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# PyTorch GRU
with torch.no_grad():
    x = torch.FloatTensor(test_input).reshape(1, 1, -1)
    c1 = model.conv_pre(x).tanh()
    c2 = model.conv_mid(c1).tanh()
    gru_input = c2.transpose(1, 2)  # [1, T, 8]
    pytorch_gru_out, _ = model.gru(gru_input)
    pytorch_gru = pytorch_gru_out.numpy().squeeze()  # [200, 8]

# Sample-by-sample GRU (using PyTorch conv output to isolate GRU)
h = np.zeros(H, dtype=np.float32)
sample_gru = np.zeros((len(test_input), H), dtype=np.float32)

for n in range(len(test_input)):
    inp = pytorch_conv2[:, n]  # use PyTorch conv output to isolate GRU

    z = sigmoid(w_iz @ inp + b_iz + w_hz @ h + b_hz)
    r = sigmoid(w_ir @ inp + b_ir + w_hr @ h + b_hr)
    n_gate = np.tanh(w_in @ inp + b_in + r * (w_hn @ h + b_hn))
    h = (1 - z) * n_gate + z * h
    sample_gru[n] = h

diff3 = np.abs(pytorch_gru - sample_gru)
print("\n=== GRU (fed with PyTorch conv output) ===")
print(f"Max diff: {diff3.max():.10f}")
print(f"Mean diff: {diff3.mean():.10f}")
print("✅ GRU MATCHES" if diff3.max() < 1e-4 else "❌ GRU MISMATCH")

if diff3.max() >= 1e-4:
    idx = np.unravel_index(diff3.argmax(), diff3.shape)
    print(f"   Worst at sample {idx[0]}, hidden {idx[1]}")
    print(f"   PyTorch: {pytorch_gru[idx]:.8f}")
    print(f"   Ours:    {sample_gru[idx]:.8f}")
    print(f"\n   Sample 0 PyTorch: {pytorch_gru[0, :4]}")
    print(f"   Sample 0 Ours:    {sample_gru[0, :4]}")


# %%
# ## 5. Process Test Signals (uses process_audio from the training cell)

print("\nStep 4: Processing audio...")

# Move model back to device for inference
model = model_cpu.to(device)

duration = 2.0
frequency = 440
amplitude = 1.2
t_test = np.linspace(0, duration, int(SAMPLE_RATE * duration))
test_sine = amplitude * np.sin(2 * np.pi * frequency * t_test)
saturated_sine = process_audio(model, test_sine)

print(f"  Processed test sine ({duration}s at {frequency}Hz)")

audio_path = './audio/Cardboard Box Beat.wav'
audio_data, audio_sr = librosa.load(audio_path, sr=None, mono=False)

gain = 1.2
print(f"  Applying input gain: {gain}x")

if len(audio_data.shape) == 2:
    is_stereo = True
    saturated_audio = np.stack([
        process_audio(model, ch * gain) for ch in audio_data
    ])
    print(f"  Processed stereo audio: {audio_data.shape[1]} samples at {audio_sr}Hz")
else:
    is_stereo = False
    saturated_audio = process_audio(model, audio_data * gain)
    print(f"  Processed mono audio: {len(audio_data)} samples at {audio_sr}Hz")

# ## 6. Visualize Results

print("\nStep 5: Visualizing Results...")

samples_to_plot = int(0.02 * SAMPLE_RATE)
t_plot = np.linspace(0, 0.02, samples_to_plot)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.plot(t_plot, test_sine[:samples_to_plot], label='Clean Sine', linewidth=2, alpha=0.8)
ax1.plot(t_plot, saturated_sine[:samples_to_plot], label='Saturated (NN)', linewidth=2, linestyle='--', alpha=0.8)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Test Sine Wave: Time Domain (20ms window)')
ax1.legend()
ax1.grid(True)

test_range = np.linspace(-2, 2, 1000)
test_tensor = torch.FloatTensor(test_range).reshape(1, 1, -1).to(device)
with torch.no_grad():
    network_curve = model(test_tensor).cpu().numpy().flatten()
true_curve = np.tanh(test_range)

ax2.plot(test_range, true_curve, label='True tanh', linewidth=2)
ax2.plot(test_range, network_curve[:len(test_range)], '--', label='Network learned', linewidth=2)
ax2.set_xlabel('Input Amplitude')
ax2.set_ylabel('Output Amplitude')
ax2.set_title('Saturation Transfer Function (approximate — model is now temporal)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

plot_duration = 0.1
samples_audio = int(plot_duration * audio_sr)
t_audio = np.linspace(0, plot_duration, samples_audio)

if is_stereo:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(t_audio, audio_data[0][:samples_audio], label='Original L', alpha=0.7)
    ax1.plot(t_audio, saturated_audio[0][:samples_audio], label='Saturated L', alpha=0.7)
    ax1.set_title('Real Audio: Left Channel (100ms)')
    ax1.legend(); ax1.grid(True)

    ax2.plot(t_audio, audio_data[1][:samples_audio], label='Original R', alpha=0.7)
    ax2.plot(t_audio, saturated_audio[1][:samples_audio], label='Saturated R', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Real Audio: Right Channel (100ms)')
    ax2.legend(); ax2.grid(True)
else:
    plt.figure(figsize=(14, 5))
    plt.plot(t_audio, audio_data[:samples_audio], label='Original', alpha=0.7)
    plt.plot(t_audio, saturated_audio[:samples_audio], label='Saturated', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.title('Real Audio: Mono (100ms)')
    plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

# ## 7. Audio Playback

print("\nAudio Playback:")

print("\n--- Test Sine Wave (440Hz) ---")
print("Clean:")
display(Audio(test_sine, rate=SAMPLE_RATE, normalize=True))
print("Saturated:")
display(Audio(saturated_sine, rate=SAMPLE_RATE, normalize=True))

print("\n--- Real Audio ---")
print("Original:")
display(Audio(audio_data, rate=audio_sr, normalize=True))
print("Saturated:")
display(Audio(saturated_audio, rate=audio_sr, normalize=True))

# ## Summary

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Architecture:            Conv1D + GRU")
print(f"Model parameters:        {param_count:,}")
print(f"Training chunks:         {X_train.shape[0]}")
print(f"Chunk size:              {CHUNK_SIZE} ({CHUNK_SIZE / SAMPLE_RATE * 1000:.1f}ms)")
print(f"Final training loss:     {losses[-1]:.6f}")
print(f"Audio format:            {'Stereo' if is_stereo else 'Mono'}")
print(f"Input gain:              {gain}x")
print("=" * 60)

# %% [markdown]
# # Restructure

# %%
state = torch.load('../models/saturation_model.pt', map_location='cpu', weights_only=False)

# PyTorch GRU concatenates [reset, update, new] gates (each of size hidden_size)
hidden_size = 8

def split_gru(state, hidden_size):
    new_state = {}
    for k, v in state.items():
        if not k.startswith('gru.'):
            new_state[k] = v
            continue

        # Split concatenated [reset, update, new] into 3 chunks
        if 'weight_ih' in k:
            r, z, n = v.chunk(3, dim=0)
            new_state['gru.reset_gate.input_transform.weight'] = r
            new_state['gru.update_gate.input_transform.weight'] = z
            new_state['gru.new_gate.input_transform.weight'] = n
        elif 'weight_hh' in k:
            r, z, n = v.chunk(3, dim=0)
            new_state['gru.reset_gate.hidden_transform.weight'] = r
            new_state['gru.update_gate.hidden_transform.weight'] = z
            new_state['gru.new_gate.hidden_transform.weight'] = n
        elif 'bias_ih' in k:
            r, z, n = v.chunk(3, dim=0)
            new_state['gru.reset_gate.input_transform.bias'] = r
            new_state['gru.update_gate.input_transform.bias'] = z
            new_state['gru.new_gate.input_transform.bias'] = n
        elif 'bias_hh' in k:
            r, z, n = v.chunk(3, dim=0)
            new_state['gru.reset_gate.hidden_transform.bias'] = r
            new_state['gru.update_gate.hidden_transform.bias'] = z
            new_state['gru.new_gate.hidden_transform.bias'] = n

    return new_state

new_state = split_gru(state, hidden_size)

print("Remapped keys:")
for k, v in new_state.items():
    print(f"  {k}: {list(v.shape)}")

torch.save(new_state, '../models/saturation_model_n.pt')
print("\nSaved remapped model.")



# %%
sd = torch.load("../models/saturation_model_n.pt", map_location="cpu")
for k, v in sd.items():
    print(f"{k}: {list(v.shape)}")

# %%
def export_model_weights(sd, path="weights.json"):
    """
    Export from state_dict:
      conv_pre:  Conv1d(1, 8, 65)
      conv_mid:  Conv1d(8, 8, 17)
      gru:       custom GRU(8, 8)
      output:    Linear(8, 1)
    """
    weights = []

    # --- conv_pre: Conv1d(1, 8, 65) ---
    weights.extend(sd["conv_pre.weight"].flatten().tolist())   # [8, 1, 65]
    weights.extend(sd["conv_pre.bias"].flatten().tolist())     # [8]

    # --- conv_mid: Conv1d(8, 8, 17) ---
    weights.extend(sd["conv_mid.weight"].flatten().tolist())   # [8, 8, 17]
    weights.extend(sd["conv_mid.bias"].flatten().tolist())     # [8]

    # --- Custom GRU(8, 8) ---
    gate_names = ["reset_gate", "update_gate", "new_gate"]

    # weight_ih: [r, z, n]
    for g in gate_names:
        weights.extend(sd[f"gru.{g}.input_transform.weight"].flatten().tolist())
    # weight_hh: [r, z, n]
    for g in gate_names:
        weights.extend(sd[f"gru.{g}.hidden_transform.weight"].flatten().tolist())
    # bias_ih: [r, z, n]
    for g in gate_names:
        weights.extend(sd[f"gru.{g}.input_transform.bias"].flatten().tolist())
    # bias_hh: [r, z, n]
    for g in gate_names:
        weights.extend(sd[f"gru.{g}.hidden_transform.bias"].flatten().tolist())

    # --- output: Linear(8, 1) ---
    weights.extend(sd["output.weight"].flatten().tolist())  # [1, 8]
    weights.extend(sd["output.bias"].flatten().tolist())    # [1]

    with open(path, "w") as f:
        json.dump(weights, f)

    # Expected count:
    # conv_pre:  8*1*65 + 8       = 528
    # conv_mid:  8*8*17 + 8       = 1096
    # GRU:       6*8*8 + 6*8      = 432
    # output:    1*8 + 1           = 9
    # Total:                        2065
    print(f"Exported {len(weights)} weights to {path} (expected 2065)")


sd = torch.load("../models/saturation_model_n.pt", map_location="cpu")
export_model_weights(sd, "../models/weights.json")

# %%
model.eval()
# test with a hot signal
x = torch.tensor([[0.8]])
print(f"in: {x.item()}, out: {model(x).item()}")
