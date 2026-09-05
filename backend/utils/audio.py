"""
Audio preprocessing utilities for EchoLocator.

Handles:
- Loading and resampling audio files
- Noise reduction
- Feature extraction (pitch, spectral, energy)
- AASIST-compatible waveform preprocessing
"""

import numpy as np
import librosa
import scipy.io.wavfile as wavfile

try:
    import noisereduce as nr
except ImportError:
    nr = None

# AASIST expects 16kHz mono audio
AASIST_SAMPLE_RATE = 16000


def remove_background_noise(file_path, out_path="clean_temp.wav"):
    """Apply noise reduction to audio file."""
    if nr is None:
        print("[Audio] noisereduce not installed. Skipping noise reduction.")
        return file_path

    try:
        rate, data = wavfile.read(file_path)
        # Handle stereo by taking first channel
        if data.ndim > 1:
            data = data[:, 0]
        reduced_noise = nr.reduce_noise(y=data.astype(np.float32), sr=rate, prop_decrease=0.7)
        wavfile.write(out_path, rate, reduced_noise.astype(data.dtype))
        return out_path
    except Exception as e:
        print(f"[Audio] Noise reduction failed: {e}")
        return file_path


def extract_audio_features(file_path):
    """
    Extract acoustic features for heuristic analysis.

    Returns dict with:
        mean_energy, energy_var, mean_zcr, mean_flatness,
        pitch_mean, pitch_std, is_voiced
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"[Audio] Librosa load error: {e}")
        return None

    if len(y) == 0:
        return None

    # Energy (volume/amplitude stability)
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))
    energy_var = float(np.var(rms))

    # Zero Crossing Rate (background noise realism)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mean_zcr = float(np.mean(zcr))

    # Spectral Flatness (synthetic = over-smoothed frequencies)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = float(np.mean(flatness))

    # Pitch tracking
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_f0 = f0[~np.isnan(f0)]
    except Exception:
        valid_f0 = np.array([])

    pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0
    pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0

    return {
        "mean_energy": mean_energy,
        "energy_var": energy_var,
        "mean_zcr": mean_zcr,
        "mean_flatness": mean_flatness,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "is_voiced": len(valid_f0) > 0,
    }


def preprocess_for_aasist(file_path, target_sr=AASIST_SAMPLE_RATE):
    """
    Convert audio file to raw waveform tensor suitable for AASIST model input.

    Loads audio, resamples to target_sr (16kHz), normalizes, and returns
    as a numpy array of shape (samples,).

    Args:
        file_path: path to WAV file
        target_sr: target sample rate (default 16000)

    Returns:
        numpy array (float32) of shape (samples,) or None if failed
    """
    try:
        # Load and resample to target_sr
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)

        if len(y) == 0:
            return None

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y.astype(np.float32)

    except Exception as e:
        print(f"[Audio] Preprocessing for AASIST failed: {e}")
        return None
