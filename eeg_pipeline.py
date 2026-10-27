"""Basic EEG signal preprocessing and feature extraction for educational experiments."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

FEATURE_NAMES = [
    "mean", "std", "variance", "rms", "line_length", "zero_crossings",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "dominant_frequency"
]

def bandpass_filter(signal, fs, low=0.5, high=40.0, order=4, notch_hz=50.0):
    """Filter a 1-D EEG signal. Ensure high < Nyquist and sampling rate is appropriate."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 16:
        raise ValueError("signal must be a 1-D array with at least 16 samples")
    nyquist = fs / 2
    if not (0 < low < high < nyquist):
        raise ValueError("Require 0 < low < high < sampling_rate/2")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    if notch_hz and 0 < notch_hz < nyquist:
        b, a = iirnotch(notch_hz, Q=30, fs=fs)
        y = filtfilt(b, a, y)
    return y

def extract_features(signal, fs):
    """Return a dictionary of simple time/frequency features for one EEG segment."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 16 or not np.isfinite(x).all():
        raise ValueError("signal must be finite, 1-D, and contain at least 16 samples")
    centered = x - np.mean(x)
    freqs, psd = welch(centered, fs=fs, nperseg=min(256, len(x)))
    bands = {
        "delta_power": (0.5, 4), "theta_power": (4, 8),
        "alpha_power": (8, 13), "beta_power": (13, 30),
        "gamma_power": (30, min(40, fs/2 - 0.1))
    }
    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "variance": float(np.var(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "line_length": float(np.mean(np.abs(np.diff(x)))),
        "zero_crossings": float(np.mean(np.diff(np.signbit(centered)) != 0)),
        "dominant_frequency": float(freqs[np.argmax(psd)]) if len(psd) else 0.0,
    }
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.sum() > 1 else 0.0
    return out
