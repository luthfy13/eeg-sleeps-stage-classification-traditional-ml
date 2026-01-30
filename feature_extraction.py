"""Feature extraction for EEG sleep staging (traditional ML)"""
import numpy as np
from scipy import signal, stats


BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'sigma': (12, 16),
    'beta': (13, 30),
    'gamma': (30, 50),
}

BAND_NAMES = list(BANDS.keys())


def band_power(psd, freqs, low, high):
    """Compute power in a frequency band from PSD."""
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])


def spectral_entropy(psd):
    """Compute spectral entropy of a PSD."""
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))


def hjorth_parameters(x):
    """Compute Hjorth activity, mobility, complexity."""
    activity = np.var(x)
    dx = np.diff(x)
    dx_var = np.var(dx)
    ddx_var = np.var(np.diff(dx))

    mobility = np.sqrt(dx_var / (activity + 1e-12))
    complexity = np.sqrt(ddx_var / (dx_var + 1e-12)) / (mobility + 1e-12)
    return activity, mobility, complexity


def zero_crossing_rate(x):
    """Count zero crossings normalized by signal length."""
    return np.sum(np.diff(np.sign(x)) != 0) / len(x)


def detect_sleep_spindles(x, sfreq=500):
    """Detect sleep spindles (11-16 Hz, characteristic of N2 stage)

    Args:
        x: 1-D signal array (single channel, single epoch)
        sfreq: sampling frequency

    Returns:
        Spindle density (proportion of samples above threshold)
    """
    # Bandpass filter for sigma band (spindles)
    sos = signal.butter(4, [11, 16], btype='band', fs=sfreq, output='sos')
    filtered = signal.sosfiltfilt(sos, x)

    # Compute envelope using Hilbert transform
    analytic_signal = signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)

    # Threshold: mean + 2*std
    threshold = np.mean(envelope) + 2 * np.std(envelope)

    # Count proportion of samples above threshold
    spindle_density = (envelope > threshold).sum() / len(x)

    return float(spindle_density)


def detect_slow_waves(x, sfreq=500):
    """Detect slow waves (0.5-2 Hz, characteristic of N3 stage)

    Args:
        x: 1-D signal array (single channel, single epoch)
        sfreq: sampling frequency

    Returns:
        Slow wave density and average amplitude
    """
    # Bandpass filter for delta band
    sos = signal.butter(4, [0.5, 2], btype='band', fs=sfreq, output='sos')
    filtered = signal.sosfiltfilt(sos, x)

    # Detect high-amplitude slow waves
    threshold = np.percentile(np.abs(filtered), 75)
    slow_wave_mask = np.abs(filtered) > threshold

    density = slow_wave_mask.sum() / len(x)
    avg_amplitude = np.mean(np.abs(filtered[slow_wave_mask])) if slow_wave_mask.any() else 0.0

    return float(density), float(avg_amplitude)


def detect_alpha_waves(x, sfreq=500):
    """Detect alpha waves (8-13 Hz, characteristic of relaxed wakefulness)

    Args:
        x: 1-D signal array (single channel, single epoch)
        sfreq: sampling frequency

    Returns:
        Alpha wave ratio and dominance
    """
    # Bandpass filter for alpha band
    sos = signal.butter(4, [8, 13], btype='band', fs=sfreq, output='sos')
    filtered = signal.sosfiltfilt(sos, x)

    # Compute power in alpha band vs total power
    alpha_power = np.mean(filtered ** 2)
    total_power = np.mean(x ** 2)
    alpha_ratio = alpha_power / (total_power + 1e-12)

    # Compute peak alpha frequency
    f, pxx = signal.welch(x, fs=sfreq, nperseg=min(4 * sfreq, len(x)))
    alpha_idx = np.logical_and(f >= 8, f <= 13)
    alpha_peak = f[alpha_idx][np.argmax(pxx[alpha_idx])] if alpha_idx.any() else 10.0

    return float(alpha_ratio), float(alpha_peak)


def detect_sawtooth_waves(x, sfreq=500):
    """Detect sawtooth waves (2-6 Hz, characteristic of REM stage)

    Args:
        x: 1-D signal array (single channel, single epoch)
        sfreq: sampling frequency

    Returns:
        Sawtooth wave presence indicator
    """
    # Bandpass filter for theta band
    sos = signal.butter(4, [2, 6], btype='band', fs=sfreq, output='sos')
    filtered = signal.sosfiltfilt(sos, x)

    # Compute envelope
    analytic_signal = signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)

    # High-frequency modulation indicates sawtooth pattern
    envelope_std = np.std(envelope)

    return float(envelope_std)


def spectral_shape_features(x, sfreq=500, n_windows=6):
    """Compute spectral shape descriptors from sub-segments.

    Splits signal into n_windows, computes spectral centroid/spread/skewness/
    kurtosis/entropy per window, then returns statistics (mean, std, median,
    iqr, min, max) across windows.

    Args:
        x: 1-D signal array (single channel, single epoch)
        sfreq: sampling frequency
        n_windows: number of sub-segments (30s / 6 = 5s each)

    Returns:
        list of floats (5 descriptors × 6 stats = 30 features)
    """
    win_len = len(x) // n_windows
    descriptors = {
        'entropy': [], 'centroid': [], 'spread': [],
        'skewness': [], 'kurtosis': [],
    }

    for w in range(n_windows):
        seg = x[w * win_len : (w + 1) * win_len]
        f, pxx = signal.welch(seg, fs=sfreq, nperseg=min(sfreq, len(seg)))
        pxx = pxx + 1e-12
        total = pxx.sum()
        pxx_norm = pxx / total

        # Spectral entropy
        descriptors['entropy'].append(-np.sum(pxx_norm * np.log2(pxx_norm)))

        # Spectral centroid (mean frequency)
        centroid = np.sum(f * pxx_norm)
        descriptors['centroid'].append(centroid)

        # Spectral spread (std of frequency)
        spread = np.sqrt(np.sum(((f - centroid) ** 2) * pxx_norm))
        descriptors['spread'].append(spread)

        # Spectral skewness
        if spread > 0:
            descriptors['skewness'].append(
                np.sum(((f - centroid) ** 3) * pxx_norm) / (spread ** 3)
            )
        else:
            descriptors['skewness'].append(0.0)

        # Spectral kurtosis
        if spread > 0:
            descriptors['kurtosis'].append(
                np.sum(((f - centroid) ** 4) * pxx_norm) / (spread ** 4)
            )
        else:
            descriptors['kurtosis'].append(0.0)

    # Aggregate stats across windows
    features = []
    for vals in descriptors.values():
        arr = np.array(vals)
        features.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.median(arr)),
            float(np.subtract(*np.percentile(arr, [75, 25]))),  # IQR
            float(np.min(arr)),
            float(np.max(arr)),
        ])

    return features


SPECTRAL_SHAPE_NAMES = []
for _desc in ('entropy', 'centroid', 'spread', 'skewness', 'kurtosis'):
    for _stat in ('mean', 'std', 'median', 'iqr', 'min', 'max'):
        SPECTRAL_SHAPE_NAMES.append(f"ss_{_desc}_{_stat}")


def extract_epoch_features(epoch, sfreq=500):
    """Extract features from a single epoch.

    Args:
        epoch: numpy array (n_channels, epoch_samples)
        sfreq: sampling frequency

    Returns:
        1-D feature vector
    """
    n_channels = epoch.shape[0]
    features = []

    # Per-channel PSD (compute once per channel)
    psds = []
    freqs = None
    for ch in range(n_channels):
        f, pxx = signal.welch(epoch[ch], fs=sfreq, nperseg=min(4 * sfreq, epoch.shape[1]))
        psds.append(pxx)
        freqs = f

    # Per-channel features
    for ch in range(n_channels):
        pxx = psds[ch]
        x = epoch[ch]

        # Absolute band power (6)
        abs_powers = []
        for low, high in BANDS.values():
            bp = band_power(pxx, freqs, low, high)
            abs_powers.append(bp)
        features.extend(abs_powers)

        # Relative band power (6)
        total_power = sum(abs_powers) + 1e-12
        features.extend([bp / total_power for bp in abs_powers])

        # Spectral entropy (1)
        features.append(spectral_entropy(pxx))

        # Hjorth parameters (3)
        features.extend(hjorth_parameters(x))

        # Statistical features (4)
        features.append(float(np.mean(x)))
        features.append(float(np.std(x)))
        features.append(float(stats.skew(x)))
        features.append(float(stats.kurtosis(x)))

        # Zero crossing rate (1)
        features.append(zero_crossing_rate(x))

        # Spectral shape descriptors from sub-segments (30)
        features.extend(spectral_shape_features(x, sfreq=sfreq))

        # Sleep-specific features (8 new features per channel)
        # Sleep spindles (N2 indicator) - 1 feature
        features.append(detect_sleep_spindles(x, sfreq=sfreq))

        # Slow waves (N3 indicator) - 2 features
        sw_density, sw_amplitude = detect_slow_waves(x, sfreq=sfreq)
        features.append(sw_density)
        features.append(sw_amplitude)

        # Alpha waves (Wake indicator) - 2 features
        alpha_ratio, alpha_peak = detect_alpha_waves(x, sfreq=sfreq)
        features.append(alpha_ratio)
        features.append(alpha_peak)

        # Sawtooth waves (REM indicator) - 1 feature
        features.append(detect_sawtooth_waves(x, sfreq=sfreq))

    # Inter-channel coherence (3 pairs × 2 bands = 6)
    pairs = [(0, 1), (0, 2), (1, 2)] if n_channels >= 3 else [(0, 1)]
    for ch_a, ch_b in pairs:
        f_coh, coh = signal.coherence(
            epoch[ch_a], epoch[ch_b], fs=sfreq,
            nperseg=min(4 * sfreq, epoch.shape[1])
        )
        # Average coherence in delta and theta bands
        for band_name in ('delta', 'theta'):
            low, high = BANDS[band_name]
            idx = np.logical_and(f_coh >= low, f_coh <= high)
            features.append(float(np.mean(coh[idx])) if idx.any() else 0.0)

    return np.array(features, dtype=np.float32)


def extract_all_features(epochs, sfreq=500, context=2, n_jobs=-1):
    """Extract features from all epochs with temporal context.

    Args:
        epochs: numpy array (n_epochs, n_channels, epoch_samples)
        sfreq: sampling frequency
        context: number of neighboring epochs on each side to include
                 (their features + delta from center epoch)
        n_jobs: number of parallel jobs (-1 for all cores)

    Returns:
        numpy array (n_epochs, n_features)
    """
    # First extract per-epoch features (parallelized)
    if n_jobs != 1:
        from joblib import Parallel, delayed
        base_features = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(extract_epoch_features)(ep, sfreq) for ep in epochs
        )
        base_features = np.array(base_features)
    else:
        base_features = np.array([extract_epoch_features(ep, sfreq) for ep in epochs])

    n_epochs, n_base = base_features.shape

    if context == 0:
        return base_features

    # Build temporal features: center + neighbor features + deltas + ratios
    # This captures more temporal patterns than just deltas
    all_features = []
    for i in range(n_epochs):
        feat = [base_features[i]]  # Center epoch features

        for offset in range(-context, context + 1):
            if offset == 0:
                continue
            neighbor_idx = np.clip(i + offset, 0, n_epochs - 1)

            # Actual neighbor features (captures absolute patterns)
            feat.append(base_features[neighbor_idx])

            # Delta (change) from center to neighbor
            feat.append(base_features[neighbor_idx] - base_features[i])

            # Ratio (relative change, more robust to scaling)
            ratio = base_features[neighbor_idx] / (np.abs(base_features[i]) + 1e-8)
            feat.append(ratio)

        all_features.append(np.concatenate(feat))

    return np.array(all_features, dtype=np.float32)


def _base_feature_names(n_channels=3):
    """Return base (per-epoch) feature names."""
    names = []
    for ch in range(n_channels):
        prefix = f"ch{ch}"
        for band in BAND_NAMES:
            names.append(f"{prefix}_abs_{band}")
        for band in BAND_NAMES:
            names.append(f"{prefix}_rel_{band}")
        names.append(f"{prefix}_spectral_entropy")
        names.append(f"{prefix}_hjorth_activity")
        names.append(f"{prefix}_hjorth_mobility")
        names.append(f"{prefix}_hjorth_complexity")
        names.append(f"{prefix}_mean")
        names.append(f"{prefix}_std")
        names.append(f"{prefix}_skewness")
        names.append(f"{prefix}_kurtosis")
        names.append(f"{prefix}_zcr")
        for sn in SPECTRAL_SHAPE_NAMES:
            names.append(f"{prefix}_{sn}")

        # Sleep-specific features
        names.append(f"{prefix}_spindle_density")
        names.append(f"{prefix}_slow_wave_density")
        names.append(f"{prefix}_slow_wave_amplitude")
        names.append(f"{prefix}_alpha_ratio")
        names.append(f"{prefix}_alpha_peak")
        names.append(f"{prefix}_sawtooth_indicator")

    pairs = [(0, 1), (0, 2), (1, 2)] if n_channels >= 3 else [(0, 1)]
    for ch_a, ch_b in pairs:
        for band in ('delta', 'theta'):
            names.append(f"coh_{ch_a}_{ch_b}_{band}")

    return names


def get_feature_names(n_channels=3, context=2):
    """Return feature names including temporal context.

    With improved temporal features:
    - center: features from current epoch
    - neighbor: actual features from neighboring epochs
    - delta: difference between neighbor and center
    - ratio: ratio between neighbor and center
    """
    base = _base_feature_names(n_channels)
    if context == 0:
        return base

    names = [f"center_{n}" for n in base]
    for offset in range(-context, context + 1):
        if offset == 0:
            continue
        sign = f"m{abs(offset)}" if offset < 0 else f"p{offset}"

        # Actual neighbor features
        for n in base:
            names.append(f"neighbor_{sign}_{n}")

        # Delta features
        for n in base:
            names.append(f"delta_{sign}_{n}")

        # Ratio features
        for n in base:
            names.append(f"ratio_{sign}_{n}")

    return names
