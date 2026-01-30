"""Preprocessing pipeline for EEG sleep staging"""
import numpy as np
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
import mne


class EEGPreprocessor:
    """Preprocess EEG data for sleep staging"""

    def __init__(self, sfreq=500, epoch_duration=30):
        self.sfreq = sfreq
        self.epoch_duration = epoch_duration
        self.epoch_samples = int(epoch_duration * sfreq)

    def bandpass_filter(self, data, lowcut=0.5, highcut=50):
        """Apply bandpass filter"""
        sos = signal.butter(5, [lowcut, highcut], btype='bandpass', fs=self.sfreq, output='sos')
        filtered = signal.sosfiltfilt(sos, data, axis=-1)
        return filtered

    def notch_filter(self, data, freq=50, Q=30):
        """Apply notch filter for power line noise"""
        b, a = signal.iirnotch(freq, Q, self.sfreq)
        filtered = signal.filtfilt(b, a, data, axis=-1)
        return filtered

    def normalize(self, data):
        """Robust normalization per channel using median and IQR"""
        median = np.median(data, axis=-1, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=-1, keepdims=True)
        iqr = q75 - q25
        normalized = (data - median) / (iqr + 1e-8)
        return normalized

    def create_epochs(self, data, labels):
        """Segment continuous data into epochs"""
        n_samples = data.shape[1]
        n_epochs = n_samples // self.epoch_samples

        # Truncate to fit complete epochs
        data_truncated = data[:, :n_epochs * self.epoch_samples]

        # Reshape into epochs
        epochs = data_truncated.reshape(data.shape[0], n_epochs, self.epoch_samples)
        epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, epoch_samples)

        # Truncate labels if needed
        if len(labels) > n_epochs:
            labels = labels[:n_epochs]

        return epochs, labels

    def remove_unknown_class(self, epochs, labels):
        """Remove epochs with label 0 (Unknown/Movement)"""
        mask = labels != 0
        return epochs[mask], labels[mask]

    def remove_artifacts(self, epochs, labels, threshold_percentile=95):
        """Remove epochs with extreme amplitudes (artifacts)

        Args:
            epochs: (n_epochs, n_channels, epoch_samples)
            labels: (n_epochs,)
            threshold_percentile: percentile for amplitude threshold (default 95)

        Returns:
            Cleaned epochs and labels
        """
        # Compute max amplitude per epoch across all channels
        max_amp = np.abs(epochs).max(axis=(1, 2))

        # Set threshold at percentile to remove extreme outliers
        threshold = np.percentile(max_amp, threshold_percentile)

        # Keep epochs below threshold
        mask = max_amp < threshold

        n_removed = (~mask).sum()
        n_total = len(mask)
        print(f"  Artifact rejection: removed {n_removed}/{n_total} epochs ({100*n_removed/n_total:.1f}%)")

        return epochs[mask], labels[mask]

    def preprocess(self, data, labels, remove_artifacts=True):
        """Complete preprocessing pipeline

        Args:
            data: raw EEG data
            labels: sleep stage labels
            remove_artifacts: whether to apply artifact rejection (default True)
        """
        # 1. Bandpass filter
        data = self.bandpass_filter(data)

        # 2. Notch filter
        data = self.notch_filter(data)

        # 3. Normalize (robust method)
        data = self.normalize(data)

        # 4. Create epochs
        epochs, labels = self.create_epochs(data, labels)

        # 5. Remove unknown class
        epochs, labels = self.remove_unknown_class(epochs, labels)

        # 6. Remove artifacts (optional)
        if remove_artifacts:
            epochs, labels = self.remove_artifacts(epochs, labels)

        return epochs, labels


def load_and_preprocess_subject(subject_id, data_dir, cache_dir=None):
    """Load and preprocess single subject, with optional .npz caching.

    Args:
        subject_id: e.g. 'sub-001'
        data_dir: path to dataset root
        cache_dir: if set, preprocessed data is saved/loaded from this directory
    """
    # Try loading from cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{subject_id}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached['epochs'], cached['labels'], subject_id

    data_dir = Path(data_dir)
    set_file = data_dir / subject_id / "eeg" / f"{subject_id}_task-sleep_eeg.set"

    # Load data using MNE
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    data = raw.get_data().astype(np.float32)
    sfreq = raw.info['sfreq']

    # Load labels
    mat_data = loadmat(str(set_file), simplify_cells=True)
    labels = mat_data['VisualHypnogram'].flatten()

    # Preprocess
    preprocessor = EEGPreprocessor(sfreq=sfreq)
    epochs, labels = preprocessor.preprocess(data, labels)

    # Save to cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{subject_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, epochs=epochs, labels=labels)

    return epochs, labels, subject_id
