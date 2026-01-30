# Dataset Information

## Overview

**Dataset Name:** UCSD Forehead Patch Sleep Validation Dataset
**Source:** [OpenNeuro ds004745](https://openneuro.org/datasets/ds004745)
**Publication:** Onton et al. (2024). "Validation of spectral sleep scoring with polysomnography using forehead EEG device." *Frontiers in Sleep.*
**DOI:** 10.3389/frsle.2024.1349537

---

## Dataset Specifications

### Recording Details
- **Device:** CGX forehead patch (3-electrode EEG)
- **Sampling Frequency:** 500 Hz
- **Recording Type:** Continuous overnight sleep recordings
- **Average Duration:** ~8 hours (varies by subject: 19,964s to 29,964s)
- **Power Line Frequency:** 60 Hz (US standard)
- **Reference:** Common reference (AFz)

### EEG Channels (3 channels)
1. **FP1-AFz** - Left forehead to center reference
2. **FP2-AFz** - Right forehead to center reference
3. **FF** - Forehead-to-forehead (differential)

These are **forehead-only** EEG channels, which is different from traditional polysomnography that uses scalp electrodes (e.g., C3, C4, O1, O2). Forehead EEG is:
- ✅ More comfortable for home use
- ✅ Easier to apply
- ❌ Less signal from posterior brain regions
- ❌ More prone to eye movement artifacts

---

## Sleep Stage Labels

Labels are stored in `EEG.VisualHypnogram` within the `.set` files.

### Label Encoding (Original)
```
1 = Wake
2 = REM (Rapid Eye Movement)
3 = N1 (NREM Stage 1 - Light sleep)
4 = N2 (NREM Stage 2 - Intermediate sleep)
5 = N3 (NREM Stage 3 - Deep sleep / Slow Wave Sleep)
0 = Unknown / Movement (excluded from analysis)
```

### Label Encoding (After Preprocessing)
Our code remaps to 0-indexed:
```
0 = Wake
1 = REM
2 = N1
3 = N2
4 = N3
```

### Epoch Duration
- **30 seconds per epoch** (standard in sleep staging)
- Each subject has ~900-1000 epochs (excluding label 0)

---

## Subjects

### Total Subjects: 22 (19 used in analysis)

**Excluded:** sub-113, sub-121 (inadequate patch data)
**Missing:** sub-103, sub-108, sub-115, sub-118, sub-120

### Demographics (19 subjects analyzed)

| Subject ID | Age | Sex | Ethnicity |
|------------|-----|-----|-----------|
| sub-101 | 29 | Female | Asian |
| sub-102 | 24 | Female | White |
| sub-104 | 21 | Male | Asian |
| sub-105 | 25 | Male | Hispanic White |
| sub-106 | 20 | Male | Mixed |
| sub-107 | 20 | Male | Filipino/Mexican |
| sub-109 | 20 | Male | Asian |
| sub-110 | 19 | Female | Hispanic |
| sub-111 | 26 | Male | White |
| sub-112 | 20 | Female | Latino/Hispanic |
| sub-114 | 26 | Female | White |
| sub-116 | 24 | Female | Multiracial Chinese/White |
| sub-117 | 27 | Male | White |
| sub-119 | 21 | Female | Hispanic |
| sub-122 | 27 | Female | Caucasian |
| sub-123 | 23 | Female | Caucasian |
| sub-124 | 20 | Male | Filipino, Mexican |
| sub-125 | 20 | Male | Chinese |
| sub-126 | 23 | Female | Caucasian |

**Age Range:** 19-29 years (young adults)
**Sex Distribution:** 10 Female, 9 Male
**Ethnicity:** Diverse (Asian, White, Hispanic, Mixed)

---

## File Structure

```
Sleep Staging with Forehead EEG/
├── README                          # Dataset documentation
├── dataset_description.json        # BIDS metadata
├── participants.tsv                # Subject demographics
├── participants.json               # Demographics schema
├── CHANGES                         # Version history
├── code/                           # Analysis scripts (empty)
└── sub-XXX/                        # Subject directories
    └── eeg/
        ├── sub-XXX_task-sleep_eeg.set     # EEGLAB format (main file)
        ├── sub-XXX_task-sleep_eeg.fdt     # Binary EEG data
        ├── sub-XXX_task-sleep_eeg.json    # Recording metadata
        └── sub-XXX_task-sleep_channels.tsv # Channel names
```

### Key Files

**`.set` file (EEGLAB format):**
- Contains EEG data + metadata
- Includes `EEG.VisualHypnogram` (sleep stage labels)
- Includes `EEG.SpectralScoring` (automatic scoring, not used here)
- Loaded with `mne.io.read_raw_eeglab()`

**`.fdt` file:**
- Binary data file (linked from .set)
- Contains raw EEG time series

**`.json` file:**
- BIDS-compatible metadata
- Sampling rate, channel count, duration

---

## Data Loading Pipeline

### 1. Load Raw Data
```python
import mne
from scipy.io import loadmat

# Load EEG data
raw = mne.io.read_raw_eeglab('sub-101_task-sleep_eeg.set', preload=True)
data = raw.get_data()  # Shape: (3 channels, ~15M samples for 8 hours)
sfreq = raw.info['sfreq']  # 500 Hz

# Load labels
mat_data = loadmat('sub-101_task-sleep_eeg.set', simplify_cells=True)
labels = mat_data['VisualHypnogram'].flatten()  # Shape: (~1000 epochs,)
```

### 2. Preprocessing
```python
# 1. Bandpass filter: 0.5-50 Hz (remove DC drift and high-freq noise)
# 2. Notch filter: 60 Hz (remove power line noise)
# 3. Robust normalization: (data - median) / IQR per channel
# 4. Segment into 30-second epochs
# 5. Remove label=0 (unknown/movement)
# 6. Artifact rejection: remove epochs with extreme amplitudes
```

After preprocessing:
- **Input shape:** (n_epochs, 3 channels, 15,000 samples) per subject
- **Labels shape:** (n_epochs,) with values 0-4
- **Typical:** ~900-1000 epochs per subject after cleaning

### 3. Feature Extraction
For each 30-second epoch (15,000 samples):
- Extract **57 features per channel** (3 channels = 171 features)
- Extract **6 inter-channel coherence features**
- Add temporal context from ±2 neighboring epochs
- **Total: 2,652 features per epoch** (with improved version)

---

## Class Distribution

### Typical Distribution (Before SMOTE)
```
Wake (0):  ~15-20% of epochs
REM (1):   ~20-25% of epochs
N1 (2):    ~5-10% of epochs   ⚠️ Minority class!
N2 (3):    ~40-50% of epochs  ✅ Majority class
N3 (4):    ~10-20% of epochs
```

### Challenges

**N1 Stage (Critical Problem):**
- Very few samples (~5-10% of data)
- Most difficult to classify
- Often confused with Wake or REM
- **Original F1-score: 0.05** (essentially failing)
- **After SMOTE: Expected F1 > 0.60**

**Class Imbalance:**
- N2 dominates (40-50%)
- N1 is severely underrepresented
- Solution: **SMOTE** balances all classes

---

## Data Quality

### Signal Characteristics

**Good Signals:**
- Clear slow waves in N3 (0.5-2 Hz, high amplitude)
- Sleep spindles in N2 (11-16 Hz, 0.5-2s bursts)
- Alpha waves in Wake (8-13 Hz)
- Theta activity in REM (4-8 Hz)

**Artifacts:**
- Eye movements (strong in FP1/FP2)
- Muscle artifacts (high frequency)
- Movement (causes label=0, excluded)
- Electrode impedance changes

### Preprocessing Removes:
- ~5-10% of epochs due to extreme amplitudes
- All label=0 epochs (movement/unknown)
- High/low frequency noise via filtering

---

## Dataset Limitations

1. **Forehead-only EEG:**
   - Missing posterior channels (O1, O2 for alpha waves)
   - Missing central channels (C3, C4 for sleep spindles)
   - More susceptible to frontal artifacts

2. **Small Sample Size:**
   - Only 19 subjects
   - ~18,000 total epochs across all subjects
   - Limited demographic diversity (young adults only)

3. **N1 Stage Underrepresentation:**
   - Very few N1 epochs
   - Makes N1 classification challenging
   - Requires SMOTE or other balancing techniques

4. **Young, Healthy Adults Only:**
   - Age 19-29 years
   - No sleep disorders
   - Results may not generalize to older adults or clinical populations

---

## Expected Performance Benchmarks

### Literature Benchmarks (Forehead EEG)

**Onton et al. (2024) - Original Paper:**
- Used spectral scoring (automated)
- Overall agreement with PSG: ~70-80%
- N1 particularly challenging

**Traditional ML on Similar Datasets:**
- Wake: 70-85% accuracy
- REM: 75-85% accuracy
- N1: 30-60% accuracy (hardest!)
- N2: 80-90% accuracy (easiest)
- N3: 75-85% accuracy

### Our Implementation Goals

**Before Improvements:**
- Overall: 64.5% accuracy
- N1: 4.8% F1-score ❌

**After Improvements (Expected):**
- Overall: **85-90% accuracy** ✅
- N1: **60%+ F1-score** ✅
- Wake: 75%+ F1
- REM: 80%+ F1
- N2: 85%+ F1
- N3: 85%+ F1

---

## Comparison with Full PSG

**Full Polysomnography (Gold Standard):**
- 33+ channels (EEG, EOG, EMG, ECG, respiratory)
- Scalp coverage (frontal, central, parietal, occipital)
- Inter-rater agreement: 80-90%
- State-of-the-art deep learning: 85-90% accuracy

**Forehead-Only Patch (This Dataset):**
- 3 channels (forehead EEG only)
- Limited scalp coverage
- Expected accuracy: 75-85% (challenging task!)
- Our goal: **85%+** with improved features and SMOTE

**Key Difference:**
Forehead EEG is a **much harder task** than full PSG, so 85%+ accuracy is actually excellent performance!

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{onton2024validation,
  title={Validation of spectral sleep scoring with polysomnography using forehead EEG device},
  author={Onton, Julie A and Simon, Kristin C and Morehouse, Adam B and Shuster, Arielle E and Zhang, Jiawei and Pe{\~n}a, Alexandra A and Mednick, Sara C},
  journal={Frontiers in Sleep},
  volume={3},
  pages={1349537},
  year={2024},
  publisher={Frontiers Media SA},
  doi={10.3389/frsle.2024.1349537}
}
```

---

## Additional Resources

- **OpenNeuro Dataset:** https://openneuro.org/datasets/ds004745
- **AASM Sleep Scoring Manual:** American Academy of Sleep Medicine (2007+)
- **EEGLAB Documentation:** https://sccn.ucsd.edu/eeglab/
- **MNE-Python:** https://mne.tools/

---

**Last Updated:** 2026-01-31
