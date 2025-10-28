# Soft-Proprioception in the Frequency Domain via Tube Resonance

Code accompanying our paper on **acoustic proprioception for soft robots**.
This toolkit provides real-time spectral acquisition, peak/dip detection, machine-learning classification, trace similarity, and theory–data matching.

---

## Repository Contents

* **`Record_Live_FFT_Data.py`** — GUI for live FFT from ESP32D microphone stream; baseline/normalize; snapshot averaging; master CSV export.
* **`Find_Peaks_And_Dips.py`** — Post-process master CSV to detect top-K spectral peaks and dips (optional parabolic refinement).
* **`classifiers.py`** — Train/Test/Live scikit-learn classifier on normalized spectra; includes pose display and optional MP4 recording.
* **`compare_FFT_traces.py`** — Dynamic-programming alignment of peak/dip frequency lists for open-cut ↔ baseline similarity scoring.
* **`Compare_to_physical_models.py`** — Generate tube-harmonic theory (closed–closed / closed–open), detect dips in FFT CSVs, and match dips to theory.

---

## Installation

```bash
# Core
pip install numpy scipy pandas

# GUI + serial + plotting
pip install PyQt5 pyqtgraph pyserial matplotlib

# ML + model I/O
pip install scikit-learn joblib

# Optional (video recording for classifiers pose display)
pip install opencv-python
```

---

## Quick Start

1. **Collect live FFT data**

   ```bash
   python3 Record_Live_FFT_Data.py
   ```

   * Update `PORT` if needed; wait for `STREAM_READY`.
   * Baseline → Measure → take snapshots → **Next (Save Position)**.
   * Data saved as master CSV (10 Hz grid, 0–8000 Hz):
     `RunID, BaselineID, Label, 0Hz, 10Hz, …, 8000Hz`.

2. **Detect peaks and dips**

   ```bash
   python3 Find_Peaks_And_Dips.py
   ```

   * Input: master CSV (`MASTER_CSV_PATH`).
   * Output: `detected_peaks_dips.csv` (semicolon-separated lists per row).

3. **Train/Test/Live classification**

   ```bash
   python3 classifiers.py
   ```

   * **Train:** select CSV, choose model/PCA, train and save.
   * **Test:** evaluate metrics + confusion matrix.
   * **Live:** connect serial, capture baseline, view predictions, optional pose display.

4. **Compare traces**

   ```bash
   python3 compare_FFT_traces.py
   ```

   * Input: `detected_peaks_dips.csv`.
   * Output: `similarity_results.csv` with per-baseline error scores and top matches.

5. **Match to physical models**

   ```bash
   python3 Compare_to_physical_models.py
   ```

   * Builds theoretical harmonics, detects dips from FFT CSV, and reports the best-fitting model.
   * Outputs: `dips_detected.csv`, `dip_matches_vs_theory.csv`.

---

## Script Summaries

### `Record_Live_FFT_Data.py`

* Real-time FFT visualization (0–8 kHz).
* Snapshot averaging, baseline locking, normalized spectra.
* Automated sequencing; CSV export with 801 bins (10 Hz spacing).

### `Find_Peaks_And_Dips.py`

* Detects prominent peaks and dips using distance/width thresholds.
* Optional sub-bin frequency refinement (quadratic interpolation).
* Produces compact frequency-list CSVs for ML or comparison tasks.

### `classifiers.py`

* GUI for training, testing, and live prediction with scikit-learn.
* Models: Logistic Regression, Linear SVC, Random Forest.
* Optional PCA (retain 99 % variance).
* Pose display window visualizes 3×3 path (e.g., “1-3-1”); optional MP4 recording.

### `compare_FFT_traces.py`

* Aligns detected frequency lists via dynamic programming:

  * Match cost = |Δf|, gap penalty (Hz) for unmatched features.
* Ranks baselines by similarity; supports open↔open comparisons.
* Pure Python (no extra dependencies).

### `Compare_to_physical_models.py`

* Generates theoretical harmonic frequencies for tubes:

  * **Closed–Closed:** fₙ = n · v / (2 L)
  * **Closed–Open:** fₙ = (2n − 1) · v / (4 L)
* Optionally estimates effective sound speed using spacing:
  v_eff = 2 · L_ref · Δf
* “Slit-aware” CC mode filters modes with strong displacement at the slit.
* Detects dips, matches them to predicted harmonics within tolerance, and reports the best fit.

---

## Data Format

All processing assumes the wide “collector” format:

```
RunID, BaselineID, Label, 0Hz, 10Hz, 20Hz, …, 8000Hz
```

* 801 frequency bins (0–8 kHz, 10 Hz spacing).
* Label can be numeric (“131”) or delimited (“1-3-1”).

---

## License

Academic and research use only.
For other use, please contact the authors.
