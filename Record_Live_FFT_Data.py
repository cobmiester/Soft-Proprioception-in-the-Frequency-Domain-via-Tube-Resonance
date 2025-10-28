#!/usr/bin/env python3
import sys
import os
import csv
from collections import deque
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui

import numpy as np
import serial

import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d

# -----------------------------
# User-configurable parameters
# -----------------------------
PORT = '/dev/cu.usbserial-0001'   # Set your serial port
BAUD = 1000000
FS = 16000                        # Sampling rate (Hz)
BLOCK = 1024                      # Samples pulled from serial per read
FFT_SIZE = 32768                  # FFT length (power of 2 recommended)
UPDATE_MS = 50                    # GUI refresh rate (ms)
SMOOTHING_SIGMA = 60              # Gaussian smoothing (points in frequency bins)
ALPHA = 0.3                       # Exponential smoothing (0..1); 0 -> sticky, 1 -> raw
BYTES_PER_SAMPLE = 4              # '<i4' from device (little-endian 32-bit int)

# Save grid: 0..8000 in 10 Hz steps => 801 points
SAVE_FMAX = 8000
SAVE_DF = 10

# -----------------------------
# New: run-name automation & auto-baseline
# -----------------------------
RUN_SEQUENCE = ["131", "123", "313", "322", "121", "113", "133", "231", "232"]
USE_RUN_SEQUENCE = True
AUTO_BASELINE_DELAY_S = 2.0  # seconds (after Next)

# -----------------------------
# Worker that reads from serial
# -----------------------------
class SerialReader(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        ser = serial.Serial(PORT, BAUD, timeout=1)
        # Wait for device to announce readiness if applicable
        try:
            while True:
                line = ser.readline().decode(errors='ignore')
                if 'STREAM_READY' in line or line == '':
                    break
        except Exception:
            pass

        buf = bytearray(BLOCK * BYTES_PER_SAMPLE)
        while True:
            n = ser.readinto(buf)
            if n < len(buf):
                continue
            raw = np.frombuffer(buf, dtype='<i4')
            # Convert to float in [-1,1) assuming full-scale int32
            samples = raw.astype(np.float32) / np.float32(2**31)
            self.data_ready.emit(samples)

# -----------------------------
# Main application
# -----------------------------
class SpectrumApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Composite Resonator – Baseline/Normalize/Snapshot Averager")
        self.resize(1200, 700)

        # Status bar
        self.status = self.statusBar()

        # Central plot
        self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)
        self.plot.setLabel('bottom', 'Frequency', units='Hz')
        self.plot.setLabel('left', 'Magnitude', units='dB')
        self.plot.setXRange(0, SAVE_FMAX)
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        # Curves
        self.live_curve = self.plot.plot(pen=pg.mkPen('y', width=1.5), name='Live FFT')
        self.baseline_curve = self.plot.plot(pen=pg.mkPen('c', width=2), name='Baseline Avg')
        self.avg_curve = self.plot.plot(pen=pg.mkPen('m', width=2), name='Snapshot Average (Measure)')
        self.norm_live_curve = self.plot.plot(pen=pg.mkPen('g', width=2), name='Live Normalized')

        # Data buffers
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1/FS)   # 0..Fs/2
        self.buffer = deque(maxlen=FFT_SIZE)
        self.prev_spectrum = None                      # for exponential smoothing

        # Phase management: 'baseline' | 'measure' | 'recal'
        self.phase = 'baseline'
        self.baseline_snapshots = []                   # list of spectra (dB)
        self.measure_snapshots = []                    # list of spectra (dB)
        self.baseline_avg = None                       # dB spectrum
        self.measure_avg = None                        # dB spectrum
        self.norm_avg = None                           # dB (measure - baseline)

        # Recalibration helpers
        self._baseline_backup = None                   # used when cancelling recalibration

        # Run/Baseline IDs for logging
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.baseline_id = 0                           # increments whenever a baseline is locked/applied

        # Multi-position CSV aggregation
        self.master_csv_path = None
        self.hole_index = 1

        # New: sequence index
        self.seq_idx = 0

        # Controls / UI
        self._build_toolbar()
        self._apply_sequence_label()  # initial label
        self._update_phase_labels()

        # Serial reader
        self.reader = SerialReader()
        self.reader.data_ready.connect(self.on_samples)
        self.reader.start()

        # UI timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw)
        self.timer.start(UPDATE_MS)

    def _build_toolbar(self):
        tb = self.addToolBar("Controls")

        # Master CSV filename (applies to multi-position saves via Next)
        self.file_edit = QtWidgets.QLineEdit(self)
        self.file_edit.setPlaceholderText("master filename (e.g., training.csv)")
        self.file_edit.setFixedWidth(260)
        tb.addWidget(self.file_edit)

        # Position label (use for actuator pose/label)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Position label:"))
        self.hole_edit = QtWidgets.QLineEdit(self)
        self.hole_edit.setFixedWidth(160)
        tb.addWidget(self.hole_edit)

        tb.addSeparator()
        self.phase_label = QtWidgets.QLabel("")
        tb.addWidget(self.phase_label)

        # --- Baseline phase actions ---
        tb.addSeparator()
        self.lock_btn = self._make_action(
            tb, "Lock Baseline → Measure", self.lock_baseline, "Ctrl+L"
        )

        # --- Recalibration actions (enabled state depends on phase) ---
        tb.addSeparator()
        self.recal_start_btn = self._make_action(
            tb, "Start Recalibration", self.start_recalibration, "Ctrl+R",
            tooltip="Switch to 'Recalibrate Baseline' without losing measure snapshots."
        )

        self.recal_apply_btn = self._make_action(
            tb, "Apply Recalibration", self.apply_recalibration, "Ctrl+Shift+R",
            tooltip="Lock new baseline and return to measurement."
        )

        self.recal_cancel_btn = self._make_action(
            tb, "Cancel Recalibration", self.cancel_recalibration, "Ctrl+.",
            tooltip="Discard recalibration baseline and return to measurement."
        )

        # Snapshot (phase-aware)
        tb.addSeparator()
        self._make_action(
            tb, "Snapshot", self.take_snapshot, "Ctrl+K",
            tooltip="In Baseline/Recal, averages baseline; in Measure, averages measurement."
        )

        # Clear (phase-aware)
        self._make_action(
            tb, "Clear Snapshots", self.clear_snapshots, "Ctrl+D"
        )

        # Next (save current position)
        tb.addSeparator()
        self._make_action(
            tb, "Next (Save Position)", self.save_current_hole_and_reset, "Ctrl+N",
            tooltip="Save normalized avg for current label to master CSV, then auto-baseline after a short delay."
        )

        # One-off save (kept from original)
        tb.addSeparator()
        self._make_action(
            tb, "Save Averaged CSV (one-off)", self.save_csv_oneoff, "Ctrl+Shift+S"
        )

        # Finish
        tb.addSeparator()
        self._make_action(
            tb, "Finish", self.finish_session, "Ctrl+Q"
        )

        # Smoothing control
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Smoothing σ:"))
        self.sigma_spin = QtWidgets.QSpinBox(self)
        self.sigma_spin.setRange(0, 500)
        self.sigma_spin.setValue(SMOOTHING_SIGMA)
        tb.addWidget(self.sigma_spin)

        # Alpha control
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Live α:"))
        self.alpha_spin = QtWidgets.QDoubleSpinBox(self)
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(ALPHA)
        tb.addWidget(self.alpha_spin)

        # Snapshot counters
        tb.addSeparator()
        self.count_label = QtWidgets.QLabel("Baseline snaps: 0 | Measure snaps: 0")
        tb.addWidget(self.count_label)

        # Initialize action enable/disable
        self._update_action_states()

    def _make_action(self, tb, label, slot, shortcut, tooltip=None):
        text = f"{label} [{shortcut}]"
        act = QtWidgets.QAction(text, self)
        if tooltip:
            act.setToolTip(tooltip)
        act.triggered.connect(slot)
        act.setShortcut(QtGui.QKeySequence(shortcut))
        act.setShortcutContext(QtCore.Qt.ApplicationShortcut)  # works regardless of focus
        tb.addAction(act)
        return act

    def _update_action_states(self):
        # Enable lock only in initial baseline phase
        self.lock_btn.setEnabled(self.phase == 'baseline')
        # Recal controls: start allowed only in measure; apply/cancel only in recal
        self.recal_start_btn.setEnabled(self.phase == 'measure')
        self.recal_apply_btn.setEnabled(self.phase == 'recal')
        self.recal_cancel_btn.setEnabled(self.phase == 'recal')

    def _update_phase_labels(self):
        if self.phase == 'baseline':
            self.phase_label.setText("<b>Phase:</b> Baseline (close actuators)")
        elif self.phase == 'measure':
            self.phase_label.setText("<b>Phase:</b> Measure (set target pose)")
        else:
            self.phase_label.setText("<b>Phase:</b> Recalibrate Baseline (close actuators)")

        self.count_label.setText(
            f"Baseline snaps: {len(self.baseline_snapshots)} | Measure snaps: {len(self.measure_snapshots)}"
        )
        self._update_action_states()

    # ----- New: sequence label helper -----
    def _apply_sequence_label(self):
        """Fill the label box from RUN_SEQUENCE if enabled, else fallback to 'Position N'."""
        label = None
        if USE_RUN_SEQUENCE and 0 <= self.seq_idx < len(RUN_SEQUENCE):
            label = RUN_SEQUENCE[self.seq_idx]
        self.hole_edit.setText(label if label else f"Position {self.hole_index}")

    # ----- Callbacks -----
    def on_samples(self, samples: np.ndarray):
        self.buffer.extend(samples)

    # ----- DSP -----
    def compute_spectrum_db(self, data_buf) -> np.ndarray:
        """Compute smoothed, dB-scaled magnitude spectrum of current buffer."""
        data = np.array(data_buf, dtype=np.float32)
        if data.size < FFT_SIZE:
            return None

        # Detrend -> window -> FFT
        data = data[-FFT_SIZE:]
        data -= np.mean(data)
        window = np.hanning(FFT_SIZE).astype(np.float32)
        data_win = data * window

        X = np.fft.rfft(data_win)
        # amplitude correction for window energy
        mags = np.abs(X) / np.sum(window)

        # dBFS with a small floor
        mags_db = 20.0 * np.log10(mags + 1e-12)

        # Exponential smoothing on the dB trace for display stability
        alpha = float(self.alpha_spin.value())
        if self.prev_spectrum is None:
            spec = mags_db
        else:
            spec = alpha * mags_db + (1.0 - alpha) * self.prev_spectrum
        self.prev_spectrum = spec

        # Optional Gaussian smoothing along frequency axis
        sigma = int(self.sigma_spin.value())
        if sigma > 0:
            spec = gaussian_filter1d(spec, sigma=sigma)

        return spec

    def normalize_db(self, spec_db: np.ndarray, baseline_db: np.ndarray) -> np.ndarray:
        """Return normalized spectrum in dB (measure - baseline)."""
        if spec_db is None or baseline_db is None:
            return None
        n = min(len(spec_db), len(baseline_db))
        return spec_db[:n] - baseline_db[:n]

    # ----- UI updates -----
    def redraw(self):
        if len(self.buffer) < FFT_SIZE:
            return
        spec_db = self.compute_spectrum_db(self.buffer)
        if spec_db is None:
            return

        self.live_curve.setData(self.freqs, spec_db)

        # Baseline avg curve
        if self.baseline_avg is not None:
            self.baseline_curve.setData(self.freqs, self.baseline_avg)
        else:
            self.baseline_curve.setData([], [])

        # Measure average curve
        if self.measure_avg is not None:
            self.avg_curve.setData(self.freqs, self.measure_avg)
        else:
            self.avg_curve.setData([], [])

        # Normalized live (whenever a baseline exists)
        if self.baseline_avg is not None:
            norm_live = self.normalize_db(spec_db, self.baseline_avg)
            self.norm_live_curve.setData(self.freqs[:len(norm_live)], norm_live)
        else:
            self.norm_live_curve.setData([], [])

    # ----- Snapshot logic -----
    def take_snapshot(self):
        if len(self.buffer) < FFT_SIZE:
            QtWidgets.QMessageBox.warning(self, "Not enough data",
                                          "Need at least FFT_SIZE samples to take a snapshot.")
            return
        spec_db = self.compute_spectrum_db(self.buffer)
        if spec_db is None:
            return

        if self.phase in ('baseline', 'recal'):
            # (Re)building a baseline by averaging snapshots
            self.baseline_snapshots.append(spec_db.copy())
            self.baseline_avg = np.mean(np.vstack(self.baseline_snapshots), axis=0)
        else:  # measure
            if self.baseline_avg is None:
                QtWidgets.QMessageBox.warning(self, "No baseline",
                                              "Lock or apply a baseline first, then take measure snapshots.")
                return
            self.measure_snapshots.append(spec_db.copy())
            self.measure_avg = np.mean(np.vstack(self.measure_snapshots), axis=0)
            self.norm_avg = self.normalize_db(self.measure_avg, self.baseline_avg)

        self._update_phase_labels()
        self.redraw()

    def clear_snapshots(self):
        if self.phase in ('baseline', 'recal'):
            self.baseline_snapshots.clear()
            self.baseline_avg = None
        else:
            self.measure_snapshots.clear()
            self.measure_avg = None
            self.norm_avg = None
        self._update_phase_labels()
        self.redraw()

    def lock_baseline(self):
        """Lock the current baseline (from baseline snapshots or current live) and enter measurement."""
        if self.phase != 'baseline':
            return
        # If no baseline snapshots, allow using current live spectrum as a quick baseline
        if self.baseline_avg is None:
            if len(self.buffer) < FFT_SIZE:
                self.status.showMessage("Auto-baseline: not enough data yet.", 3000)
                return
            spec_db = self.compute_spectrum_db(self.buffer)
            if spec_db is None:
                self.status.showMessage("Auto-baseline: compute failed.", 3000)
                return
            self.baseline_avg = spec_db.copy()
            self.baseline_snapshots = [spec_db.copy()]

        # Baseline locked -> increment ID
        self.baseline_id += 1

        # Switch to measurement phase
        self.phase = 'measure'
        self.measure_snapshots.clear()
        self.measure_avg = None
        self.norm_avg = None
        self._update_phase_labels()
        self.redraw()
        self.status.showMessage("Baseline locked → Measure", 2000)

    # ----- Recalibration workflow -----
    def start_recalibration(self):
        """Enter recalibration phase without losing measure snapshots."""
        if self.phase != 'measure':
            QtWidgets.QMessageBox.information(self, "Recalibration",
                                              "You can only start recalibration while measuring.")
            return
        # Backup current baseline in case of cancel
        self._baseline_backup = None if self.baseline_avg is None else self.baseline_avg.copy()

        # Clear baseline-building buffers; user can take new baseline snapshots
        self.baseline_snapshots.clear()
        self.baseline_avg = None

        self.phase = 'recal'
        self._update_phase_labels()
        self.redraw()

    def apply_recalibration(self):
        """Apply the recalibrated baseline (from snapshots or live) and return to measurement."""
        if self.phase != 'recal':
            return

        # If user didn't snapshot, allow quick baseline from live
        if self.baseline_avg is None:
            if len(self.buffer) < FFT_SIZE:
                QtWidgets.QMessageBox.warning(self, "No baseline available",
                                              "Take at least one recalibration snapshot or wait for live data.")
                return
            spec_db = self.compute_spectrum_db(self.buffer)
            if spec_db is None:
                return
            self.baseline_avg = spec_db.copy()
            self.baseline_snapshots = [spec_db.copy()]

        # Baseline changed -> increment ID
        self.baseline_id += 1

        # Recompute normalized average if we already had measurement snapshots
        if self.measure_avg is not None:
            self.norm_avg = self.normalize_db(self.measure_avg, self.baseline_avg)

        self.phase = 'measure'
        self._baseline_backup = None
        self._update_phase_labels()
        self.redraw()

    def cancel_recalibration(self):
        """Discard recalibration and restore previous baseline."""
        if self.phase != 'recal':
            return
        # Restore backup
        if self._baseline_backup is not None:
            self.baseline_avg = self._baseline_backup.copy()
        else:
            self.baseline_avg = None
        self.baseline_snapshots.clear()
        self._baseline_backup = None

        # Recompute normalization against restored baseline
        if self.measure_avg is not None and self.baseline_avg is not None:
            self.norm_avg = self.normalize_db(self.measure_avg, self.baseline_avg)
        else:
            self.norm_avg = None

        self.phase = 'measure'
        self._update_phase_labels()
        self.redraw()

    # ----- CSV helpers -----
    def _build_save_grid(self):
        f_save = np.arange(0, SAVE_FMAX + SAVE_DF, SAVE_DF, dtype=float)  # 0..8000 inclusive
        return f_save

    def _interp_to_grid(self, spec_db: np.ndarray, f_save: np.ndarray):
        return np.interp(f_save, self.freqs[:len(spec_db)], spec_db)

    def _ensure_master_path(self):
        # If a path is already set, ensure extension and return
        base = self.file_edit.text().strip()
        if not base and not self.master_csv_path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Choose/Name Master CSV", os.getcwd(), "CSV Files (*.csv)"
            )
            if not path:
                return None
            if not path.lower().endswith('.csv'):
                path += '.csv'
            self.master_csv_path = path
            self.file_edit.setText(os.path.basename(path))
            return self.master_csv_path

        if base and not self.master_csv_path:
            if not base.lower().endswith('.csv'):
                base = base + '.csv'
            self.master_csv_path = os.path.join(os.getcwd(), base)
        return self.master_csv_path

    def _append_row_to_master(self, label: str, norm_spec_on_grid: np.ndarray, f_save: np.ndarray):
        path = self._ensure_master_path()
        if path is None:
            return False

        file_exists = os.path.exists(path) and os.path.getsize(path) > 0

        try:
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if not file_exists:
                    # Header: RunID + BaselineID + Label + frequency bins
                    headers = ['RunID', 'BaselineID', 'Label'] + [f"{int(f)}Hz" for f in f_save]
                    w.writerow(headers)
                row = [self.run_id, self.baseline_id, label] + list(np.round(norm_spec_on_grid, 3))
                w.writerow(row)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not append to master CSV:\n{e}")
            return False

        return True

    # ----- Saving flows -----
    def save_current_hole_and_reset(self):
        """Save normalized averaged spectrum for current position to the master CSV, then restart at baseline,
        auto-fill next label from RUN_SEQUENCE, and auto-lock baseline after a short delay."""
        if self.phase != 'measure':
            QtWidgets.QMessageBox.warning(self, "Not in Measure phase",
                                          "Lock a baseline, take measure snapshots, then click Next.")
            return
        if self.baseline_avg is None:
            QtWidgets.QMessageBox.warning(self, "No baseline",
                                          "Lock a baseline first.")
            return
        # Use measure_avg if available, else allow saving the current live normalized
        if self.measure_avg is None:
            if len(self.buffer) < FFT_SIZE:
                QtWidgets.QMessageBox.warning(self, "Nothing to save",
                                              "Take at least one measure snapshot or wait for live data.")
                return
            live_db = self.compute_spectrum_db(self.buffer)
            self.norm_avg = self.normalize_db(live_db, self.baseline_avg)
        else:
            self.norm_avg = self.normalize_db(self.measure_avg, self.baseline_avg)

        if self.norm_avg is None:
            QtWidgets.QMessageBox.warning(self, "Normalization error",
                                          "Could not compute normalized spectrum.")
            return

        f_save = self._build_save_grid()
        norm_on_grid = self._interp_to_grid(self.norm_avg, f_save)

        label = self.hole_edit.text().strip() or f"Position {self.hole_index}"
        ok = self._append_row_to_master(label, norm_on_grid, f_save)
        if not ok:
            return

        QtWidgets.QMessageBox.information(self, "Saved",
                                          f"Saved normalized spectrum for '{label}' to:\n{self.master_csv_path}")

        # Prepare for next position
        self.phase = 'baseline'
        self.baseline_snapshots.clear()
        self.baseline_avg = None
        self.measure_snapshots.clear()
        self.measure_avg = None
        self.norm_avg = None

        # Advance counters and auto-fill next label from sequence
        self.hole_index += 1
        self.seq_idx += 1
        self._apply_sequence_label()
        self._update_phase_labels()
        self.redraw()

        # Schedule auto-baseline lock
        delay_ms = int(max(0.0, AUTO_BASELINE_DELAY_S) * 1000)
        if delay_ms > 0:
            self.status.showMessage(f"Arming auto-baseline in {AUTO_BASELINE_DELAY_S:.1f}s…", delay_ms)
            QtCore.QTimer.singleShot(delay_ms, self._auto_lock_baseline)

    def _auto_lock_baseline(self):
        """Called after Next; tries to lock baseline using current live spectrum."""
        if self.phase != 'baseline':
            return  # user may have moved on manually
        self.lock_baseline()

    def save_csv_oneoff(self):
        """Save the *current averaged* (or live) spectrum to its own CSV (10 Hz grid)."""
        # Preference: measurement average -> baseline average -> live
        if self.measure_avg is not None:
            save_spec = self.measure_avg
            default_name = "measure_avg.csv"
        elif self.baseline_avg is not None:
            save_spec = self.baseline_avg
            default_name = "baseline_avg.csv"
        else:
            if len(self.buffer) < FFT_SIZE:
                QtWidgets.QMessageBox.warning(self, "Nothing to save",
                                              "Take at least one snapshot or wait for live data.")
                return
            save_spec = self.compute_spectrum_db(self.buffer)
            default_name = "live.csv"

        f_save = self._build_save_grid()
        spec_on_grid = self._interp_to_grid(save_spec, f_save)

        # Ask for a filename
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", os.path.join(os.getcwd(), default_name), "CSV Files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'

        headers = [f"{int(f)}Hz" for f in f_save]
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerow(np.round(spec_on_grid, 3))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not save CSV:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved {len(f_save)} points to:\n{path}")

    def finish_session(self):
        """Finish the session; the master CSV has been appended along the way."""
        msg = "Session finished.\n"
        if self.master_csv_path:
            msg += f"Master CSV: {self.master_csv_path}"
        else:
            msg += "No master CSV was created."
        QtWidgets.QMessageBox.information(self, "Done", msg)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumApp()
    win.show()
    sys.exit(app.exec_())
