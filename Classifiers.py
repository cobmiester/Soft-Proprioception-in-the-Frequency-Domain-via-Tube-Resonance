#!/usr/bin/env python3
"""
Soft-Robot Pose Classifier (with persistence + live pose display/recording)

- Train tab:
  * Save Model… / Load Model… buttons.
  * Auto-save trained model to ./models/CLASSIFIER_YYYYmmdd-HHMMSS.joblib
  * Remembers last paths across runs via QSettings.

- Live tab:
  * "Show Pose Display" opens a large window rendering a 3x3 pin board and the
    predicted woven path for labels like "1-3-1" (right→middle→left columns;
    1=bottom, 2=middle, 3=top).
  * "Start Recording" / "Stop Recording" records the pose display to MP4.

Pipeline persistence:
- We save the entire sklearn Pipeline (scaler/PCA/clf) AND the metadata it
  needs for live prediction:
    model_metadata = {
        "class_names": [...],
        "f_save": np.arange(0, SAVE_FMAX+SAVE_DF, SAVE_DF, dtype=float),  # 801 bins
        "save_fmax": SAVE_FMAX,
        "save_df": SAVE_DF,
    }
  This is attached on the pipeline as attribute "model_metadata" before saving.

CSV format expected (from the collector):
RunID, BaselineID, Label, 0Hz, 10Hz, ..., 8000Hz
"""

import sys
import os
import csv
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# DSP
from scipy.ndimage import gaussian_filter1d

# Serial (for live mode)
import serial

# ML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Persistence
import joblib

# Optional video recording (OpenCV). If not available, recording is disabled.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# -----------------------------
# User-configurable parameters
# -----------------------------
PORT = '/dev/cu.usbserial-0001'   # Serial port
BAUD = 1000000
FS = 16000                        # Sampling rate (Hz)
BLOCK = 1024                      # Samples pulled from serial per read
FFT_SIZE = 32768                  # FFT length (power of 2 recommended)
UPDATE_MS = 50                    # GUI refresh rate (ms)
SMOOTHING_SIGMA = 60              # Gaussian smoothing (points in frequency bins)
ALPHA = 0.3                       # Exponential smoothing for live display (0..1); 0 -> sticky, 1 -> raw
BYTES_PER_SAMPLE = 4              # '<i4' from device (little-endian 32-bit int)

# Save grid: 0..8000 in 10 Hz steps => 801 points
SAVE_FMAX = 8000
SAVE_DF = 10

ORG_NAME = "SoftRobotLab"
APP_NAME = "SoftPoseClassifier"

# -----------------------------
# Serial reader thread
# -----------------------------
class SerialReader(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(str)

    def __init__(self, port, baud, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.error.emit(f"Could not open serial port {self.port}: {e}")
            return

        # Optional: wait for device readiness
        try:
            while not self._stop:
                line = ser.readline().decode(errors='ignore')
                if 'STREAM_READY' in line or line == '':
                    break
        except Exception:
            pass

        buf = bytearray(BLOCK * BYTES_PER_SAMPLE)
        while not self._stop:
            try:
                n = ser.readinto(buf)
                if n < len(buf):
                    continue
                raw = np.frombuffer(buf, dtype='<i4')
                samples = raw.astype(np.float32) / np.float32(2**31)
                self.data_ready.emit(samples)
            except Exception as e:
                self.error.emit(f"Serial read error: {e}")
                break

        try:
            ser.close()
        except Exception:
            pass

# Matplotlib dialog for confusion matrix

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ConfMatDialog(QtWidgets.QDialog):
    def __init__(self, cm: np.ndarray, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        layout = QtWidgets.QVBoxLayout(self)
        fig = Figure(figsize=(6, 5), tight_layout=True)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        # Annotate counts
        thresh = cm.max() / 2 if cm.size else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        layout.addWidget(canvas)


# Pose display widget & window

class PoseCanvas(QtWidgets.QWidget):
    """
    Draws a 3x3 pin board and a polyline path for labels like "1-3-1".
    Convention:
      - Columns traverse RIGHT -> MIDDLE -> LEFT (in this order).
      - Digits map to rows: 1=bottom, 2=middle, 3=top.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 640)
        self._label = None
        self._confidence = None
        self._bg = QtGui.QColor("#0f1117")
        self._grid = QtGui.QColor("#3a3f58")
        self._pin = QtGui.QColor("#aab2d5")
        self._path = QtGui.QColor("#7bd88f")
        self._text = QtGui.QColor("#e5e9f0")
        self._font = QtGui.QFont("Arial", 20, QtGui.QFont.Bold)

    def set_prediction(self, label: str, confidence: str):
        self._label = label
        self._confidence = confidence
        self.update()

    def _parse_label(self, s: str):
        # Accept "1-3-1", "131", "1_3_1" etc.
        if not s:
            return None
        cleaned = ''.join(ch for ch in s if ch.isdigit())
        if len(cleaned) < 3:
            return None
        return [int(ch) for ch in cleaned[:3]]  # only 3 steps (R, M, L)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()

        # background
        p.fillRect(rect, self._bg)

        # square drawing area
        margin = int(min(rect.width(), rect.height()) * 0.08)
        board = QtCore.QRect(rect.left()+margin, rect.top()+margin,
                             rect.width()-2*margin, rect.height()-2*margin)

        # grid 3x3
        p.setPen(QtGui.QPen(self._grid, 2))
        for i in range(4):
            # verticals
            x = board.left() + i * board.width() / 3
            p.drawLine(int(x), board.top(), int(x), board.bottom())
            # horizontals
            y = board.top() + i * board.height() / 3
            p.drawLine(board.left(), int(y), board.right(), int(y))

        # pins
        p.setBrush(self._pin)
        pin_r = max(6, int(board.width() * 0.015))
        for ci in range(3):
            for ri in range(3):
                cx = board.left() + (ci + 0.5) * board.width() / 3
                cy = board.top() + (ri + 0.5) * board.height() / 3
                p.drawEllipse(QtCore.QPointF(cx, cy), pin_r, pin_r)

        # label/path
        label_text = self._label if self._label else "—"
        conf_text = f"{self._confidence}" if self._confidence else "—"

        # Draw path if valid prediction
        digits = self._parse_label(label_text)
        if digits and all(d in (1,2,3) for d in digits):
            # columns: right(0), middle(1), left(2) in screen space (so invert when painting)
            cols = [2, 1, 0]  # map sequence indices 0..2 to x grid columns
            points = []
            for step, row_digit in enumerate(digits[:3]):
                col = cols[step]
                row = 3 - row_digit  # 1(bottom)->2, 2(middle)->1, 3(top)->0
                x = board.left() + (col + 0.5) * board.width() / 3
                y = board.top() + (row + 0.5) * board.height() / 3
                points.append(QtCore.QPointF(x, y))

            p.setPen(QtGui.QPen(self._path, max(6, int(board.width()*0.02))))
            for i in range(len(points)-1):
                p.drawLine(points[i], points[i+1])

            # emphasize endpoints
            p.setBrush(self._path)
            end_r = max(7, int(board.width()*0.018))
            p.drawEllipse(points[0], end_r, end_r)
            p.drawEllipse(points[-1], end_r, end_r)

        # Title
        p.setFont(self._font)
        p.setPen(self._text)
        title = f"Live Pose: {label_text}   |   Confidence: {conf_text}"
        p.drawText(rect.adjusted(20, 10, -20, -rect.height()+60),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, title)

        p.end()

class PoseDisplayWindow(QtWidgets.QMainWindow):
    frame_captured = QtCore.pyqtSignal(int)  # emits frame count

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Soft Pipe Pose Display")
        self.resize(800, 800)
        self.canvas = PoseCanvas(self)
        self.setCentralWidget(self.canvas)

        # Recording state
        self._recording = False
        self._writer = None
        self._fps = 20
        self._frame_timer = QtCore.QTimer()
        self._frame_timer.timeout.connect(self._grab_frame)

        tb = self.addToolBar("Controls")
        self.rec_btn = QtWidgets.QAction("Start Recording", self)
        self.rec_btn.triggered.connect(self.toggle_recording)
        tb.addAction(self.rec_btn)

        self.stop_btn = QtWidgets.QAction("Stop Recording", self)
        self.stop_btn.setEnabled(False)
        self.stop_btn.triggered.connect(self.stop_recording)
        tb.addAction(self.stop_btn)

        self.status = self.statusBar()

        if not _HAS_CV2:
            self.status.showMessage("Recording disabled (OpenCV not installed).")

    def set_prediction(self, label: str, confidence: str):
        self.canvas.set_prediction(label, confidence)

    def toggle_recording(self):
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not _HAS_CV2:
            QtWidgets.QMessageBox.information(self, "Recording Unavailable",
                                              "Install OpenCV (cv2) to enable recording.")
            return
        # Pick output path
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = os.path.join(os.getcwd(), f"pose_{ts}.mp4")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save MP4",
                                                        default, "MP4 Video (*.mp4)")
        if not path:
            return

        # Initialize writer with current widget size
        size = self.canvas.size()
        w, h = int(size.width()), int(size.height())
        # Prefer H264, fall back to MJPG if unavailable
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
        if not writer.isOpened():
            QtWidgets.QMessageBox.warning(self, "Recorder Error",
                                          "Could not open video writer.")
            return

        self._writer = writer
        self._recording = True
        self.rec_btn.setText("Recording…")
        self.rec_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._frame_timer.start(int(1000 / self._fps))
        self.status.showMessage(f"Recording to: {path}")

    def _grab_frame(self):
        if not self._recording or self._writer is None:
            return
        # Render widget to image
        img = self.canvas.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(h * img.bytesPerLine())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, img.bytesPerLine()))
        frame = arr[:, :w*3].reshape((h, w, 3))
        # BGR for OpenCV
        frame_bgr = frame[:, :, ::-1]
        self._writer.write(frame_bgr)
        self.frame_captured.emit(1)

    def stop_recording(self):
        self._frame_timer.stop()
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._recording = False
        self.rec_btn.setText("Start Recording")
        self.rec_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage("Recording stopped.")

    def closeEvent(self, e):
        if self._recording:
            self.stop_recording()
        super().closeEvent(e)

# -----------------------------
# Main application
# -----------------------------
class ClassifierApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soft-Robot Pose Classifier – Train / Test / Live")
        self.resize(1400, 860)

        # Settings
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)

        # Central split: left controls, right plot + status
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        self.controls = QtWidgets.QTabWidget()
        h.addWidget(self.controls, 0)

        # Right side: plot + prediction readout
        right = QtWidgets.QWidget()
        h.addWidget(right, 1)
        rv = QtWidgets.QVBoxLayout(right)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Frequency', units='Hz')
        self.plot.setLabel('left', 'Magnitude', units='dB')
        self.plot.setXRange(0, SAVE_FMAX)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        rv.addWidget(self.plot, 1)

        self.live_curve = self.plot.plot(pen=pg.mkPen('y', width=1.5), name='Live FFT (dB)')
        self.baseline_curve = self.plot.plot(pen=pg.mkPen('c', width=2), name='Baseline (dB)')
        self.norm_live_curve = self.plot.plot(pen=pg.mkPen('g', width=2), name='Live Normalized (dB)')

        pred_box = QtWidgets.QGroupBox("Live Prediction")
        rv.addWidget(pred_box, 0)
        gl = QtWidgets.QGridLayout(pred_box)
        self.pred_label = QtWidgets.QLabel("—")
        self.pred_prob = QtWidgets.QLabel("—")
        gl.addWidget(QtWidgets.QLabel("Predicted class:"), 0, 0)
        gl.addWidget(self.pred_label, 0, 1)
        gl.addWidget(QtWidgets.QLabel("Confidence:"), 1, 0)
        gl.addWidget(self.pred_prob, 1, 1)

        # Pose display window
        self.pose_win = None

        # Buffers and DSP state
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1/FS)   # 0..Fs/2
        self.buffer = deque(maxlen=FFT_SIZE)
        self.prev_spectrum = None
        self.baseline_db = None
        self.sigma = SMOOTHING_SIGMA
        self.alpha = ALPHA

        # ML state
        self.model: Pipeline = None
        self.class_names = []
        self.pipeline_choice = 'LogisticRegression'  # default

        # Serial thread
        self.reader = None

        # Timer for redraw
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw)
        self.timer.start(UPDATE_MS)

        # Build tabs
        self._build_train_tab()
        self._build_test_tab()
        self._build_live_tab()

        # Bottom status
        self.status = self.statusBar()
        self.status.showMessage("Load training CSV or load a saved model to start.")

        # Restore last paths
        self._restore_last_paths()

    # ----------------- Tabs -----------------
    def _build_train_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # File picker
        fh = QtWidgets.QHBoxLayout()
        self.train_path_edit = QtWidgets.QLineEdit()
        self.train_browse_btn = QtWidgets.QPushButton("Browse…")
        self.train_browse_btn.clicked.connect(self.pick_train_csv)
        fh.addWidget(QtWidgets.QLabel("Training CSV:"))
        fh.addWidget(self.train_path_edit, 1)
        fh.addWidget(self.train_browse_btn)
        v.addLayout(fh)

        # Model chooser
        mh = QtWidgets.QHBoxLayout()
        mh.addWidget(QtWidgets.QLabel("Model:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["LogisticRegression (L2)", "LinearSVC", "RandomForest"])
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        mh.addWidget(self.model_combo)
        # Optional PCA
        self.pca_check = QtWidgets.QCheckBox("PCA (keep 99% var)")
        self.pca_check.setChecked(False)
        mh.addWidget(self.pca_check)

        mh.addStretch(1)
        v.addLayout(mh)

        # Train/Save/Load buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        btn_row.addWidget(self.train_btn)

        self.save_model_btn = QtWidgets.QPushButton("Save Model…")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        btn_row.addWidget(self.save_model_btn)

        self.load_model_btn = QtWidgets.QPushButton("Load Model…")
        self.load_model_btn.clicked.connect(self.load_model)
        btn_row.addWidget(self.load_model_btn)

        v.addLayout(btn_row)

        # Log window
        self.train_log = QtWidgets.QPlainTextEdit()
        self.train_log.setReadOnly(True)
        v.addWidget(self.train_log, 1)

        self.controls.addTab(w, "1) Train")

    def _build_test_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # File picker
        fh = QtWidgets.QHBoxLayout()
        self.test_path_edit = QtWidgets.QLineEdit()
        self.test_browse_btn = QtWidgets.QPushButton("Browse…")
        self.test_browse_btn.clicked.connect(self.pick_test_csv)
        fh.addWidget(QtWidgets.QLabel("Test CSV:"))
        fh.addWidget(self.test_path_edit, 1)
        fh.addWidget(self.test_browse_btn)
        v.addLayout(fh)

        # Run test
        self.test_btn = QtWidgets.QPushButton("Evaluate on Test CSV")
        self.test_btn.clicked.connect(self.run_test)
        v.addWidget(self.test_btn)

        # Output
        self.test_log = QtWidgets.QPlainTextEdit()
        self.test_log.setReadOnly(True)
        v.addWidget(self.test_log, 1)

        self.controls.addTab(w, "2) Offline Test")

    def _build_live_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # Serial config
        sh = QtWidgets.QHBoxLayout()
        self.port_edit = QtWidgets.QLineEdit(PORT)
        self.baud_edit = QtWidgets.QLineEdit(str(BAUD))
        sh.addWidget(QtWidgets.QLabel("Port:"))
        sh.addWidget(self.port_edit)
        sh.addWidget(QtWidgets.QLabel("Baud:"))
        sh.addWidget(self.baud_edit)
        v.addLayout(sh)

        # Controls
        bh = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_serial)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)

        self.capture_baseline_btn = QtWidgets.QPushButton("Capture Baseline")
        self.capture_baseline_btn.clicked.connect(self.capture_baseline)
        self.capture_baseline_btn.setEnabled(False)

        self.pose_display_btn = QtWidgets.QPushButton("Show Pose Display")
        self.pose_display_btn.setCheckable(True)
        self.pose_display_btn.toggled.connect(self.toggle_pose_window)

        self.start_rec_btn = QtWidgets.QPushButton("Start Recording")
        self.start_rec_btn.clicked.connect(self.start_pose_recording)
        self.stop_rec_btn = QtWidgets.QPushButton("Stop Recording")
        self.stop_rec_btn.clicked.connect(self.stop_pose_recording)
        self.stop_rec_btn.setEnabled(False)

        bh.addWidget(self.connect_btn)
        bh.addWidget(self.disconnect_btn)
        bh.addWidget(self.capture_baseline_btn)
        bh.addWidget(self.pose_display_btn)
        bh.addWidget(self.start_rec_btn)
        bh.addWidget(self.stop_rec_btn)
        v.addLayout(bh)

        # DSP smoothing controls
        dh = QtWidgets.QHBoxLayout()
        self.sigma_spin = QtWidgets.QSpinBox()
        self.sigma_spin.setRange(0, 500)
        self.sigma_spin.setValue(SMOOTHING_SIGMA)
        self.sigma_spin.valueChanged.connect(lambda val: setattr(self, 'sigma', int(val)))
        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(ALPHA)
        self.alpha_spin.valueChanged.connect(lambda val: setattr(self, 'alpha', float(val)))

        dh.addWidget(QtWidgets.QLabel("Smoothing σ:")); dh.addWidget(self.sigma_spin)
        dh.addWidget(QtWidgets.QLabel("Live α:")); dh.addWidget(self.alpha_spin)
        dh.addStretch(1)
        v.addLayout(dh)

        # Info
        self.live_log = QtWidgets.QPlainTextEdit()
        self.live_log.setReadOnly(True)
        v.addWidget(self.live_log, 1)

        self.controls.addTab(w, "3) Live Predict")

    # ----------------- Utilities -----------------
    def _restore_last_paths(self):
        self.train_path_edit.setText(self.settings.value("train_csv", ""))
        self.test_path_edit.setText(self.settings.value("test_csv", ""))
        last_model = self.settings.value("last_model_path", "")
        if last_model:
            self.train_log.appendPlainText(f"Tip: last model path: {last_model}")

    def _remember_path(self, key, path):
        self.settings.setValue(key, path)

    def pick_train_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training CSV",
                                                        self.settings.value("train_dir", os.getcwd()),
                                                        "CSV Files (*.csv)")
        if path:
            self.train_path_edit.setText(path)
            self.train_log.appendPlainText(f"Selected training CSV: {path}")
            self._remember_path("train_csv", path)
            self._remember_path("train_dir", os.path.dirname(path))

    def pick_test_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test CSV",
                                                        self.settings.value("test_dir", os.getcwd()),
                                                        "CSV Files (*.csv)")
        if path:
            self.test_path_edit.setText(path)
            self.test_log.appendPlainText(f"Selected test CSV: {path}")
            self._remember_path("test_csv", path)
            self._remember_path("test_dir", os.path.dirname(path))

    def _model_changed(self, idx):
        self.pipeline_choice = self.model_combo.currentText()

    def _load_dataset(self, path):
        """Load CSV and return (X, y, freq_cols, df). y may be None if 'Label' missing."""
        df = pd.read_csv(path)
        # Find frequency columns (end with "Hz")
        freq_cols = [c for c in df.columns if c.endswith("Hz")]
        if not freq_cols:
            raise ValueError("No frequency columns found (expected headers like '0Hz', '10Hz', ...).")
        # Try to get labels
        y = df['Label'].astype(str) if 'Label' in df.columns else None
        X = df[freq_cols].to_numpy(dtype=np.float32)
        return X, y, freq_cols, df

    def _build_pipeline(self, n_features: int):
        """Create an sklearn Pipeline according to the UI selections."""
        steps = [('scaler', StandardScaler())]
        if self.pca_check.isChecked():
            steps.append(('pca', PCA(n_components=0.99, svd_solver='full', whiten=False)))
        choice = self.pipeline_choice
        if choice.startswith('LogisticRegression'):
            clf = LogisticRegression(max_iter=2000, multi_class='auto')
        elif choice.startswith('LinearSVC'):
            clf = LinearSVC(dual=False, max_iter=5000)
        else:
            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        steps.append(('clf', clf))
        pipe = Pipeline(steps)
        return pipe

    # ----------------- Train -----------------
    def train_model(self):
        path = self.train_path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "No training CSV", "Please select a training CSV.")
            return
        try:
            X, y, freq_cols, df = self._load_dataset(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Could not load training CSV:\n{e}")
            return
        if y is None:
            QtWidgets.QMessageBox.critical(self, "No labels", "Training CSV must include a 'Label' column.")
            return

        self.class_names = sorted(pd.unique(y))
        self.train_log.appendPlainText(f"Loaded {X.shape[0]} samples, {X.shape[1]} features.")
        self.train_log.appendPlainText(f"Classes: {self.class_names}")

        pipe = self._build_pipeline(X.shape[1])
        pipe.fit(X, y)

        # Attach metadata needed for live predictions
        f_save = np.arange(0, SAVE_FMAX + SAVE_DF, SAVE_DF, dtype=float)
        pipe.model_metadata = {
            "class_names": self.class_names,
            "f_save": f_save,
            "save_fmax": SAVE_FMAX,
            "save_df": SAVE_DF,
        }

        self.model = pipe
        self.save_model_btn.setEnabled(True)
        self.train_log.appendPlainText("Training complete.")
        self.status.showMessage(f"Model trained on {X.shape[0]} samples, {len(self.class_names)} classes.")

        # Auto-save snapshot
        try:
            os.makedirs("models", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            auto_path = os.path.join("models", f"MODEL_{ts}.joblib")
            joblib.dump(self.model, auto_path)
            self.train_log.appendPlainText(f"Auto-saved model → {auto_path}")
            self._remember_path("last_model_path", auto_path)
        except Exception as e:
            self.train_log.appendPlainText(f"[Auto-save warning] {e}")

    def save_model(self):
        if self.model is None:
            QtWidgets.QMessageBox.information(self, "No model", "Train or load a model first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Model",
                                                        self.settings.value("model_dir", os.getcwd()),
                                                        "Joblib Model (*.joblib)")
        if not path:
            return
        try:
            joblib.dump(self.model, path)
            self.train_log.appendPlainText(f"Model saved to: {path}")
            self._remember_path("model_dir", os.path.dirname(path))
            self._remember_path("last_model_path", path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not save model:\n{e}")

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model",
                                                        self.settings.value("model_dir", os.getcwd()),
                                                        "Joblib Model (*.joblib)")
        if not path:
            return
        try:
            pipe = joblib.load(path)
            # Back-compat: if older models lack metadata, create minimal defaults
            if not hasattr(pipe, "model_metadata") or pipe.model_metadata is None:
                f_save = np.arange(0, SAVE_FMAX + SAVE_DF, SAVE_DF, dtype=float)
                pipe.model_metadata = {
                    "class_names": [],
                    "f_save": f_save,
                    "save_fmax": SAVE_FMAX,
                    "save_df": SAVE_DF,
                }
            self.model = pipe
            self.class_names = list(pipe.model_metadata.get("class_names", []))
            self.save_model_btn.setEnabled(True)
            self.train_log.appendPlainText(f"Loaded model: {path}")
            if self.class_names:
                self.train_log.appendPlainText(f"Model classes: {self.class_names}")
            self._remember_path("model_dir", os.path.dirname(path))
            self._remember_path("last_model_path", path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Could not load model:\n{e}")

    # ----------------- Offline Test -----------------
    def run_test(self):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "No model", "Train or load a model first.")
            return
        path = self.test_path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "No test CSV", "Please select a test CSV.")
            return

        try:
            X, y, freq_cols, df = self._load_dataset(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Could not load test CSV:\n{e}")
            return

        # Predict
        y_pred = self.model.predict(X)

        if y is not None:
            # Compute metrics
            acc = accuracy_score(y, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
            report = classification_report(y, y_pred, zero_division=0)

            self.test_log.appendPlainText(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
            self.test_log.appendPlainText(f"Accuracy: {acc:.4f}")
            self.test_log.appendPlainText(f"Precision (weighted): {prec:.4f}")
            self.test_log.appendPlainText(f"Recall (weighted):    {rec:.4f}")
            self.test_log.appendPlainText(f"F1 (weighted):        {f1:.4f}")
            self.test_log.appendPlainText("")
            self.test_log.appendPlainText(report)

            # Confusion matrix
            labels = sorted(pd.unique(pd.concat([pd.Series(y), pd.Series(y_pred)])))
            cm = confusion_matrix(y, y_pred, labels=labels)
            dlg = ConfMatDialog(cm, labels, self)
            dlg.exec_()
        else:
            # No labels: just output predictions
            out = df.copy()
            out['Pred'] = y_pred
            preview = out[['Pred']].head(10).to_string(index=False)
            self.test_log.appendPlainText("No labels found in test CSV. Predicted classes (first 10):")
            self.test_log.appendPlainText(preview)

            # Ask to save predictions
            path_out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Predictions CSV",
                                                                os.path.join(os.getcwd(), "predictions.csv"),
                                                                "CSV Files (*.csv)")
            if path_out:
                out.to_csv(path_out, index=False)
                self.test_log.appendPlainText(f"Predictions saved to: {path_out}")

    # ----------------- Live Mode -----------------
    def connect_serial(self):
        if self.reader is not None:
            return
        port = self.port_edit.text().strip()
        try:
            baud = int(self.baud_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Bad baud", "Baud must be an integer.")
            return

        self.reader = SerialReader(port, baud)
        self.reader.data_ready.connect(self.on_samples)
        self.reader.error.connect(self.on_serial_error)
        self.reader.start()
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.capture_baseline_btn.setEnabled(True)
        self.live_log.appendPlainText(f"Serial connected: {port} @ {baud}")

    def disconnect_serial(self):
        if self.reader is None:
            return
        self.reader.stop()
        self.reader.wait(1000)
        self.reader = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.capture_baseline_btn.setEnabled(False)
        self.live_log.appendPlainText("Serial disconnected.")

    def on_serial_error(self, msg: str):
        self.live_log.appendPlainText(f"[Serial error] {msg}")
        self.disconnect_serial()

    def on_samples(self, samples: np.ndarray):
        self.buffer.extend(samples)

    def capture_baseline(self):
        """Capture a baseline spectrum in dB (single snapshot from current buffer)."""
        if len(self.buffer) < FFT_SIZE:
            QtWidgets.QMessageBox.warning(self, "Not enough data", "Wait for live data before capturing baseline.")
            return
        spec_db = self._compute_spectrum_db(self.buffer)
        if spec_db is None:
            return
        self.baseline_db = spec_db.copy()
        self.baseline_curve.setData(self.freqs, self.baseline_db)
        self.live_log.appendPlainText("Baseline captured.")
        self.status.showMessage("Baseline captured for live normalization.")

    def _compute_spectrum_db(self, data_buf) -> np.ndarray:
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
        mags = np.abs(X) / np.sum(window)

        mags_db = 20.0 * np.log10(mags + 1e-12)

        # Exponential smoothing on the dB trace for display stability
        if self.prev_spectrum is None:
            spec = mags_db
        else:
            spec = self.alpha * mags_db + (1.0 - self.alpha) * self.prev_spectrum
        self.prev_spectrum = spec

        # Optional Gaussian smoothing along frequency axis
        sigma = int(self.sigma)
        if sigma > 0:
            spec = gaussian_filter1d(spec, sigma=sigma)

        return spec

    def _build_save_grid(self):
        return np.arange(0, SAVE_FMAX + SAVE_DF, SAVE_DF, dtype=float)

    def _interp_to_grid(self, spec_db: np.ndarray, f_save: np.ndarray):
        return np.interp(f_save, self.freqs[:len(spec_db)], spec_db)

    def _normalize_db(self, spec_db: np.ndarray, baseline_db: np.ndarray) -> np.ndarray:
        if spec_db is None or baseline_db is None:
            return None
        n = min(len(spec_db), len(baseline_db))
        return spec_db[:n] - baseline_db[:n]

    def redraw(self):
        """Refresh plots and, if possible, do live prediction."""
        if len(self.buffer) < FFT_SIZE:
            return

        spec_db = self._compute_spectrum_db(self.buffer)
        if spec_db is None:
            return

        # live dB curve
        self.live_curve.setData(self.freqs, spec_db)

        # normalized live
        if self.baseline_db is not None:
            norm_live = self._normalize_db(spec_db, self.baseline_db)
            # Use the model's save grid if available to guarantee shape match
            f_save = None
            if self.model is not None and hasattr(self.model, "model_metadata"):
                f_save = self.model.model_metadata.get("f_save", None)
            if f_save is None:
                f_save = self._build_save_grid()

            norm_on_grid = self._interp_to_grid(norm_live, f_save)
            self.norm_live_curve.setData(self.freqs[:len(norm_live)], norm_live)

            # Predict
            if self.model is not None:
                try:
                    X_live = norm_on_grid.reshape(1, -1).astype(np.float32)
                    y_pred = self.model.predict(X_live)[0]
                    # Try to get probability/confidence if available
                    conf_str = "—"
                    clf = self.model.named_steps['clf']
                    if hasattr(clf, "predict_proba"):
                        proba = self.model.predict_proba(X_live)[0]
                        pmax = float(np.max(proba))
                        conf_str = f"{pmax:.3f}"
                    elif hasattr(clf, "decision_function"):
                        df = clf.decision_function(X_live)
                        df = np.atleast_2d(df)
                        exps = np.exp(df - np.max(df))
                        probs = exps / np.sum(exps)
                        conf_str = f"{float(np.max(probs)):.3f}"

                    self.pred_label.setText(str(y_pred))
                    self.pred_prob.setText(conf_str)

                    # Update pose display if open
                    if self.pose_win is not None:
                        self.pose_win.set_prediction(str(y_pred), conf_str)

                except Exception as e:
                    self.status.showMessage(f"Live prediction error: {e}")
        else:
            # No baseline yet => clear normalized curve + prediction
            self.norm_live_curve.setData([], [])
            self.pred_label.setText("—")
            self.pred_prob.setText("—")
            if self.pose_win is not None:
                self.pose_win.set_prediction("—", "—")

    # -------- Pose display controls --------
    def toggle_pose_window(self, checked: bool):
        if checked:
            if self.pose_win is None:
                self.pose_win = PoseDisplayWindow(self)
                self.pose_win.show()
                # Initialize display with current prediction if any
                self.pose_win.set_prediction(self.pred_label.text(), self.pred_prob.text())
                self.pose_win.destroyed.connect(self._pose_win_closed)
                self.live_log.appendPlainText("Pose display opened.")
        else:
            if self.pose_win is not None:
                self.pose_win.close()

    def _pose_win_closed(self):
        self.pose_win = None
        self.pose_display_btn.setChecked(False)
        self.live_log.appendPlainText("Pose display closed.")

    def start_pose_recording(self):
        if self.pose_win is None:
            QtWidgets.QMessageBox.information(self, "No Pose Display",
                                              "Open the Pose Display before recording.")
            return
        self.pose_win.start_recording()
        self.start_rec_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(True)

    def stop_pose_recording(self):
        if self.pose_win is not None:
            self.pose_win.stop_recording()
        self.start_rec_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)

    # ----------------- Cleanup -----------------
    def closeEvent(self, event):
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1000)
        # Persist last-used paths
        self._remember_path("train_csv", self.train_path_edit.text().strip())
        self._remember_path("test_csv", self.test_path_edit.text().strip())
        event.accept()

# -----------------------------
# Entry point
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)
    win = ClassifierApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
