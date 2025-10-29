#!/usr/bin/env python3
"""
Batch FFT tube BC classifier (CC/OO vs CO) by matching observed peaks
to theoretical resonance lines generated up to FREQ_MAX_HZ.

What it does
------------
- Recursively scans a folder for CSV files.
- For each CSV: loads frequency spectrum, finds peaks, refines peak frequencies (optional).
- Prompts you for the *true* tube length (cm) for that CSV.
- Generates *theoretical* resonance frequencies up to FREQ_MAX_HZ for:
    CC/OO: f_n = n * v / (2 * L)
    CO   : f_n = (2n - 1) * v / (4 * L)
- Matches observed peaks to nearest theoretical lines (one-to-one),
  computes residuals and RMSEs, and picks the lower-RMSE model.
- Exports:
    - `batch_peaks_diagnostics.csv` (one row per observed peak per file; contains nearest theory line and residuals for CC and CO)
    - `batch_file_summary.csv` (one row per file with CC/CO RMSE, match coverage, chosen model)
    - `results_for_latex.csv` (compact table-ready CSV for Overleaf/LaTeX)

Notes
-----
- CC == OO for spacing; we treat them as one class (CC/OO).
- Default speed of sound v = 343 m/s (20°C, dry air). Override at top if needed.
- Peak picking uses SciPy if available; otherwise a robust NumPy fallback.
- You can limit number of peaks used and frequency range to avoid spurious highs/lows.
- Matching uses greedy nearest-neighbor with uniqueness (each theory line at most once).
"""

# =========================
# ====== USER INPUTS ======
# =========================

# --- Folder ---
FOLDER_PATH             = r"James_baseline_FFTs"   # <--- SET THIS

# --- Physics ---
V_SOUND_MS              = 343.0     # speed of sound to use in formulas

# --- Peak-use policy ---
USE_TOP_K_PEAKS         = None      # e.g., 10 to cap number of peaks; None = use all detected
FREQ_MIN_HZ             = 0.0       # ignore peaks below this frequency
FREQ_MAX_HZ             = 8000.0    # ignore peaks above this frequency

# --- Matching policy ---
MAX_MATCH_GAP_HZ        = 120.0     # if observed->nearest theory gap > this, leave unmatched
MIN_MATCHED_FRACTION    = 0.5       # require at least this fraction matched to call a verdict

# --- CSV writing ---
WRITE_PEAKS_DIAG_CSV    = True
PEAKS_DIAG_OUT          = "batch_peaks_diagnostics.csv"
WRITE_FILE_SUMMARY_CSV  = True
FILE_SUMMARY_OUT        = "batch_file_summary.csv"

# --- LaTeX table output (compact) ---
WRITE_LATEX_TABLE_CSV   = True
LATEX_TABLE_OUT         = "results_for_latex.csv"  # Columns: file, L_true_cm, n_peaks, RMSE_CC_Hz, RMSE_CO_Hz, Best

# --- Peak detection (adapted from your demo) ---
SMOOTH_WINDOW_BINS      = 5
PEAK_MIN_PROMINENCE     = None
PEAK_REL_PROMINENCE     = 0.02
PEAK_MIN_DISTANCE_HZ    = 100.0
PEAK_MIN_WIDTH_BINS     = 1
ENABLE_PARABOLIC_REFINE = True

# =========================
# ========= CODE ==========
# =========================

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os, csv, re, sys
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# ---------- CSV loader (wide or long) ----------

_HZ_NUM = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

def _parse_freq_from_colname(name: str) -> Optional[float]:
    m = _HZ_NUM.search(str(name))
    return float(m.group(1)) if m else None

def load_fft_csv(csv_path: str,
                 freq_col: Optional[str] = None,
                 val_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Flexible loader for: wide 0–8000 Hz headers, or long two-column [freq, value]."""
    if pd is not None:
        try:
            df = pd.read_csv(csv_path)
            # Wide? headers look like numbers with Hz
            hz_cols = [c for c in df.columns if _parse_freq_from_colname(c) is not None]
            if len(hz_cols) >= 50 and len(df) >= 1:
                pairs = [(float(_parse_freq_from_colname(c)), c) for c in hz_cols]
                pairs.sort(key=lambda t: t[0])
                freqs = np.array([p[0] for p in pairs], float)
                vals  = df[[p[1] for p in pairs]].iloc[0].to_numpy(float)
                return freqs, vals
            # Long two-column
            if freq_col is None:
                candidates = [c for c in df.columns if ("freq" in c.lower()) or ("hz" in c.lower())]
                freq_col = candidates[0] if candidates else df.columns[0]
            if val_col is None:
                others = [c for c in df.columns if c != freq_col]
                val_col = others[0] if others else df.columns[-1]
            freqs = df[freq_col].to_numpy(float)
            vals  = df[val_col].to_numpy(float)
            return freqs, vals
        except Exception:
            pass
    # Fallback parser
    with open(csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        rows = [[c.strip() for c in r] for r in reader]
    if not rows:
        raise ValueError("CSV appears empty.")
    header = rows[0]
    freq_from_header = [(_parse_freq_from_colname(h), i) for i, h in enumerate(header)]
    freq_from_header = [(f, i) for (f, i) in freq_from_header if f is not None]
    if len(freq_from_header) >= 50:
        freq_from_header.sort(key=lambda t: t[0])
        freqs = np.array([f for (f, _) in freq_from_header], float)
        data_row = None
        for r in rows[1:]:
            if any(cell for cell in r):
                data_row = r; break
        if data_row is None:
            raise ValueError("CSV has headers but no data rows.")
        vals = []
        for (_, idx) in freq_from_header:
            cell = data_row[idx] if idx < len(data_row) else ""
            try: vals.append(float(cell))
            except Exception: vals.append(np.nan)
        return freqs, np.array(vals, float)
    # Long format
    def _is_num_row(r):
        if len(r) < 2: return False
        try: float(r[0]); float(r[1]); return True
        except Exception: return False
    start = 0 if _is_num_row(rows[0]) else 1
    freq_vals, mag_vals = [], []
    for r in rows[start:]:
        if len(r) < 2: continue
        try:
            freq_vals.append(float(r[0])); mag_vals.append(float(r[1]))
        except Exception:
            continue
    if len(freq_vals) < 2:
        raise ValueError("Could not parse CSV in long format.")
    return np.array(freq_vals, float), np.array(mag_vals, float)

# ---------- Peak detection ----------

def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    if window is None or window <= 1: return np.asarray(y, float)
    w = int(window); w = w if w % 2 == 1 else w + 1
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode="valid")

def _find_peaks_numpy(f: np.ndarray,
                      v: np.ndarray,
                      min_prom: float,
                      min_distance_hz: float,
                      min_width_bins: int):
    vs = v
    g = np.gradient(vs)
    cand = np.where((g[:-1] > 0) & (g[1:] < 0))[0] + 1
    if cand.size == 0:
        return np.array([], int), {}
    df = float(np.median(np.diff(f)))
    dist_bins = max(1, int(round(min_distance_hz / max(df, 1e-9))))
    peaks = []
    for idx in cand:
        vi = vs[idx]
        left = idx
        while left > 0 and vs[left - 1] <= vs[left]:
            left -= 1
        right = idx
        while right < vs.size - 1 and vs[right + 1] <= vs[right]:
            right += 1
        ref = min(vs[left], vs[right])
        prom = vi - ref
        width_bins = right - left
        if prom >= min_prom and width_bins >= int(min_width_bins):
            peaks.append((idx, prom, width_bins, left, right))
    peaks.sort(key=lambda t: t[1], reverse=True)
    taken = np.zeros_like(vs, dtype=bool)
    kept = []
    for idx, prom, wb, lb, rb in peaks:
        if taken[max(0, idx - dist_bins): min(vs.size, idx + dist_bins + 1)].any():
            continue
        kept.append((idx, prom, wb, lb, rb))
        taken[max(0, idx - dist_bins): min(vs.size, idx + dist_bins + 1)] = True
    kept.sort(key=lambda t: t[0])
    if not kept:
        return np.array([], int), {}
    peaks_idx = np.array([k[0] for k in kept], int)
    props = {
        'prominences': np.array([k[1] for k in kept], float),
        'widths':       np.array([k[2] for k in kept], float),
        'left_bases':   np.array([k[3] for k in kept], int),
        'right_bases':  np.array([k[4] for k in kept], int),
    }
    return peaks_idx, props

@dataclass
class PeakRow:
    f_Hz: float
    value: float
    prominence: float
    left_base_Hz: float
    right_base_Hz: float
    width_bins: int
    index: int
    f_refined_Hz: float

def parabolic_refine(f: np.ndarray, y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(y) - 1:
        return float(f[idx])
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return float(f[idx])
    delta = 0.5 * (y0 - y2) / denom
    df = np.median(np.diff(f))
    return float(f[idx] + delta * df)

def find_peaks_with_props(freq_hz: np.ndarray,
                          values: np.ndarray,
                          smooth_window_bins: int = 7,
                          min_prominence: Optional[float] = None,
                          rel_prominence: float = 0.03,
                          min_distance_hz: float = 250.0,
                          min_width_bins: int = 1,
                          enable_parabolic_refine: bool = True) -> List[PeakRow]:
    f = np.asarray(freq_hz, float)
    v = np.asarray(values, float)
    order = np.argsort(f)
    f = f[order]; v = v[order]
    m = np.isfinite(f) & np.isfinite(v)
    f = f[m]; v = v[m]
    vs = moving_average(v, smooth_window_bins)
    dyn = float(np.nanmax(vs) - np.nanmin(vs))
    prom_thr = (rel_prominence * dyn) if (min_prominence is None) else float(min_prominence)
    try:
        from scipy.signal import find_peaks  # type: ignore
        df = float(np.median(np.diff(f)))
        dist_bins = max(1, int(round(min_distance_hz / max(df, 1e-9))))
        idx, props = find_peaks(vs, prominence=prom_thr, width=min_width_bins, distance=dist_bins)
    except Exception:
        idx, props = _find_peaks_numpy(f, vs, prom_thr, min_distance_hz, min_width_bins)
    peaks: List[PeakRow] = []
    for k, i in enumerate(idx):
        lb = props['left_bases'][k] if 'left_bases' in props else i
        rb = props['right_bases'][k] if 'right_bases' in props else i
        f_ref = parabolic_refine(f, vs, i) if enable_parabolic_refine else float(f[i])
        peaks.append(PeakRow(
            f_Hz=float(f[i]),
            value=float(v[i]),
            prominence=float(props['prominences'][k]) if 'prominences' in props else float('nan'),
            left_base_Hz=float(f[lb]),
            right_base_Hz=float(f[rb]),
            width_bins=int(props['widths'][k]) if 'widths' in props else 1,
            index=int(i),
            f_refined_Hz=f_ref,
        ))
    return peaks

# ---------- Theory lines (up to FREQ_MAX_HZ) ----------

def theory_freqs_CC(v_ms: float, L_cm: float, fmax_hz: float) -> np.ndarray:
    L_m = L_cm / 100.0
    if L_m <= 0: return np.array([], float)
    df = v_ms / (2.0 * L_m)
    n_max = int(np.floor(fmax_hz / max(df, 1e-9)))
    if n_max < 1: return np.array([], float)
    n = np.arange(1, n_max + 1, dtype=float)
    return n * df

def theory_freqs_CO(v_ms: float, L_cm: float, fmax_hz: float) -> np.ndarray:
    L_m = L_cm / 100.0
    if L_m <= 0: return np.array([], float)
    df = v_ms / (2.0 * L_m)  # note: spacing equals CC; CO lines are offset by half
    # CO frequencies are at (n - 0.5) * df (i.e., (2n-1) * v / 4L)
    f = []
    n = 1
    while True:
        fn = (n - 0.5) * df
        if fn > fmax_hz: break
        f.append(fn); n += 1
    return np.array(f, float)

# ---------- Nearest neighbor one-to-one matching ----------

@dataclass
class MatchResult:
    rmse_hz: float
    mae_hz: float
    matched_fraction: float
    n_obs: int
    n_matched: int
    # For diagnostics:
    matched_theory_idx: List[Optional[int]]  # same length as observed: index into theory or None
    residuals_hz: List[Optional[float]]      # |f_obs - f_theory| or None

def nearest_match_unique(observed: np.ndarray,
                         theory: np.ndarray,
                         max_gap_hz: Optional[float] = None) -> MatchResult:
    """
    Greedy nearest-neighbor with uniqueness constraint: each theory line can be used at most once.
    For each observed peak (ascending), pick the closest free theory line.
    If the best distance exceeds max_gap_hz (if provided), leave unmatched.
    """
    obs = np.asarray(observed, float)
    thr = np.asarray(theory, float)
    N, M = obs.size, thr.size
    used = np.zeros(M, dtype=bool)
    matched_idx: List[Optional[int]] = [None] * N
    residuals: List[Optional[float]] = [None] * N

    if N == 0 or M == 0:
        return MatchResult(float('nan'), float('nan'), 0.0, int(N), 0, matched_idx, residuals)

    # Precompute insertion indices to speed neighbor checks
    pos = np.searchsorted(thr, obs)
    for i in range(N):
        # candidates: pos[i]-1, pos[i]
        candidates = []
        if pos[i] > 0: candidates.append(pos[i]-1)
        if pos[i] < M: candidates.append(pos[i])
        # Expand outwards a few steps to find a free line if needed.
        left = (candidates[0] - 1) if candidates else -1
        right = (candidates[-1] + 1) if candidates else 0
        best_j = None
        best_d = np.inf
        # check initial candidates
        for j in candidates:
            if 0 <= j < M and not used[j]:
                d = abs(obs[i] - thr[j])
                if d < best_d:
                    best_d, best_j = d, j
        # try more neighbors if both initial ones are taken
        hops = 0
        while best_j is None and hops < 5:
            if left >= 0 and not used[left]:
                d = abs(obs[i] - thr[left])
                if d < best_d:
                    best_d, best_j = d, left
            if right < M and not used[right]:
                d = abs(obs[i] - thr[right])
                if d < best_d:
                    best_d, best_j = d, right
            left -= 1; right += 1; hops += 1

        if best_j is not None:
            if (max_gap_hz is None) or (best_d <= max_gap_hz):
                used[best_j] = True
                matched_idx[i] = int(best_j)
                residuals[i] = float(best_d)
            else:
                matched_idx[i] = None
                residuals[i] = None
        else:
            matched_idx[i] = None
            residuals[i] = None

    # compute stats on matched only
    matched_resids = np.array([r for r in residuals if r is not None], float)
    n_matched = matched_resids.size
    if n_matched == 0:
        return MatchResult(float('nan'), float('nan'), 0.0, int(N), 0, matched_idx, residuals)
    rmse = float(np.sqrt(np.mean(matched_resids**2)))
    mae = float(np.mean(matched_resids))
    frac = n_matched / max(1, N)
    return MatchResult(rmse, mae, frac, int(N), int(n_matched), matched_idx, residuals)

# ---------- Batch driver ----------

def list_csvs(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def main():
    if not os.path.isdir(FOLDER_PATH):
        print(f"Folder not found: {FOLDER_PATH}")
        sys.exit(1)

    files = list_csvs(FOLDER_PATH)
    if not files:
        print(f"No CSVs found under: {FOLDER_PATH}")
        sys.exit(0)

    print(f"Found {len(files)} CSV file(s).")
    peaks_diag_rows = []
    file_summary_rows = []
    latex_rows = []  # compact, Overleaf-friendly

    # Aggregate wins to choose overall model at the end
    wins: Dict[str, int] = {"CC/OO": 0, "CO": 0}

    for idx_file, csv_path in enumerate(files, 1):
        print("\n" + "="*80)
        print(f"[{idx_file}/{len(files)}] Processing: {csv_path}")

        try:
            f_Hz, vals = load_fft_csv(csv_path)
        except Exception as e:
            print(f"  ! Failed to load CSV: {e}")
            continue

        # Peak detection
        peaks = find_peaks_with_props(
            f_Hz, vals,
            smooth_window_bins=SMOOTH_WINDOW_BINS,
            min_prominence=PEAK_MIN_PROMINENCE,
            rel_prominence=PEAK_REL_PROMINENCE,
            min_distance_hz=PEAK_MIN_DISTANCE_HZ,
            min_width_bins=PEAK_MIN_WIDTH_BINS,
            enable_parabolic_refine=ENABLE_PARABOLIC_REFINE
        )
        if not peaks:
            print("  ! No peaks detected; consider relaxing thresholds.")
            continue

        # Filter frequency range and cap peak count
        fpeaks = np.array([p.f_refined_Hz for p in peaks], float)
        mask = (fpeaks >= FREQ_MIN_HZ) & (fpeaks <= FREQ_MAX_HZ)
        fpeaks = fpeaks[mask]
        if fpeaks.size == 0:
            print("  ! Peaks exist but none within configured frequency range.")
            continue
        fpeaks.sort()
        if USE_TOP_K_PEAKS and USE_TOP_K_PEAKS > 0:
            fpeaks = fpeaks[:USE_TOP_K_PEAKS]

        # Ask for true length for this file (used to generate theory lines)
        while True:
            try:
                s = input(f"Enter TRUE tube length in cm for\n    {os.path.basename(csv_path)}\n>>> ")
                L_true_cm = float(s)
                if L_true_cm <= 0:
                    raise ValueError
                break
            except Exception:
                print("Please enter a positive number (cm).")

        # Generate theory lines up to FREQ_MAX_HZ
        f_the_CC = theory_freqs_CC(V_SOUND_MS, L_true_cm, FREQ_MAX_HZ)
        f_the_CO = theory_freqs_CO(V_SOUND_MS, L_true_cm, FREQ_MAX_HZ)

        if f_the_CC.size == 0 or f_the_CO.size == 0:
            print("  ! Could not generate theory lines (check L and v).")
            continue

        # Match observed -> theory (one-to-one), compute residuals
        mCC = nearest_match_unique(fpeaks, f_the_CC, MAX_MATCH_GAP_HZ)
        mCO = nearest_match_unique(fpeaks, f_the_CO, MAX_MATCH_GAP_HZ)

        # Decide based on RMSE, with coverage requirement
        chosen = "Inconclusive"
        cc_ok = (mCC.n_obs > 0) and (mCC.matched_fraction >= MIN_MATCHED_FRACTION) and np.isfinite(mCC.rmse_hz)
        co_ok = (mCO.n_obs > 0) and (mCO.matched_fraction >= MIN_MATCHED_FRACTION) and np.isfinite(mCO.rmse_hz)

        if cc_ok and co_ok:
            chosen = "CC/OO" if mCC.rmse_hz <= mCO.rmse_hz else "CO"
        elif cc_ok:
            chosen = "CC/OO"
        elif co_ok:
            chosen = "CO"

        if chosen in wins:
            wins[chosen] += 1

        # Console report (existing stats)
        print(f"  Peaks used: {fpeaks.size} in [{float(fpeaks.min()):.1f}, {float(fpeaks.max()):.1f}] Hz")
        print(f"  Speed of sound v = {V_SOUND_MS:.2f} m/s,  L_true = {L_true_cm:.3f} cm")
        print(f"  CC/OO: RMSE = {mCC.rmse_hz if np.isfinite(mCC.rmse_hz) else float('nan'):.2f} Hz,"
              f" MAE = {mCC.mae_hz if np.isfinite(mCC.mae_hz) else float('nan'):.2f} Hz,"
              f" matched = {mCC.n_matched}/{mCC.n_obs} ({mCC.matched_fraction*100:.1f}%)")
        print(f"  CO   : RMSE = {mCO.rmse_hz if np.isfinite(mCO.rmse_hz) else float('nan'):.2f} Hz,"
              f" MAE = {mCO.mae_hz if np.isfinite(mCO.mae_hz) else float('nan'):.2f} Hz,"
              f" matched = {mCO.n_matched}/{mCO.n_obs} ({mCO.matched_fraction*100:.1f}%)")
        print(f"  ==> Best for this file: {chosen}")

        # ---- NEW: Console details: unmatched & best-fitting peaks ----
        # Peaks unmatched by BOTH models:
        unmatched_both_idxs = []
        for i in range(fpeaks.size):
            if (mCC.matched_theory_idx[i] is None) and (mCO.matched_theory_idx[i] is None):
                unmatched_both_idxs.append(i)
        if unmatched_both_idxs:
            print("  Peaks unmatched by either model (index : f_obs Hz):")
            for i in unmatched_both_idxs:
                print(f"    {i+1:>3d} : {fpeaks[i]:.1f} Hz")
        else:
            print("  All peaks matched by at least one model.")

        # Best-fitting peaks for the chosen model (lowest residuals)
        def _print_best_fits(model_name: str, match: 'MatchResult', theory: np.ndarray, top_k: int = 5):
            res = [(i, match.residuals_hz[i], match.matched_theory_idx[i])
                   for i in range(len(match.residuals_hz))
                   if match.residuals_hz[i] is not None and match.matched_theory_idx[i] is not None]
            if not res:
                print(f"  No matched peaks for {model_name}.")
                return
            res.sort(key=lambda t: t[1])
            print(f"  Best-matching peaks for {model_name} (top {min(top_k, len(res))}):")
            for i, (idx_obs, resid, idx_the) in enumerate(res[:top_k], start=1):
                print(f"    #{i}: peak_idx {idx_obs+1:>3d}, f_obs={fpeaks[idx_obs]:.1f} Hz, "
                      f"f_the={theory[idx_the]:.1f} Hz, |Δ|={resid:.2f} Hz")

        if chosen == "CC/OO":
            _print_best_fits("CC/OO", mCC, f_the_CC, top_k=5)
        elif chosen == "CO":
            _print_best_fits("CO", mCO, f_the_CO, top_k=5)
        else:
            # If inconclusive, show best for both
            _print_best_fits("CC/OO", mCC, f_the_CC, top_k=5)
            _print_best_fits("CO", mCO, f_the_CO, top_k=5)

        # Per-peak diagnostics rows
        if WRITE_PEAKS_DIAG_CSV:
            for i, fpk in enumerate(fpeaks):
                # Matched theory for CC
                t_idx_cc = mCC.matched_theory_idx[i]
                t_freq_cc = f_the_CC[t_idx_cc] if (t_idx_cc is not None) else None
                resid_cc = mCC.residuals_hz[i]

                # Matched theory for CO
                t_idx_co = mCO.matched_theory_idx[i]
                t_freq_co = f_the_CO[t_idx_co] if (t_idx_co is not None) else None
                resid_co = mCO.residuals_hz[i]

                peaks_diag_rows.append({
                    "file": csv_path,
                    "file_base": _basename_no_ext(csv_path),
                    "peak_idx": i+1,
                    "f_obs_Hz": fpk,
                    "L_true_cm": L_true_cm,
                    "theory_CC_idx": (t_idx_cc + 1) if (t_idx_cc is not None) else None,  # n for CC
                    "theory_CC_f_Hz": t_freq_cc,
                    "resid_CC_Hz": resid_cc,
                    "theory_CO_idx": (t_idx_co + 1) if (t_idx_co is not None) else None,  # n for CO list
                    "theory_CO_f_Hz": t_freq_co,
                    "resid_CO_Hz": resid_co,
                    "unmatched_both": int((t_idx_cc is None) and (t_idx_co is None))
                })

        # Per-file summary (full)
        file_summary_rows.append({
            "file": csv_path,
            "file_base": _basename_no_ext(csv_path),
            "n_observed_peaks": int(fpeaks.size),
            "L_true_cm": L_true_cm,
            "CC_rmse_Hz": mCC.rmse_hz,
            "CC_mae_Hz": mCC.mae_hz,
            "CC_matched_fraction": mCC.matched_fraction,
            "CO_rmse_Hz": mCO.rmse_hz,
            "CO_mae_Hz": mCO.mae_hz,
            "CO_matched_fraction": mCO.matched_fraction,
            "best_model": chosen
        })

        # Compact LaTeX row
        latex_rows.append({
            "file": _basename_no_ext(csv_path),
            "L_true_cm": L_true_cm,
            "n_peaks": int(fpeaks.size),
            "RMSE_CC_Hz": mCC.rmse_hz,
            "RMSE_CO_Hz": mCO.rmse_hz,
            "Best": chosen
        })

    # Exports
    if WRITE_PEAKS_DIAG_CSV and peaks_diag_rows:
        with open(PEAKS_DIAG_OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(peaks_diag_rows[0].keys()))
            w.writeheader(); w.writerows(peaks_diag_rows)
        print(f"\nSaved per-peak diagnostics to: {os.path.abspath(PEAKS_DIAG_OUT)}")

    if WRITE_FILE_SUMMARY_CSV and file_summary_rows:
        with open(FILE_SUMMARY_OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(file_summary_rows[0].keys()))
            w.writeheader(); w.writerows(file_summary_rows)
        print(f"Saved per-file summary to: {os.path.abspath(FILE_SUMMARY_OUT)}")

    if WRITE_LATEX_TABLE_CSV and latex_rows:
        # Ensure values are simple for LaTeX tables (rounding)
        for r in latex_rows:
            # keep file and Best as-is; round numeric fields
            if isinstance(r.get("L_true_cm"), (int, float)):
                r["L_true_cm"] = round(float(r["L_true_cm"]), 3)
            if isinstance(r.get("RMSE_CC_Hz"), (int, float)):
                r["RMSE_CC_Hz"] = round(float(r["RMSE_CC_Hz"]), 2) if np.isfinite(r["RMSE_CC_Hz"]) else None
            if isinstance(r.get("RMSE_CO_Hz"), (int, float)):
                r["RMSE_CO_Hz"] = round(float(r["RMSE_CO_Hz"]), 2) if np.isfinite(r["RMSE_CO_Hz"]) else None
        with open(LATEX_TABLE_OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file","L_true_cm","n_peaks","RMSE_CC_Hz","RMSE_CO_Hz","Best"])
            w.writeheader(); w.writerows(latex_rows)
        print(f"Saved LaTeX-ready table to: {os.path.abspath(LATEX_TABLE_OUT)}")

    # Overall winner
    total = wins["CC/OO"] + wins["CO"]
    if total > 0:
        if wins["CC/OO"] > wins["CO"]:
            overall = "CC/OO"
        elif wins["CO"] > wins["CC/OO"]:
            overall = "CO"
        else:
            overall = "Tie (equal wins)"
        print("\n" + "="*80)
        print("Folder verdict:")
        print(f"  CC/OO wins: {wins['CC/OO']}  |  CO wins: {wins['CO']}  |  Total files: {total}")
        print(f"  ==> Overall best boundary condition: {overall}")
    else:
        print("\nNo files successfully evaluated.")

if __name__ == "__main__":
    main()
