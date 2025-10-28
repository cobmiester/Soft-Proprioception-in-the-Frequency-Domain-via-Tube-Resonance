
MASTER_CSV_PATH         = r"30base-h10-h20/baselines_master.csv"

RUNID_COL_CANDIDATES    = ["RunID", "run_id", "runid"]
BASELINE_COL_CANDIDATES = ["BaselineID", "baseline_id", "baselineid"]
LABEL_COL_CANDIDATES    = ["Label", "label", "Class", "class"]

USE_TOP_K_PEAKS         = 15
USE_TOP_K_DIPS          = 15
FREQ_MIN_HZ             = 0.0
FREQ_MAX_HZ             = 8000.0

SMOOTH_WINDOW_BINS      = 8
PEAK_MIN_PROMINENCE     = None
PEAK_REL_PROMINENCE     = 0.03
PEAK_MIN_DISTANCE_HZ    = 40.0
PEAK_MIN_WIDTH_BINS     = 1
ENABLE_PARABOLIC_REFINE = True

DIP_REL_PROMINENCE      = 0.03
DIP_MIN_DISTANCE_HZ     = 40.0
DIP_MIN_WIDTH_BINS      = 1

WRITE_DETECTIONS_CSV    = True
DETECTIONS_OUT          = "detected_peaks_dips.csv"

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os, csv, re, sys
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None
_HZ_NUM = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
def _parse_freq_from_colname(name: str) -> Optional[float]:
    m = _HZ_NUM.search(str(name))
    return float(m.group(1)) if m else None
def load_wide_row_to_spectrum(row: "pd.Series") -> Tuple[np.ndarray, np.ndarray]:
    """Given one DataFrame row with many *Hz columns, return (freqs, values)."""
    hz_cols = [(float(_parse_freq_from_colname(c)), c)
               for c in row.index if _parse_freq_from_colname(c) is not None]
    hz_cols = [(f, c) for (f, c) in hz_cols if c in row and pd.notna(row[c])]
    if len(hz_cols) < 10:
        raise ValueError("Not enough frequency columns found in this row.")
    hz_cols.sort(key=lambda t: t[0])
    freqs = np.array([f for (f, _) in hz_cols], float)
    vals  = np.array([float(row[c]) for (_, c) in hz_cols], float)
    return freqs, vals
###peeaks
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
    out: List[PeakRow] = []
    for k, i in enumerate(idx):
        lb = props['left_bases'][k] if 'left_bases' in props else i
        rb = props['right_bases'][k] if 'right_bases' in props else i
        f_ref = parabolic_refine(f, vs, i) if enable_parabolic_refine else float(f[i])
        out.append(PeakRow(float(f[i]), float(v[i]),
                           float(props['prominences'][k]) if 'prominences' in props else float('nan'),
                           float(f[lb]), float(f[rb]),
                           int(props['widths'][k]) if 'widths' in props else 1,
                           int(i), f_ref))
    return out

def find_dips_with_props(freq_hz: np.ndarray,
                         values: np.ndarray,
                         smooth_window_bins: int = 7,
                         rel_prominence: float = 0.03,
                         min_distance_hz: float = 250.0,
                         min_width_bins: int = 1) -> List[PeakRow]:
    f = np.asarray(freq_hz, float)
    v = np.asarray(values, float)
    order = np.argsort(f)
    f = f[order]; v = v[order]
    vs = moving_average(v, smooth_window_bins)
    inv = -(vs - np.nanmin(vs))
    dyn = float(np.nanmax(inv) - np.nanmin(inv))
    prom_thr = rel_prominence * dyn
    try:
        from scipy.signal import find_peaks  # type: ignore
        df = float(np.median(np.diff(f)))
        dist_bins = max(1, int(round(min_distance_hz / max(df, 1e-9))))
        idx, props = find_peaks(inv, prominence=prom_thr, width=min_width_bins, distance=dist_bins)
    except Exception:
        idx, props = _find_peaks_numpy(f, inv, prom_thr, min_distance_hz, min_width_bins)
    out: List[PeakRow] = []
    for k, i in enumerate(idx):
        lb = props['left_bases'][k] if 'left_bases' in props else i
        rb = props['right_bases'][k] if 'right_bases' in props else i
        f_ref = parabolic_refine(f, inv, i)
        out.append(PeakRow(float(f[i]), float(v[i]),
                           float(props['prominences'][k]) if 'prominences' in props else float('nan'),
                           float(f[lb]), float(f[rb]),
                           int(props['widths'][k]) if 'widths' in props else 1,
                           int(i), f_ref))
    return out

def _first_present(df: "pd.DataFrame", candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None
##reporting detectionss
def _fmt_list(fs: np.ndarray, ndp: int = 2) -> str:
    if fs.size == 0: return "-"
    return ", ".join(f"{x:.{ndp}f}" for x in fs)

def main():
    if pd is None:
        print("This program requires pandas.")
        sys.exit(1)

    if not os.path.isfile(MASTER_CSV_PATH):
        print(f"Master CSV not found: {MASTER_CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(MASTER_CSV_PATH)

    run_col  = _first_present(df, RUNID_COL_CANDIDATES)
    base_col = _first_present(df, BASELINE_COL_CANDIDATES)
    label_col= _first_present(df, LABEL_COL_CANDIDATES)

    print(f"Loaded {len(df)} run(s) from master CSV.")

    out_rows = []

    for idx_row, row in df.iterrows():
        # Spectrum
        try:
            f_Hz, vals = load_wide_row_to_spectrum(row)
        except Exception as e:
            print(f"[row {idx_row}] ! Failed to parse spectrum: {e}")
            continue

        # Detect peaks
        peaks = find_peaks_with_props(
            f_Hz, vals,
            smooth_window_bins=SMOOTH_WINDOW_BINS,
            min_prominence=PEAK_MIN_PROMINENCE,
            rel_prominence=PEAK_REL_PROMINENCE,
            min_distance_hz=PEAK_MIN_DISTANCE_HZ,
            min_width_bins=PEAK_MIN_WIDTH_BINS,
            enable_parabolic_refine=ENABLE_PARABOLIC_REFINE
        )
        fpeaks = np.array([p.f_refined_Hz for p in peaks], float)
        mask = (fpeaks >= FREQ_MIN_HZ) & (fpeaks <= FREQ_MAX_HZ)
        fpeaks = np.sort(fpeaks[mask])
        if USE_TOP_K_PEAKS and USE_TOP_K_PEAKS > 0:
            fpeaks = fpeaks[:USE_TOP_K_PEAKS]

        # Detect dips
        dips = find_dips_with_props(
            f_Hz, vals,
            smooth_window_bins=SMOOTH_WINDOW_BINS,
            rel_prominence=DIP_REL_PROMINENCE,
            min_distance_hz=DIP_MIN_DISTANCE_HZ,
            min_width_bins=DIP_MIN_WIDTH_BINS
        )
        fdips = np.array([d.f_refined_Hz for d in dips], float)
        dmask = (fdips >= FREQ_MIN_HZ) & (fdips <= FREQ_MAX_HZ)
        fdips = np.sort(fdips[dmask])
        if USE_TOP_K_DIPS and USE_TOP_K_DIPS > 0:
            fdips = fdips[:USE_TOP_K_DIPS]

        run_id   = str(row[run_col]) if (run_col is not None and run_col in row and not pd.isna(row[run_col])) else f"row_{idx_row}"
        base_id  = (str(row[base_col]) if (base_col is not None and base_col in row and not pd.isna(row[base_col])) else "")
        label    = (str(row[label_col]) if (label_col is not None and label_col in row and not pd.isna(row[label_col])) else "")

        # --- Console output ---
        print(f"[{run_id}]  peaks: {fpeaks.size}  -> { _fmt_list(fpeaks) }")
        print(f"           dips : {fdips.size}  -> { _fmt_list(fdips) }")

        # --- Row for CSV (store as semicolon-separated lists for clarity) ---
        out_rows.append({
            "RunID": run_id,
            "BaselineID": base_id,
            "Label": label,
            "n_peaks": int(fpeaks.size),
            "n_dips": int(fdips.size),
            "peaks_Hz": ";".join(f"{x:.3f}" for x in fpeaks) if fpeaks.size else "",
            "dips_Hz":  ";".join(f"{x:.3f}" for x in fdips)  if fdips.size else "",
        })

    # --- Save CSV ---
    if WRITE_DETECTIONS_CSV and out_rows:
        with open(DETECTIONS_OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"\nSaved detections to: {os.path.abspath(DETECTIONS_OUT)}")

if __name__ == "__main__":
    main()
