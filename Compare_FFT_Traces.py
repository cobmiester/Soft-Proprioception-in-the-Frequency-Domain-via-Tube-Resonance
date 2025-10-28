# Requires recorded data to be run through "Find_Peaks_And_Dips" before using this program

import csv
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =========================
# USER SETTINGS
# =========================
CSV_PATH = "detected_peaks_dips.csv"   # <-- set your CSV filename here
INCLUDE_DIPS = True         # set True to include dips in the comparison
PEAK_WEIGHT = 1.0            # weight for peaks
DIP_WEIGHT = 1.0             # weight for dips (only used if INCLUDE_DIPS=True)
GAP_PENALTY_HZ = 15.0        # penalty for an unmatched feature (Hz). Tune to your FFT resolution.
OUTPUT_RESULTS_CSV = "similarity_results.csv"

# If you want to restrict which baselines count as "baselines", list the cm values here.
# Leave as None to auto-detect any numeric baseline (e.g., 10, 20, 30).
RECOGNIZED_BASELINES_CM: Optional[List[int]] = None  # e.g., [10, 20, 30]


# =========================
# HELPER FUNCTIONS
# =========================
def parse_freq_list(cell: str) -> List[float]:
    """Parse a semicolon-separated list of frequencies like '12.3;45;78.9'."""
    if cell is None:
        return []
    items = [x.strip() for x in str(cell).split(";")]
    out = []
    for x in items:
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

def alignment_cost_sorted(a: List[float], b: List[float], gap_penalty: float) -> float:
    """
    Align two sorted frequency lists with DP:
      - match cost: |a_i - b_j|
      - gap cost: gap_penalty (for unmatched element)
    Returns normalized cost: total_cost / max(len(a), len(b))  (0 if both empty).
    """
    a = sorted(a)
    b = sorted(b)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            match = dp[i - 1][j - 1] + abs(ai - bj)
            gap_a = dp[i - 1][j] + gap_penalty
            gap_b = dp[i][j - 1] + gap_penalty
            dp[i][j] = min(match, gap_a, gap_b)

    total_cost = dp[n][m]
    denom = max(n, m)
    return total_cost / denom if denom > 0 else 0.0

def combined_error(
    peaks_a: List[float], dips_a: List[float],
    peaks_b: List[float], dips_b: List[float],
    include_dips: bool, peak_w: float, dip_w: float, gap_penalty: float
) -> float:
    """Weighted combination of peak and (optional) dip alignment errors."""
    peak_err = alignment_cost_sorted(peaks_a, peaks_b, gap_penalty)
    if include_dips:
        dip_err = alignment_cost_sorted(dips_a, dips_b, gap_penalty)
        wsum = (peak_w + dip_w) if (peak_w + dip_w) > 0 else 1.0
        return (peak_w * peak_err + dip_w * dip_err) / wsum
    else:
        return peak_err

_num_re = re.compile(r"(\d+)\s*(?:cm)?")

def _extract_cm(s: str) -> Optional[int]:
    """Try to extract a baseline distance in cm from a string, e.g., 'baseline 20cm' -> 20."""
    if not s:
        return None
    m = _num_re.search(s.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def classify_row(label: str, baseline_id: str) -> Tuple[str, Optional[int], str]:
    """
    Classify a row as 'baseline' or 'open' (or 'other').
    Return (kind, baseline_cm, display_name)
    - Baseline if BaselineID or Label indicates a number (10,20,30...) and (optionally) contains 'baseline'
    - Open if label contains 'open' or 'cut'
    """
    lbl = (label or "").strip()
    lbl_l = lbl.lower()
    bid = (str(baseline_id) if baseline_id is not None else "").strip()
    bid_l = bid.lower()

    # Try baseline by BaselineID number first
    cm_bid = _extract_cm(bid_l)
    cm_lbl = _extract_cm(lbl_l)

    # Heuristic: if "baseline" in label OR BaselineID is present -> treat numeric as baseline
    if cm_bid is not None or ("baseline" in lbl_l and cm_lbl is not None):
        cm_val = cm_bid if cm_bid is not None else cm_lbl
        if RECOGNIZED_BASELINES_CM is None or (cm_val in RECOGNIZED_BASELINES_CM):
            return "baseline", cm_val, f"Baseline {cm_val}cm"

    # Open-cut by label heuristics
    if any(k in lbl_l for k in ["open", "cut", "slot", "notch"]):
        return "open", None, lbl if lbl else "Open-cut"

    # Fallback: if no baseline id and label doesn't look like a baseline, call it open
    if (cm_bid is None) and ("baseline" not in lbl_l):
        return "open", None, lbl if lbl else "Open-cut"

    return "other", None, lbl or "Unlabeled"


# =========================
# IO + CORE
# =========================
def load_rows(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"RunID", "BaselineID", "Label", "n_peaks", "n_dips", "peaks_Hz", "dips_Hz"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for r in reader:
            rows.append({
                "RunID": r.get("RunID", "").strip(),
                "BaselineID": r.get("BaselineID", "").strip(),
                "Label": r.get("Label", "").strip(),
                "n_peaks": int(r.get("n_peaks", "0") or 0),
                "n_dips": int(r.get("n_dips", "0") or 0),
                "peaks": parse_freq_list(r.get("peaks_Hz", "")),
                "dips": parse_freq_list(r.get("dips_Hz", "")),
            })
    return rows

def main():
    csv_file = Path(CSV_PATH)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file.resolve()}")

    rows = load_rows(str(csv_file))

    # classify rows
    for r in rows:
        kind, cm, disp = classify_row(r["Label"], r["BaselineID"])
        r["__kind"] = kind              # 'baseline' | 'open' | 'other'
        r["__cm"] = cm                  # e.g., 10, 20, 30 (for baselines)
        r["__name"] = disp

    # Gather baselines (by cm) and open-cuts
    baselines: Dict[int, Dict] = {}
    for r in rows:
        if r["__kind"] == "baseline" and r["__cm"] is not None:
            # keep the baseline with the most peaks if duplicates exist
            prev = baselines.get(r["__cm"])
            if (prev is None) or (len(r["peaks"]) > len(prev["peaks"])):
                baselines[r["__cm"]] = r

    opens = [r for r in rows if r["__kind"] == "open"]

    if not baselines:
        print("ERROR: No baselines detected. Ensure BaselineID/Label indicates cm like 'baseline 10cm/20/30'.")
        return

    print("\n=== Detected Samples ===")
    if baselines:
        print("Baselines:")
        for cm in sorted(baselines.keys()):
            b = baselines[cm]
            print(f"  - {b['__name']}: RunID={b['RunID']}, peaks={len(b['peaks'])}")
    if opens:
        for i, oc in enumerate(opens, 1):
            print(f"Open-cut {i}: RunID={oc['RunID']}, Label={oc['__name']}, peaks={len(oc['peaks'])}")
    else:
        print("WARNING: No open-cut samples detected (labels with 'open' or 'cut').")
        return

    # comparison helper
    def compare_pair(a: Dict, b: Dict) -> float:
        return combined_error(
            a["peaks"], a["dips"],
            b["peaks"], b["dips"],
            INCLUDE_DIPS, PEAK_WEIGHT, DIP_WEIGHT, GAP_PENALTY_HZ
        )

    # Compute open→baseline errors
    results_rows = []
    print("\n=== Open-cut → Baseline Similarities (lower is more similar) ===")
    for oc in opens:
        errs = []
        for cm, base in baselines.items():
            e = compare_pair(oc, base)
            errs.append((cm, e))
        # Rank by error
        errs.sort(key=lambda x: x[1])
        # Print summary line
        rank_str = ", ".join([f"{cm}cm={e:.3f}" for cm, e in errs[:3]])
        print(f"- {oc['__name']} (RunID={oc['RunID']}): top matches → {rank_str}")

        # Build a row with per-baseline errors (up to top-3 plus all for CSV)
        row = {
            "RunID": oc["RunID"],
            "Label": oc["__name"],
            "CompareTo": "Baselines",
            "Top1": f"{errs[0][0]}cm ({errs[0][1]:.6f})" if len(errs) >= 1 else "",
            "Top2": f"{errs[1][0]}cm ({errs[1][1]:.6f})" if len(errs) >= 2 else "",
            "Top3": f"{errs[2][0]}cm ({errs[2][1]:.6f})" if len(errs) >= 3 else "",
        }
        # Also include columns for each baseline error (useful for spreadsheets)
        for cm, e in errs:
            row[f"Error_to_{cm}cm"] = f"{e:.6f}"
        results_rows.append(row)

    # Open-cut vs open-cut (unchanged)
    if len(opens) >= 2:
        print("\n=== Open-cut ↔ Open-cut Similarities ===")
        for i in range(len(opens)):
            for j in range(i + 1, len(opens)):
                a, b = opens[i], opens[j]
                e = compare_pair(a, b)
                print(f"- {a['__name']} (RunID={a['RunID']}) vs {b['__name']} (RunID={b['RunID']}): error = {e:.3f}")
                results_rows.append({
                    "RunID": f"{a['RunID']} vs {b['RunID']}",
                    "Label": f"{a['__name']} vs {b['__name']}",
                    "CompareTo": "Open-vs-Open",
                    "Top1": "",
                    "Top2": "",
                    "Top3": "",
                    "Error_Open_vs_Open": f"{e:.6f}",
                })

    # Write results CSV
    # Collect dynamic fieldnames: base fixed + all "Error_to_Xcm" fields if present
    base_fields = ["RunID", "Label", "CompareTo", "Top1", "Top2", "Top3"]
    dynamic_fields = set()
    for r in results_rows:
        for k in r.keys():
            if k.startswith("Error_to_"):
                dynamic_fields.add(k)
    if any("Error_Open_vs_Open" in r for r in results_rows):
        dynamic_fields.add("Error_Open_vs_Open")

    fieldnames = base_fields + sorted(dynamic_fields, key=lambda s: (s.endswith("Open"), s))
    out_path = Path(OUTPUT_RESULTS_CSV)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    # Final summary per open-cut
    print("\n=== Summary (Top-3 per open-cut) ===")
    for r in results_rows:
        if r.get("CompareTo") == "Baselines":
            print(f"{r['Label']}: {r.get('Top1','')}, {r.get('Top2','')}, {r.get('Top3','')}")

    print(f"\nResults written to: {out_path.resolve()}")
    print(f"Settings: INCLUDE_DIPS={INCLUDE_DIPS}, PEAK_WEIGHT={PEAK_WEIGHT}, "
          f"DIP_WEIGHT={DIP_WEIGHT}, GAP_PENALTY_HZ={GAP_PENALTY_HZ}")

if __name__ == "__main__":
    main()
