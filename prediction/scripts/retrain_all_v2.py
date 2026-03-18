#!/usr/bin/env python3
"""Retrain all price models with 80 unified features.

Steps:
  1. Load data from SQLite archive → compute 80 features → write parquets
  2. Capture "before" metrics from existing evaluation reports
  3. Retrain DAM v2 (6 SPs), RTM (3 horizons), Spike
  4. Print before / after metrics comparison

Usage:
    python -m prediction.scripts.retrain_all_v2
    python -m prediction.scripts.retrain_all_v2 --dam-trials 20 --rtm-trials 15 --spike-trials 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]          # prediction/
PROJECT_ROOT = ROOT.parent                           # ercot/
DATA_DIR = PROJECT_ROOT / "data" / "training"
DB_PATH = PROJECT_ROOT / "scraper" / "data" / "ercot_archive.db"

DAM_REPORT = ROOT / "models" / "dam_v2" / "evaluation_report.json"
RTM_REPORT = ROOT / "models" / "rtm" / "training_report.json"
SPIKE_REPORT = ROOT / "models" / "spike" / "training_report.json"

SETTLEMENT_POINTS = [
    "hb_busavg", "hb_houston", "hb_hubavg", "hb_north", "hb_pan", "hb_south", "hb_west",
    "lz_aen", "lz_cps", "lz_houston", "lz_lcra", "lz_north", "lz_raybn", "lz_south", "lz_west",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_dam_before(report: dict) -> dict:
    """Extract per-SP test MAE / RMSE from DAM evaluation report."""
    out = {}
    for sp, data in report.items():
        best = data.get("best_model", "lightgbm")
        tm = data.get(best, {}).get("test_metrics", {})
        out[sp] = {"mae": tm.get("mae"), "rmse": tm.get("rmse"), "mape_pct": tm.get("mape_pct")}
    return out


def extract_rtm_before(report: dict) -> dict:
    out = {}
    for sp, sp_data in report.items():
        for target, data in sp_data.get("horizons", {}).items():
            best = data.get("best_model", "lightgbm")
            tm = data.get(best, {}).get("test_metrics", {})
            out[f"{sp}/{target}"] = {"mae": tm.get("mae"), "rmse": tm.get("rmse"), "mape_pct": tm.get("mape_pct")}
    return out


def extract_spike_before(report: dict) -> dict:
    out = {}
    for sp, data in report.items():
        tm = data.get("test_metrics", {})
        out[sp] = {"precision": tm.get("precision"), "recall": tm.get("recall"), "f1": tm.get("f1")}
    return out


def print_comparison(title: str, before: dict, after: dict, metrics: list[str]):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = f"{'Key':<28}"
    for m in metrics:
        header += f" {'Before '+m:>14} {'After '+m:>14} {'Δ':>8}"
    print(header)
    print("-" * 90)
    for key in sorted(set(list(before.keys()) + list(after.keys()))):
        b = before.get(key, {})
        a = after.get(key, {})
        row = f"{key:<28}"
        for m in metrics:
            bv = b.get(m)
            av = a.get(m)
            bs = f"{bv:.4f}" if bv is not None else "N/A"
            astr = f"{av:.4f}" if av is not None else "N/A"
            if bv is not None and av is not None:
                delta = av - bv
                ds = f"{delta:+.4f}"
            else:
                ds = "—"
            row += f" {bs:>14} {astr:>14} {ds:>8}"
        print(row)
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrain all price models with 80 features")
    parser.add_argument("--dam-trials", type=int, default=50, help="Optuna trials for DAM v2")
    parser.add_argument("--rtm-trials", type=int, default=30, help="Optuna trials for RTM")
    parser.add_argument("--spike-trials", type=int, default=50, help="Optuna trials for Spike")
    parser.add_argument("--skip-data", action="store_true", help="Skip parquet regeneration")
    parser.add_argument("--skip-dam", action="store_true", help="Skip DAM v2 training")
    parser.add_argument("--skip-rtm", action="store_true", help="Skip RTM training")
    parser.add_argument("--skip-spike", action="store_true", help="Skip Spike training")
    args = parser.parse_args()

    t0 = time.time()

    # --- Capture "before" metrics ---
    dam_before = extract_dam_before(load_json(DAM_REPORT))
    rtm_before = extract_rtm_before(load_json(RTM_REPORT))
    spike_before = extract_spike_before(load_json(SPIKE_REPORT))

    # --- Step 1: Regenerate parquets with 80 features ---
    if not args.skip_data:
        log.info("Step 1: Regenerating training parquets from SQLite …")
        from prediction.src.data.training_pipeline import run_pipeline
        run_pipeline(
            db_path=DB_PATH,
            output_dir=DATA_DIR,
            settlement_points=[sp.upper() for sp in SETTLEMENT_POINTS],
            verbose=True,
        )
        log.info("Parquet regeneration complete.")
    else:
        log.info("Step 1: Skipped parquet regeneration (--skip-data)")

    # --- Step 2: Retrain DAM v2 ---
    if not args.skip_dam:
        log.info("Step 2: Training DAM v2 models for %d settlement points …", len(SETTLEMENT_POINTS))
        from prediction.models.dam_v2.train import train_settlement_point as dam_train_sp
        dam_results = {}
        for sp in SETTLEMENT_POINTS:
            log.info("  DAM v2: %s (%d trials)", sp, args.dam_trials)
            try:
                dam_results[sp] = dam_train_sp(sp, n_trials=args.dam_trials)
            except Exception as e:
                log.error("  DAM v2 %s FAILED: %s", sp, e)
                dam_results[sp] = {"error": str(e)}

        # Save evaluation report
        with open(DAM_REPORT, "w") as f:
            json.dump(dam_results, f, indent=2)
        log.info("DAM v2 evaluation report saved to %s", DAM_REPORT)
    else:
        log.info("Step 2: Skipped DAM v2 training (--skip-dam)")

    # --- Step 3: Retrain RTM ---
    if not args.skip_rtm:
        log.info("Step 3: Training RTM multi-horizon models …")
        from prediction.models.rtm.train import train_settlement_point as rtm_train_sp
        rtm_results = {}
        rtm_points = SETTLEMENT_POINTS  # Train RTM for all 15 SPs
        for sp in rtm_points:
            log.info("  RTM: %s (%d trials)", sp, args.rtm_trials)
            try:
                rtm_results[sp] = rtm_train_sp(sp, n_trials=args.rtm_trials)
            except Exception as e:
                log.error("  RTM %s FAILED: %s", sp, e)
                rtm_results[sp] = {"error": str(e)}

        with open(RTM_REPORT, "w") as f:
            json.dump(rtm_results, f, indent=2)
        log.info("RTM training report saved to %s", RTM_REPORT)
    else:
        log.info("Step 3: Skipped RTM training (--skip-rtm)")

    # --- Step 4: Retrain Spike ---
    if not args.skip_spike:
        log.info("Step 4: Training Spike detection models …")
        from prediction.models.spike.train import train_settlement_point as spike_train_sp
        spike_results = {}
        spike_points = SETTLEMENT_POINTS  # Train Spike for all 15 SPs
        for sp in spike_points:
            log.info("  Spike: %s (%d trials)", sp, args.spike_trials)
            try:
                spike_results[sp] = spike_train_sp(sp, n_trials=args.spike_trials)
            except Exception as e:
                log.error("  Spike %s FAILED: %s", sp, e)
                spike_results[sp] = {"error": str(e)}

        with open(SPIKE_REPORT, "w") as f:
            json.dump(spike_results, f, indent=2)
        log.info("Spike training report saved to %s", SPIKE_REPORT)
    else:
        log.info("Step 4: Skipped Spike training (--skip-spike)")

    # --- Step 5: Before / After comparison ---
    elapsed = time.time() - t0
    print(f"\n\nTotal training time: {elapsed / 60:.1f} min")

    # DAM comparison
    dam_after = extract_dam_before(load_json(DAM_REPORT))
    print_comparison("DAM v2 — Before vs After (test set)", dam_before, dam_after, ["mae", "rmse", "mape_pct"])

    # RTM comparison
    rtm_after = extract_rtm_before(load_json(RTM_REPORT))
    print_comparison("RTM Multi-Horizon — Before vs After (test set)", rtm_before, rtm_after, ["mae", "rmse", "mape_pct"])

    # Spike comparison
    spike_after = extract_spike_before(load_json(SPIKE_REPORT))
    print_comparison("Spike Detection — Before vs After (test set)", spike_before, spike_after, ["precision", "recall", "f1"])

    print("\nDone. All models retrained with 80 unified features.")


if __name__ == "__main__":
    main()
