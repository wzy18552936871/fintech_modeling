"""
Fix model.pkl feature names: Column_N -> original feature name.

Usage:
    python fix_model_feature_names.py --project-dir <your_project_dir>
"""
import argparse
import re
import sys
import tempfile
import os
from pathlib import Path

import joblib
import lightgbm
import pandas as pd


def build_col_name_map(project_dir, task_id):
    """Build list of original feature names from selected_features CSV."""
    feat_file = None
    for p in project_dir.glob("data/processed/selected_features_*" + task_id + "*.csv"):
        feat_file = p
        break
    if feat_file is None:
        for p in project_dir.glob("data/processed/selected_features_*.csv"):
            feat_file = p
            break
    if feat_file and feat_file.exists():
        df = pd.read_csv(feat_file)
        if not df.empty and "feature" in df.columns:
            return [str(x) for x in df["feature"].values]
    return None


def fix_model_feature_names(pkl_path, new_names):
    """Load LGBMClassifier, fix _Booster feature names, save back."""
    model = joblib.load(str(pkl_path))

    old_names = list(model.booster_.feature_name())
    if len(old_names) != len(new_names):
        print(f"  [SKIP] {pkl_path.name} - feature count mismatch ({len(old_names)} vs {len(new_names)})")
        return False

    changed = any(a != b for a, b in zip(old_names, new_names))
    if not changed:
        print(f"  [SKIP] {pkl_path.name} - already correct")
        return False

    # Save internal _Booster to text
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        tmp_path = f.name
    model._Booster.save_model(tmp_path)
    with open(tmp_path, 'r') as f:
        model_text = f.read()
    os.remove(tmp_path)

    # Replace feature_names line (space-separated)
    old_line = "feature_names=" + " ".join(old_names)
    new_line = "feature_names=" + " ".join(new_names)
    model_text = model_text.replace(old_line, new_line)

    # Create new Booster from modified text and replace _Booster
    new_internal = lightgbm.Booster(model_str=model_text)
    model._Booster = new_internal

    joblib.dump(model, str(pkl_path))
    print(f"  [FIX]  {pkl_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    models_dir = project_dir / "models"

    if not models_dir.exists():
        print(f"[ERROR] Models dir not found: {models_dir}")
        return 1

    all_pkls = sorted(models_dir.glob("*model.pkl"))
    print(f"Found {len(all_pkls)} model files")

    # Find task IDs from round eval summaries
    task_ids = set()
    for sf in models_dir.glob("round_eval_summary_*.json"):
        m = re.search(r"round_eval_summary_(.+)\.json", sf.name)
        if m:
            task_ids.add(m.group(1))

    print(f"Found {len(task_ids)} task IDs")

    fixed_count = 0
    for pkl in all_pkls:
        # Determine which task_id this pkl belongs to
        matched_tid = None
        for tid in task_ids:
            if tid in pkl.name:
                matched_tid = tid
                break
        if matched_tid is None:
            print(f"  [SKIP] {pkl.name} - cannot determine task_id")
            continue

        new_names = build_col_name_map(project_dir, matched_tid)
        if not new_names:
            print(f"  [WARN] No feature mapping for {matched_tid}")
            continue

        try:
            if fix_model_feature_names(pkl, new_names):
                fixed_count += 1
        except Exception as e:
            print(f"  [ERR]  {pkl.name}: {e}")

    print(f"\nDone. Fixed {fixed_count}/{len(all_pkls)} model files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
