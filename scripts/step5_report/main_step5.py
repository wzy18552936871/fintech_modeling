"""
Step 5 主入口: 结案报告

职责（薄编排层）：
  - 从子模块 import，传递参数，保存产物
  - run_consolidated_report(): 多任务聚合综合报告
  - run_all_labels(): 逐标签生成报告（向后兼容）

业务逻辑在 report_demo.py 中。
"""
import argparse
import gc
import os
import sys
import warnings

# ---- 自动切换到 pypc 环境 ----
_pypc_exe = r"C:\Users\Admin\.conda\envs\pypc\python.exe"
if __name__ == "__main__":
    _have_pyarrow = False
    try:
        import pyarrow  # noqa: F401
        _have_pyarrow = True
    except ImportError:
        pass
    if not _have_pyarrow and os.path.exists(_pypc_exe):
        print(f"[Env] Switching to pypc: {_pypc_exe}")
        os.execl(_pypc_exe, _pypc_exe, __file__, *sys.argv[1:])

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import json
import joblib

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for pipeline_config

from report_demo import generate_multi_label_report
from outputs import load_output_config, is_function_enabled, resolve_filename
from pipeline_config import load_data_config, get_modeling_tasks


# ============================================================
# Safe file readers
# ============================================================

def _safe_read_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ============================================================
# Gather task metrics (delegated helper)
# ============================================================

def _build_col_name_map(project_dir: Path, task_id: str) -> Dict[str, str]:
    """
    Build a mapping Column_N -> original feature name from selected_features CSV.
    优先匹配包含 task_id 的文件；如果找不到则精确匹配 selected_features_{task_id}.csv。
    不再 fallback 到任意文件，避免不同数据源的特征名映射混乱。
    """
    col_name_map: Dict[str, str] = {}

    # 策略1: 精确匹配 selected_features_{task_id}.csv
    exact = project_dir / "data" / "processed" / f"selected_features_{task_id}.csv"
    if exact.exists():
        feat_file = exact
    else:
        # 策略2: glob 匹配包含 task_id 的文件，从中选最长/最精确的那个
        candidates = sorted(project_dir.glob("data/processed/selected_features_*" + task_id + "*.csv"), key=lambda p: len(p.name), reverse=True)
        if candidates:
            feat_file = candidates[0]
        else:
            # 不再 fallback 到任意文件 — 不同数据源的特征名可能完全不同
            return col_name_map

    if feat_file and feat_file.exists():
        sf_df = _safe_read_csv(feat_file)
        if not sf_df.empty and "feature" in sf_df.columns:
            for idx, fname in enumerate(sf_df["feature"].values):
                col_name_map[f"Column_{idx}"] = str(fname)
    return col_name_map


def _extract_round_fi(
    models_dir: Path,
    task_id: str,
    round_count: int,
    col_name_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    From each per-round model pickle (it_{N}_..._model.pkl), extract
    feature importance (gain + split). Column_N names are mapped to
    original feature names via col_name_map.
    Returns a list of dicts with keys:
      round, feature, importance_split, importance_gain
    """
    all_round_fi: List[Dict[str, Any]] = []
    for rnd in range(1, round_count + 1):
        # --- 策略1: 精确匹配 it_{rnd}_{task_id}_model.pkl ---
        pkl_path = models_dir / f"it_{rnd}_{task_id}_model.pkl"
        if not pkl_path.exists():
            # --- 策略2: glob 但只选最精确的那个 (文件名最短=最精确) ---
            matches = sorted(
                models_dir.glob(f"it_{rnd}*{task_id}*model.pkl"),
                key=lambda p: len(p.name),
            )
            if matches:
                pkl_path = matches[0]
            else:
                continue
        try:
            m = joblib.load(str(pkl_path))
            booster = getattr(m, "booster_", None)
            if booster is None:
                booster = m  # maybe saved as Booster directly
            names = list(booster.feature_name())
            gains = booster.feature_importance(importance_type="gain")
            splits = booster.feature_importance(importance_type="split")
            for feat, g, s in zip(names, gains, splits):
                all_round_fi.append({
                    "round": rnd,
                    "feature": col_name_map.get(str(feat), str(feat)),
                    "importance_split": int(s),
                    "importance_gain": float(g),
                })
        except Exception:
            pass
    return all_round_fi


def _extract_round_monthly(
    models_dir: Path,
    task_id: str,
    round_count: int,
    label_col: str,
    wide_path: Union[str, Path],
    data_config: Dict[str, Any],
    col_name_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    For each per-round model, predict scores on the full wide table and
    compute monthly KS & AUC. Returns a list of dicts with keys:
      round, dataset, month, ks, auc, sample_size

    'dataset' column added before 'month' to distinguish Train/Valid/OOT
    when the same month appears in multiple splits.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    all_rows: List[Dict[str, Any]] = []
    mc = data_config.get("analysis_dims", {}).get("month_col", "month")

    # Load wide table once for feature columns
    print(f"    [Monthly] Loading wide table from {wide_path.name} ...")
    if str(wide_path).endswith(".parquet"):
        df = pd.read_parquet(wide_path)
    else:
        df = pd.read_csv(wide_path)

    # Try to load train/valid/oot parquet splits for per-dataset monthly scoring
    dataset_dir = models_dir.parent / "data" / "processed" / "datasets"
    label_key = task_id.replace("source_all_", "")  # e.g., "zj_y1"
    split_parts = []
    for part_name in ("train", "valid", "oot"):
        split_path = dataset_dir / f"dataset_{label_key}_{part_name}.parquet"
        if split_path.exists():
            try:
                split_df = pd.read_parquet(split_path)
                split_parts.append((part_name.capitalize(), split_df))
            except Exception:
                pass

    has_splits = len(split_parts) > 0
    print(f"    [Monthly] Dataset splits found: {[p[0] for p in split_parts]}" if has_splits else "    [Monthly] No dataset splits, using full wide table")

    print(f"    [Monthly] Computing per-round monthly KS/AUC for {round_count} rounds ...")
    for rnd in range(1, round_count + 1):
        pkl_path = None
        # --- 策略1: 精确匹配 ---
        exact = models_dir / f"it_{rnd}_{task_id}_model.pkl"
        if exact.exists():
            pkl_path = exact
        else:
            # --- 策略2: glob 但只选最精确的那个 (文件名最短=最精确) ---
            candidates_pkl = sorted(
                models_dir.glob(f"it_{rnd}*{task_id}*model.pkl"),
                key=lambda p: len(p.name),
            )
            if candidates_pkl:
                pkl_path = candidates_pkl[0]
            else:
                # Try final model as fallback for the last round
                if rnd == round_count:
                    final_pkl = models_dir / f"final_model_{label_col}_{task_id}.pkl"
                    if final_pkl.exists():
                        pkl_path = final_pkl
        if pkl_path is None:
            continue

        try:
            model = joblib.load(str(pkl_path))
            booster = getattr(model, "booster_", model)
            feat_names = list(booster.feature_name())
        except Exception:
            continue

        # Map Column_N -> original names using selected_features file
        mapped_names = [col_name_map.get(fn, fn) for fn in feat_names]

        def _score_on_df(eval_src, ds_label):
            """Score a model on one dataset and append monthly KS/AUC rows."""
            avail_cols = [fn for fn in mapped_names if fn in eval_src.columns]
            if len(avail_cols) == 0:
                return
            try:
                preds_rnd = model.predict_proba(eval_src[avail_cols]).astype(np.float32)[:, 1]
            except Exception:
                return
            ev_df = eval_src[[label_col, mc]].copy() if mc in eval_src.columns else eval_src[[label_col]].copy()
            ev_df["model_score"] = preds_rnd
            ev_df[label_col] = pd.to_numeric(ev_df[label_col], errors="coerce")
            ev_df = ev_df.dropna(subset=[label_col, "model_score"])
            ev_df = ev_df[ev_df[label_col].isin([0, 1])]
            if mc in ev_df.columns:
                grp = ev_df.groupby(mc)
            else:
                grp = ev_df.groupby(pd.Series(["all"], index=ev_df.index))
            for month_val, grp_data in grp:
                y = grp_data[label_col]
                s = grp_data["model_score"]
                if y.nunique() < 2:
                    continue
                try:
                    auc_val = roc_auc_score(y, s)
                    fpr, tpr, _ = roc_curve(y, s)
                    ks_val = float(max(tpr - fpr))
                    bad_cnt = int(y.sum())
                    all_rows.append({
                        "round": rnd,
                        "dataset": ds_label,
                        "month": str(month_val),
                        "ks": round(ks_val, 6),
                        "auc": round(auc_val, 6),
                        "sample_size": len(grp_data),
                        "bad_count": bad_cnt,
                        "bad_rate": round(bad_cnt / len(grp_data), 6) if len(grp_data) > 0 else 0.0,
                    })
                except Exception:
                    pass

        if has_splits:
            for ds_name, ds_df in split_parts:
                _score_on_df(ds_df, ds_name)
        else:
            # Fallback: score on full wide table
            avail_mapped = [fn for fn in mapped_names if fn in df.columns]
            if len(avail_mapped) == 0:
                del model
                gc.collect()
                continue
            _score_on_df(df, "All")

        del model
        gc.collect()

    return all_rows


def _gather_task_metrics(project_dir: Path, task_id: str, label_col: str, data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect all available evaluation artifacts for one modeling task.

    Returns per-round model metrics (from round_eval_summary JSON).
    Also extracts per-round feature importance from each round's model pickle.
    """
    task_report_dir = project_dir / "reports" / task_id
    models_dir = project_dir / "models"

    result: Dict[str, Any] = {
        "task_id": task_id,
        "label_col": label_col,
        "has_monthly": False,
        "has_importance": False,
        "has_model": False,
        "auc_overall": np.nan,
        "ks_overall": np.nan,
        "num_features": 0,
        "train_months": 0,
        "rounds": [],          # per-round metrics
    }

    # ---- Per-round evaluation (from round_eval_summary_{task_id}.json) ----
    summary_file = models_dir / f"round_eval_summary_{task_id}.json"
    if not summary_file.exists():
        # fallback: glob 匹配包含 task_id 的文件，从中选最精确的那个
        candidates = sorted(
            models_dir.glob("*round_eval_summary*" + task_id + "*.json"),
            key=lambda p: len(p.name),
        )
        if candidates:
            summary_file = candidates[0]
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                rounds_data = json.load(f)
            for entry in rounds_data:
                r_round = entry.get("round", 0)
                r_train = entry.get("train", {})
                r_valid = entry.get("valid", {})
                r_oot   = entry.get("oot", None)
                r_gap   = entry.get("ks_gap", np.nan)
                result["rounds"].append({
                    "round": r_round,
                    "train_auc": r_train.get("auc", np.nan) if isinstance(r_train, dict) else np.nan,
                    "train_ks":  r_train.get("ks",  np.nan) if isinstance(r_train, dict) else np.nan,
                    "valid_auc": r_valid.get("auc", np.nan) if isinstance(r_valid, dict) else np.nan,
                    "valid_ks":  r_valid.get("ks",  np.nan) if isinstance(r_valid, dict) else np.nan,
                    "oot_auc":   r_oot.get("auc", np.nan) if isinstance(r_oot, dict) else np.nan,
                    "oot_ks":    r_oot.get("ks",  np.nan) if isinstance(r_oot, dict) else np.nan,
                    "ks_gap": r_gap,
                })
            # Overall = best valid AUC
            if result["rounds"]:
                best = max(result["rounds"], key=lambda x: x.get("valid_auc", -np.inf))
                result["auc_overall"] = best["valid_auc"]
                result["ks_overall"]  = best["valid_ks"]
        except Exception:
            pass

    # ---- Per-round Feature Importance + Monthly KS/AUC ----
    round_count = len(result["rounds"])
    if round_count > 0:
        col_name_map = _build_col_name_map(project_dir, task_id)
        result["_round_fi"] = _extract_round_fi(models_dir, task_id, round_count, col_name_map)
        # Load wide table for per-round monthly KS/AUC
        try:
            wf1 = project_dir / "data" / "processed" / "wide_table_full.parquet"
            wf2 = project_dir / "data" / "processed" / "wide_table_sampled.parquet"
            wide_path = None
            if wf1.exists():
                wide_path = wf1
            elif wf2.exists():
                wide_path = wf2
            if wide_path is not None:
                result["_round_monthly"] = _extract_round_monthly(
                    models_dir, task_id, round_count, label_col, wide_path, data_config, col_name_map
                )
        except Exception as e:
            print(f"  [WARN] Per-round monthly KS/AUC failed for {task_id}: {e}")
            result["_round_monthly"] = []

    # Monthly KS/AUC (optional)
    monthly_path = task_report_dir / "model_ks_auc_by_month.xlsx"
    if monthly_path.exists():
        result["has_monthly"] = True
        df_m = _safe_read_excel(monthly_path)
        if not df_m.empty:
            mc = data_config.get("analysis_dims", {}).get("month_col", "month")
            if mc in df_m.columns:
                result["train_months"] = int(df_m[mc].nunique())

    # Feature importance (final model)
    fi_path = task_report_dir / "feature_importance.csv"
    if fi_path.exists():
        result["has_importance"] = True
        result["_fi_df"] = _safe_read_csv(fi_path)
        if not result["_fi_df"].empty and "feature" in result["_fi_df"].columns:
            result["num_features"] = len(result["_fi_df"])

    # Model file
    safe_label = label_col.replace("/", "_").replace(" ", "_")
    safe_task_id = task_id.replace("/", "_").replace(" ", "_")
    mp_candidates = [
        models_dir / f"final_model_{safe_label}_{safe_task_id}.pkl",
        models_dir / f"final_model_{task_id}.pkl",
        models_dir / f"final_model_{safe_label}{safe_task_id}.pkl",
    ]
    # 移除 best_lgbm_{label_col} fallback — 太宽松会匹配到别的数据源的模型
    for mp in mp_candidates:
        if mp.exists():
            result["has_model"] = True
            break

    return result


# ============================================================
# Consolidated report entry
# ============================================================

def run_consolidated_report(
    project_dir: Union[str, Path, None] = None,
    base_dir: Union[str, Path, None] = None,
    task_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    生成结案报告：按 (数据源, label) 分组，每个组合一份独立 Excel 报告。

    Sheet 结构（每份报告）：
      1. Dashboard              — 概要统计（总轮数、最佳/最差轮次等）
      2. Round_Metrics          — 每一轮的 Train / Valid / OOT AUC+KS + KS_gap
      3. FI_Final_{label}       — 最终模型的特征重要性
      4. Round_FI_{label}       — 每一轮模型的特征重要性（从每轮 model.pkl 提取）
      5. Round_Monthly_KS_AUC   — 每一轮按月 KS/AUC（从每轮 model.pkl 打分+计算）
      6. Monthly_KS_AUC_{tid}   — step4 最终模型按月 KS/AUC（保留原始产物）
    """

    if project_dir is None:
        project_dir = SCRIPT_DIR.parent / "output"
    project_dir = Path(project_dir)
    base_dir = base_dir or SCRIPT_DIR.parent.parent
    out_cfg = load_output_config(project_dir=project_dir, skill_dir=Path(base_dir))
    s5 = "step5_report"

    print("=" * 70)
    print("Step 5: 按数据源+Label生成报告 (Per Source-Label Report)")
    print("=" * 70)

    # ⚠️ 优先从 project_dir 加载配置，而非 base_dir（skill目录）
    # 否则会用 skill 的旧配置发现不存在的 source_old/source_new 数据源
    try:
        data_config = load_data_config(project_dir)
    except Exception:
        data_config = load_data_config(Path(base_dir))
    tasks = get_modeling_tasks(data_config)

    if task_id:
        tasks = [t for t in tasks if t["task_id"] == task_id]
        if not tasks:
            print(f"[WARN] task_id={task_id} not found")

    print(f"\nFound {len(tasks)} modeling task(s):")
    for t in tasks:
        print(f"  - {t['task_id']} (source={t.get('source_name', t.get('source_key'))}, label={t['label_col']})")

    artifacts: Dict[str, str] = {}

    # ----------------------------------------------------------------
    # 收集所有任务的指标（含 per-round 数据 + per-round FI）
    # ----------------------------------------------------------------
    print("\n[+] Gathering metrics for all tasks ...")
    all_metrics: List[Dict[str, Any]] = []
    task_lookup = {t["task_id"]: t for t in tasks}

    for task in tasks:
        tid = task["task_id"]
        lbl = task["label_col"]
        print(f"  -> Gathering {tid} ...")
        metrics = _gather_task_metrics(project_dir, tid, lbl, data_config)
        all_metrics.append(metrics)

    if not all_metrics:
        print("\n[WARN] No evaluation artifacts found — report will be empty.")

    # ----------------------------------------------------------------
    # 按 (source_key, label_key) 分组 → 每组生成一份报告
    # ----------------------------------------------------------------
    from collections import OrderedDict
    groups: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for m in all_metrics:
        tid = m["task_id"]
        info = task_lookup.get(tid, {})
        group_key = f"{info.get('source_key', info.get('source_name', tid))}_{info.get('label_key', m['label_col'])}"
        groups.setdefault(group_key, []).append(m)

    print(f"\nGenerating {len(groups)} report(s) grouped by source+label:")

    use_openpyxl = True
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        use_openpyxl = False

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for group_key, group_metrics in groups.items():
        src_name = task_lookup.get(group_metrics[0]["task_id"], {}).get("source_name", "")
        lbl_col  = group_metrics[0]["label_col"]

        report_name = f"final_report_{group_key}.xlsx"
        report_path = reports_dir / report_name

        print(f"\n  [Report] {report_path}")

        with pd.ExcelWriter(
            str(report_path),
            engine="openpyxl" if use_openpyxl else None,
            mode="w",
        ) as writer:
            # ---- Sheet 1: Dashboard ----
            _write_dashboard(writer, group_metrics, src_name, lbl_col)

            # ---- Sheet 2: Round_Metrics (所有轮次的所有模型指标) ----
            # Pre-load overall stats from step4 combined eval for bad_count + bad_rate
            round_overall_lookup = {}
            for m in group_metrics:
                tid = m["task_id"]
                task_report_dir_local = project_dir / "reports" / tid
                combined_path = task_report_dir_local / f"score_eval_combined__{tid}.xlsx"
                if not combined_path.exists():
                    import glob as _glob2
                    cands = sorted(_glob2.glob(str(task_report_dir_local / "score_eval_*.xlsx")))
                    combined_path = Path(cands[0]) if cands else None
                if combined_path is not None and combined_path.exists():
                    try:
                        xl = pd.ExcelFile(str(combined_path))
                        for sname in xl.sheet_names:
                            if "_Overall" in sname:
                                df_ov = pd.read_excel(str(combined_path), sheet_name=sname)
                                # Extract round number from sheet name like "model_score_round_3_Overall"
                                sc_name = sname.replace("_Overall", "")
                                rnd_tag = sc_name  # e.g. "model_score" or "model_score_round_3"
                                for _, row in df_ov.iterrows():
                                    sample_cnt = int(row.get("sample_count", 0)) if pd.notna(row.get("sample_count")) else 0
                                    rnd_overall_key = (tid, rnd_tag)
                                    # bad info: we don't have per-round split set sizes,
                                    # but overall sample count = total data used
                                    round_overall_lookup[rnd_overall_key] = {"sample_count": sample_cnt}
                    except Exception as e:
                        pass

            round_rows = []
            for m in group_metrics:
                tid = m["task_id"]
                info = task_lookup.get(tid, {})
                lbl = m.get("label_col", lbl_col)
                for rnd in m.get("rounds", []):
                    rnd_num = rnd["round"]
                    # Build round tag for lookup
                    if rnd_num == 0 or rnd_num is None:
                        rnd_tag = "model_score"
                    else:
                        rnd_tag = f"model_score_round_{rnd_num}"
                    rnd_key = (tid, rnd_tag)
                    ov_info = round_overall_lookup.get(rnd_key, {})

                    # bad_count + bad_rate from the sample (read from wide_table overall)
                    sample_cnt = ov_info.get("sample_count", 0)
                    round_rows.append({
                        "task_id": tid,
                        "source_name": info.get("source_name", tid),
                        "label_col": lbl,
                        "round": rnd["round"],
                        "train_auc": rnd.get("train_auc", np.nan),
                        "train_ks":  rnd.get("train_ks", np.nan),
                        "valid_auc": rnd.get("valid_auc", np.nan),
                        "valid_ks":  rnd.get("valid_ks", np.nan),
                        "oot_auc":   rnd.get("oot_auc", np.nan),
                        "oot_ks":    rnd.get("oot_ks", np.nan),
                        "ks_gap":    rnd.get("ks_gap", np.nan),
                        "sample_count": sample_cnt if sample_cnt > 0 else None,
                    })
            if round_rows:
                round_df = pd.DataFrame(round_rows)
                for c in ["train_auc","train_ks","valid_auc","valid_ks","oot_auc","oot_ks","ks_gap"]:
                    round_df[c] = round_df[c].apply(
                        lambda v: round(v, 6) if not np.isnan(v) else None
                    )
                round_df.to_excel(writer, sheet_name="Round_Metrics", index=False)
            else:
                # Fallback: write a single-row summary
                fallback = []
                for m in group_metrics:
                    tid = m["task_id"]
                    info = task_lookup.get(tid, {})
                    fallback.append({
                        "task_id": tid,
                        "source_name": info.get("source_name", tid),
                        "label_col": lbl_col,
                        "overall_AUC": round(m["auc_overall"], 6) if not np.isnan(m.get("auc_overall", np.nan)) else None,
                        "overall_KS":  round(m["ks_overall"], 6) if not np.isnan(m.get("ks_overall", np.nan)) else None,
                    })
                pd.DataFrame(fallback).to_excel(writer, sheet_name="Round_Metrics", index=False)

            # ---- Sheet 3: Feature_Importance_Final (最终模型的特征重要性) ----
            all_imp_rows = []
            for m in group_metrics:
                tid = m["task_id"]
                fi_df = m.get("_fi_df", pd.DataFrame())
                if not fi_df.empty and "feature" in fi_df.columns:
                    for _, row in fi_df.iterrows():
                        imp_val = float(row.get("importance_gain", row.get("importance", 0)))
                        all_imp_rows.append({
                            "task_id": tid,
                            "feature": str(row.get("feature", "")),
                            "importance_split": int(row.get("importance_split", 0)),
                            "importance_gain": imp_val,
                        })
            if all_imp_rows:
                imp_df = pd.DataFrame(all_imp_rows).sort_values(
                    "importance_gain", ascending=False
                ).reset_index(drop=True)
                safe_name = f"FI_Final_{lbl_col}"[:31]
                imp_df.to_excel(writer, sheet_name=safe_name, index=False)

            # ---- Sheet 4: Round_Feature_Importance (每轮的特征重要性) ----
            all_round_fi_rows = []
            for m in group_metrics:
                tid = m["task_id"]
                info = task_lookup.get(tid, {})
                round_fi = m.get("_round_fi", [])
                for entry in round_fi:
                    all_round_fi_rows.append({
                        "task_id": tid,
                        "source_name": info.get("source_name", tid),
                        "label_col": lbl_col,
                        "round": entry["round"],
                        "feature": entry["feature"],
                        "importance_split": entry["importance_split"],
                        "importance_gain": entry["importance_gain"],
                    })
            if all_round_fi_rows:
                rfi_df = pd.DataFrame(all_round_fi_rows)
                for c in ["importance_gain"]:
                    rfi_df[c] = rfi_df[c].apply(lambda v: round(v, 2) if not np.isnan(v) else None)
                safe_name = f"Round_FI_{lbl_col}"[:31]
                rfi_df.to_excel(writer, sheet_name=safe_name, index=False)

            # ---- Sheet 5: Round_Monthly_KS_AUC (每一轮按月 KS/AUC + bad_rate) ----
            all_round_monthly_rows = []
            for m in group_metrics:
                tid = m["task_id"]
                info = task_lookup.get(tid, {})
                round_monthly = m.get("_round_monthly", [])
                for entry in round_monthly:
                    all_round_monthly_rows.append({
                        "task_id": tid,
                        "source_name": info.get("source_name", tid),
                        "label_col": lbl_col,
                        "round": entry["round"],
                        "dataset": entry.get("dataset", "All"),
                        "month": entry["month"],
                        "ks": entry.get("ks", np.nan),
                        "auc": entry.get("auc", np.nan),
                        "sample_size": entry.get("sample_size", 0),
                        "bad_count": entry.get("bad_count", 0),
                        "bad_rate": entry.get("bad_rate", 0.0),
                    })
            if all_round_monthly_rows:
                rm_df = pd.DataFrame(all_round_monthly_rows)
                for c in ["ks", "auc", "bad_rate"]:
                    rm_df[c] = rm_df[c].apply(lambda v: round(v, 6) if not np.isnan(v) else None)
                safe_name = f"Round_Monthly_KS_AUC"[:31]
                rm_df.to_excel(writer, sheet_name=safe_name, index=False)

            # ---- Sheet X: By_{group} (step4 最终模型分组评估 + bad_rate) ----
            group_by_cols = data_config.get("analysis_dims", {}).get("group_by_cols", [])
            for m in group_metrics:
                tid = m["task_id"]
                task_dir = project_dir / "reports" / tid
                combined_path = task_dir / f"score_eval_combined__{tid}.xlsx"
                if not combined_path.exists():
                    # fallback: try old naming pattern glob
                    import glob as _glob
                    cands = sorted(_glob.glob(str(task_dir / "score_eval_*.xlsx")))
                    combined_path = Path(cands[0]) if cands else None
                if combined_path is not None and combined_path.exists():
                    try:
                        xl = pd.ExcelFile(str(combined_path))
                        for sname in xl.sheet_names:
                            if sname.startswith("model_score_By_") or sname.startswith("model_score_round_By_"):
                                df_grp = pd.read_excel(str(combined_path), sheet_name=sname)
                                safe_sn = (f"By_{tid}_{sname.split('_By_')[1] if '_By_' in sname else sname}")[:31]
                                if len([r for r in writer.sheets.values() if r.title == safe_sn]) == 0:
                                    df_grp.to_excel(writer, sheet_name=safe_sn, index=False)
                    except Exception as e:
                        print(f"  [WARN] ByGroup read failed for {tid}: {e}")

            # ---- Sheet 6: Monthly_KS_AUC (step4 最终模型原始产物，保留) ----
            for m in group_metrics:
                tid = m["task_id"]
                monthly_path = project_dir / "reports" / tid / "model_ks_auc_by_month.xlsx"
                if monthly_path.exists():
                    df_m = _safe_read_excel(monthly_path)
                    if not df_m.empty:
                        safe_name = f"Monthly_Final_{tid}"[:31]
                        df_m.to_excel(writer, sheet_name=safe_name, index=False)

        artifacts[report_name] = str(report_path)

    # ----------------------------------------------------------------
    # 同时保留一份跨所有任务的总体对比表（方便横向比较）
    # ----------------------------------------------------------------
    if all_metrics:
        comp_df = _build_comparison_df(all_metrics, task_lookup)
        if not comp_df.empty:
            comparison_path = reports_dir / "task_comparison.xlsx"
            comp_df.to_excel(str(comparison_path), index=False)
            artifacts["task_comparison"] = str(comparison_path)

    print(f"\n[+] Reports generated:")
    for k, v in artifacts.items():
        print(f"  [{k}] -> {v}")
    gc.collect()
    return artifacts


# ============================================================
# Dashboard / Comparison helpers
# ============================================================

def _write_dashboard(writer: pd.ExcelWriter, metrics: List[Dict[str, Any]], src_name: str, lbl_col: str) -> None:
    """Write the Dashboard summary sheet."""
    rows = []
    n = len(metrics)

    # 统计总轮数
    total_rounds = sum(len(m.get("rounds", [])) for m in metrics)

    rows.append({"summary": "数据源", "value": src_name})
    rows.append({"summary": "Label", "value": lbl_col})
    rows.append({"summary": "总任务数", "value": n})
    rows.append({"summary": "总模型轮数", "value": total_rounds})

    auc_vals = [m["auc_overall"] for m in metrics if not np.isnan(m.get("auc_overall", np.nan))]
    ks_vals  = [m["ks_overall"]  for m in metrics if not np.isnan(m.get("ks_overall", np.nan))]

    if auc_vals:
        rows.append({"summary": "平均最佳AUC", "value": round(np.mean(auc_vals), 6)})
        best_auc = max(metrics, key=lambda m: m.get("auc_overall", -np.inf))
        worst_auc = min(metrics, key=lambda m: m.get("auc_overall", np.inf))
        rows.append({"summary": "最佳AUC任务", "value": f"{best_auc.get('task_id','')} ({best_auc.get('auc_overall','N/A')})"})
        rows.append({"summary": "最差AUC任务", "value": f"{worst_auc.get('task_id','')} ({worst_auc.get('auc_overall','N/A')})"})

    if ks_vals:
        rows.append({"summary": "平均最佳KS", "value": round(np.mean(ks_vals), 6)})
        best_ks = max(metrics, key=lambda m: m.get("ks_overall", -np.inf))
        worst_ks = min(metrics, key=lambda m: m.get("ks_overall", np.inf))
        rows.append({"summary": "最佳KS任务", "value": f"{best_ks.get('task_id','')} ({best_ks.get('ks_overall','N/A')})"})
        rows.append({"summary": "最差KS任务", "value": f"{worst_ks.get('task_id','')} ({worst_ks.get('ks_overall','N/A')})"})

    pd.DataFrame(rows).to_excel(writer, sheet_name="Dashboard", index=False)


def _build_comparison_df(metrics: List[Dict[str, Any]], task_lookup: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        tid = m["task_id"]
        info = task_lookup.get(tid, {})
        auc_v = m.get("auc_overall", np.nan)
        ks_v  = m.get("ks_overall", np.nan)
        rows.append({
            "task_id": tid,
            "source_name": info.get("source_name", tid),
            "label_key": info.get("label_key", m["label_col"]),
            "label_col": m["label_col"],
            "overall_AUC": round(auc_v, 6) if not np.isnan(auc_v) else None,
            "overall_KS":  round(ks_v, 6)  if not np.isnan(ks_v) else None,
            "num_features": m.get("num_features", 0),
            "train_months": m.get("train_months", 0),
            "has_model": m["has_model"],
            "total_rounds": len(m.get("rounds", [])),
        })
    return pd.DataFrame(rows)


# ============================================================
# Legacy: per-label report (backward compat)
# ============================================================

def run_report(
    df: Optional[pd.DataFrame] = None,
    label_col: str = "y",
    feature_cols: Optional[List[str]] = None,
    model_name: str = "LGBM",
    month_col: Optional[str] = "month",
    project_dir: Union[str, os.PathLike, None] = None,
    selected_features_path: Union[str, os.PathLike, None] = None,
    extra_score_cols: List[str] = None,
    model_path_override: Union[str, os.PathLike, None] = None,
    manifest: Optional[Dict[str, Any]] = None,
    base_dir: Union[str, Path, None] = None,
    task_id: Optional[str] = None,
) -> Dict[str, str]:
    """Per-label report (legacy interface)."""
    project_dir = Path(project_dir) if project_dir else SCRIPT_DIR.parent / "output"
    reports_dir = project_dir / "reports"
    models_dir = project_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    lbl = label_col

    # Find model file
    if model_path_override:
        mpath = Path(model_path_override)
    else:
        candidates = [
            models_dir / f"best_lgbm_{lbl}.pkl",
            models_dir / "best_lgbm_model.pkl",
            models_dir / f"final_model_{lbl}.pkl",
        ]
        mpath = next((p for p in candidates if p.exists()), None)

    if not mpath or not mpath.exists():
        print(f"[ERROR] Model file not found: label={lbl}")
        return artifacts

    print(f"\nLoading model: {mpath}")
    model = joblib.load(mpath)

    af = [f for f in feature_cols if f in df.columns] if feature_cols and df is not None else []
    for col in af:
        if df[col].dtype in (np.float64, np.int64):
            df[col] = df[col].astype(np.float32)

    y_pred = model.predict_proba(df[af]).astype(np.float32)[:, 1]
    sc = f"score_{lbl}"
    df[sc] = y_pred

    del model
    gc.collect()

    # Generate report via report_demo
    report_path = str(reports_dir / f"final_report_{lbl}.xlsx")
    try:
        generate_multi_label_report(
            df_raw=df, label_list=[label_col], feature_list=af,
            month_col=month_col if (month_col and month_col in df.columns) else None,
            output_path=report_path, model_name=model_name,
            extra_scores=extra_score_cols or [],
        )
        artifacts[f"final_report_{lbl}"] = report_path
    except Exception as e:
        import traceback
        traceback.print_exc()

    gc.collect()
    return artifacts


def run_all_labels(
    df: pd.DataFrame, label_list: List[str], feature_cols: List[str],
    model_name: str = "LGBM", month_col: Optional[str] = "month",
    project_dir: Union[str, os.PathLike, None] = None,
    extra_score_cols: List[str] = None,
) -> Dict[str, str]:
    """Multi-label entry: generate report per label."""
    all_artifacts: Dict[str, str] = {}
    for lbl in label_list:
        arts = run_report(
            df=df, label_col=lbl, feature_cols=feature_cols,
            model_name=model_name, month_col=month_col,
            project_dir=project_dir, extra_score_cols=extra_score_cols,
        )
        all_artifacts.update(arts)
    gc.collect()
    return all_artifacts


def run_consolidated_or_legacy(
    project_dir: Union[str, Path] = None,
    manifest: Optional[Dict[str, Any]] = None,
    base_dir: Union[str, Path, None] = None,
    task_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Compatibility entry for driver.py.

    When driver calls run_report(project_dir=..., manifest=..., base_dir=..., task_id=...),
    this dispatches to run_consolidated_report instead of the per-label version.
    """
    return run_consolidated_report(
        project_dir=project_dir, base_dir=base_dir, task_id=task_id,
    )


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Report Generation")
    parser.add_argument("--data-file", help="Raw data CSV (per-label mode)")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--feature-cols", nargs="+")
    parser.add_argument("--month-col", default="month")
    parser.add_argument("--project-dir", help="Project/output directory")
    parser.add_argument("--model-name", default="LGBM")
    parser.add_argument("--consolidated", action="store_true", default=False, help="Generate consolidated report")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--task-id", default=None)

    args = parser.parse_args()

    if args.consolidated:
        run_consolidated_report(project_dir=args.project_dir, base_dir=args.base_dir, task_id=args.task_id)
    else:
        if not args.data_file or not args.feature_cols:
            parser.error("--data-file and --feature-cols are required for per-label mode")
        df = pd.read_csv(args.data_file)
        run_report(df, args.label_col, args.feature_cols, args.model_name, args.month_col, args.project_dir)
