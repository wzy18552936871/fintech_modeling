"""
第四步主入口：严苛的模型评估 (Strict Evaluation)

职责（薄编排层）：
  1. CLI 参数解析
  2. 加载模型 + 预测打分
  3. 依次调用子模块：KS/AUC → 特征重要性 → 每月分箱坏率图 → 过拟合检查

实际业务逻辑在 ks_auc.py / feature_importance.py / monthly_bin_bad_rate.py 中。
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
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import joblib
import glob as _glob
import re

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for pipeline_config

from outputs import load_output_config, is_function_enabled, resolve_filename
from feature_importance import extract_model_importance
from monthly_bin_bad_rate import draw_graph_batch
from pipeline_config import (
    load_evaluation_thresholds,
    load_stability_thresholds,
    load_monthly_evaluation,
    make_safe_task_id,
)


# ============================================================
# 辅助函数
# ============================================================

def _eval_score_ks_auc(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    month_col: Optional[str],
    group_cols: Optional[List[str]],
    reports_dir: Path,
    safe_task_id: str,
    dataset_splits: Optional[List[Tuple[str, pd.DataFrame]]] = None,
) -> Dict:
    """
    对单个评分列做完整 KS/AUC 评估。返回 DataFrame 字典，由调用方统一写入一个 Excel。

    Returns dict with keys:
      overall_auc, overall_ks, combined_excel (only for first call),
      sheets: list of (sheet_name, DataFrame) tuples
    
    If dataset_splits is provided (list of (name, DataFrame)), monthly KS/AUC
    will include a 'dataset' column to distinguish Train/Valid/OOT.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    results = {}
    sheets = []

    # --- 整体 KS/AUC ---
    eval_df = df[[score_col, label_col]].dropna()
    if len(eval_df) > 0 and eval_df[label_col].nunique() >= 2:
        auc_val = roc_auc_score(eval_df[label_col], eval_df[score_col])
        fpr, tpr, _ = roc_curve(eval_df[label_col], eval_df[score_col])
        ks_val = max(tpr - fpr)
        results["overall_auc"] = round(auc_val, 6)
        results["overall_ks"] = round(ks_val, 6)
        print("  [%s] Overall AUC=%.4f  KS=%.4f  (n=%d)" % (score_col, auc_val, ks_val, len(eval_df)))

        overall_summary_df = pd.DataFrame({
            "score_col": [score_col],
            "overall_auc": [auc_val],
            "overall_ks": [ks_val],
            "sample_count": [len(eval_df)],
        })
        sheets.append(("Overall", overall_summary_df))
    else:
        results["overall_auc"] = None
        results["overall_ks"] = None
        print("  [%s] Skip - insufficient data" % score_col)
        results["sheets"] = sheets
        return results

    # --- Monthly: KS + AUC + bad_rate 合并到一张 sheet ---
    if month_col and month_col in df.columns:
        try:
            has_splits = dataset_splits is not None and len(dataset_splits) > 0
            monthly_rows = []
            eval_sources = dataset_splits if has_splits else [("All", df)]
            for ds_name, src_df in eval_sources:
                for month_val, grp_df in src_df.groupby(month_col):
                    sub = grp_df[[score_col, label_col]].dropna()
                    if len(sub) > 0 and sub[label_col].nunique() >= 2:
                        m_auc = roc_auc_score(sub[label_col], sub[score_col])
                        m_fpr, m_tpr, _ = roc_curve(sub[label_col], sub[score_col])
                        m_ks = max(m_tpr - m_fpr)
                        bad_cnt = int(sub[label_col].sum())
                        row = {
                            month_col: month_val,
                            "auc": round(m_auc, 6),
                            "ks": round(m_ks, 6),
                            "bad_rate": round(bad_cnt / len(sub), 6),
                            "total": len(sub),
                            "bad": bad_cnt,
                        }
                        if has_splits:
                            row["dataset"] = ds_name
                        monthly_rows.append(row)
            if monthly_rows:
                monthly_df = pd.DataFrame(monthly_rows)
                # Ensure 'dataset' column comes before 'month' when splits exist
                if has_splits and "dataset" in monthly_df.columns:
                    col_order = ["dataset", month_col] + [c for c in monthly_df.columns if c not in ("dataset", month_col)]
                    monthly_df = monthly_df[col_order]
                sheets.append(("Monthly", monthly_df))
            print("  [%s] Monthly saved." % score_col)
        except Exception as e:
            print("  [%s] Monthly failed: %s" % (score_col, e))

    # --- By group_col ---
    for gc in (group_cols or []):
        if gc not in df.columns:
            continue
        try:
            grp_rows = []
            for grp_val, grp_df in df.groupby(gc):
                sub = grp_df[[score_col, label_col]].dropna()
                if len(sub) > 0 and sub[label_col].nunique() >= 2:
                    g_auc = roc_auc_score(sub[label_col], sub[score_col])
                    g_fpr, g_tpr, _ = roc_curve(sub[label_col], sub[score_col])
                    g_ks = max(g_tpr - g_fpr)
                    bad_cnt = int(sub[label_col].sum())
                    grp_rows.append({
                        gc: grp_val,
                        "auc": round(g_auc, 6),
                        "ks": round(g_ks, 6),
                        "bad_rate": round(bad_cnt / len(sub), 6),
                        "total": len(sub),
                        "bad": bad_cnt,
                    })
            if grp_rows:
                sheet_name = ("By_" + gc)[:31]
                sheets.append((sheet_name, pd.DataFrame(grp_rows)))
            print("  [%s] By-%s saved." % (score_col, gc))
        except Exception as e:
            print("  [%s] By-%s failed: %s" % (score_col, gc, e))

    results["sheets"] = sheets
    return results


def _out_path(reports_dir: Path, step_key: str, output_key: str, task_id: Optional[str] = None, out_cfg: Any = None) -> Path:
    """Resolve configured output path under reports_dir."""
    if out_cfg is not None:
        resolved = resolve_filename(step_key, output_key, task_id=None, config=out_cfg)
    else:
        # 默认文件名
        default_names = {
            "model_ks_auc_by_month": "model_ks_auc_by_month.xlsx",
            "feature_importance": "feature_importance.csv",
            "monthly_bin_bad_rate": "monthly_bin_bad_rate.html",
        }
        resolved = Path(default_names.get(output_key, output_key + ".xlsx"))
    return reports_dir / resolved.name


# ============================================================
# 核心入口
# ============================================================

def run_evaluation(
    df: pd.DataFrame,
    model_path: Union[str, Path],
    label_col: str,
    feature_cols: List[str],
    month_col: Optional[str] = "month",
    project_dir: Union[str, Path, None] = None,
    base_dir: Union[str, Path, None] = None,
    num_bins: int = 10,
    task_id: Optional[str] = None,
    external_scores: Optional[List[str]] = None,
    group_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parameters
    ----------
    df : pd.DataFrame           全量数据
    model_path : str or Path    训练好的模型 (.pkl)
    label_col : str             标签列
    feature_cols : list         入选特征名单
    month_col : str             月份分组列
    project_dir : str           项目输出目录
    base_dir : str              skill 根目录
    num_bins : int              分箱数
    task_id : str               任务 ID（文件名后缀）
    external_scores : list      外部分数列名列表
    group_cols : list           分组维度列表

    Returns
    -------
    artifacts : dict            产物路径映射
    summary : dict              评估摘要
    """
    safe_task_id = make_safe_task_id(task_id)
    project_dir = Path(project_dir) if project_dir else SCRIPT_DIR.parent / "output"

    if task_id:
        reports_dir = project_dir / "reports" / (task_id.replace(" ", "_").replace("/", "_"))
    else:
        reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -- 加载配置 --
    base_dir = base_dir or SCRIPT_DIR.parent.parent
    thresholds = load_evaluation_thresholds(project_dir=project_dir, skill_dir=base_dir)
    stability = load_stability_thresholds(project_dir=project_dir, skill_dir=base_dir)
    monthly_eval = load_monthly_evaluation(project_dir=project_dir, skill_dir=base_dir)

    try:
        out_cfg = load_output_config(project_dir=project_dir, skill_dir=base_dir)
    except Exception:
        out_cfg = None

    s4 = "step4_evaluation"

    # -- Column_x → 原始特征名映射 --
    col_name_map = {}
    for i, fname in enumerate(feature_cols):
        col_name_map["Column_%d" % i] = fname

    print("=" * 60)
    print("第四步：严苛的模型评估 (Strict Evaluation)")
    print("=" * 60)

    artifacts: Dict[str, str] = {}
    summary: Dict[str, Any] = {}

    # ================================================================
    # 0. 加载模型 & 预测打分
    # ================================================================
    print("\n" + "=" * 50)
    print("[Step 4-0] Load Model & Predict")
    print("=" * 50)

    model = joblib.load(model_path)
    print("Model loaded: %s" % model_path)
    gc.collect()

    available_features = [f for f in feature_cols if f in df.columns]
    missing_feat = set(feature_cols) - set(available_features)
    if missing_feat:
        print("Warning: Missing features %s, using available." % str(missing_feat))

    for col in available_features:
        if df[col].dtype in (np.float64, np.int64):
            df[col] = df[col].astype(np.float32)
    gc.collect()

    y_pred_prob = model.predict_proba(df[available_features]).astype(np.float32)[:, 1]
    df["model_score"] = y_pred_prob
    print("Predictions generated. Score range: [%.4f, %.4f]" % (y_pred_prob.min(), y_pred_prob.max()))

    # ========================
    # ⚠️ Also load per-round models and predict (model_score_round_X)
    # ========================
    model_dir = Path(model_path).parent
    task_label = label_col if label_col in str(model_path) else "unknown"
    round_pattern = os.path.join(str(model_dir), f"it_*_{label_col}_*_model.pkl")
    round_files = sorted(_glob.glob(round_pattern))
    print(f"Found {len(round_files)} per-round model(s) to evaluate")

    for rf in round_files:
        rnd_name = Path(rf).stem
        # Extract round number from filename like "it_1_p3m2_source_all_p3m2_model"
        rnd_match = re.search(r'it_(\d+)', rnd_name)
        rnd_num = int(rnd_match.group(1)) if rnd_match else 0
        try:
            rnd_model = joblib.load(rf)
            rnd_pred = rnd_model.predict_proba(df[available_features]).astype(np.float32)[:, 1]
            score_col_name = f"model_score_round_{rnd_num}"
            df[score_col_name] = rnd_pred
            print(f"  Round {rnd_num}: Score range [%.4f, %.4f]" % (rnd_pred.min(), rnd_pred.max()))
        except Exception as e:
            print(f"  Failed to load round model {rf}: {e}")

    # Collect all score columns for evaluation (final + rounds)
    all_score_cols = ["model_score"]
    for i in range(1, 99):
        sc = f"model_score_round_{i}"
        if sc in df.columns:
            all_score_cols.append(sc)
    print(f"Total score columns to evaluate: {all_score_cols}")

    # Load dataset splits for per-dataset monthly KS/AUC
    label_key = task_id.replace("source_all_", "") if task_id else "unknown"
    dataset_splits_list = []
    processed_dir = project_dir / "data" / "processed"
    for part_name in ("train", "valid", "oot"):
        split_path = processed_dir / "datasets" / f"dataset_{label_key}_{part_name}.parquet"
        if not split_path.exists():
            split_path = processed_dir / f"dataset_{label_key}_{part_name}.parquet"
        if split_path.exists():
            try:
                split_df = pd.read_parquet(split_path)
                dataset_splits_list.append((part_name.capitalize(), split_df))
            except Exception as e:
                print(f"  [WARN] Failed to load {split_path}: {e}")
    if dataset_splits_list:
        print(f"[Step 4-0] Dataset splits loaded: {[s[0] for s in dataset_splits_list]}")
    else:
        print("[Step 4-0] No dataset splits found, monthly evaluation will use full data")


    # ================================================================
    # 1. model_score + per-round scores 完整 KS/AUC（整体 + 分月 + 分group）
    # ================================================================
    if out_cfg is None or is_function_enabled(s4, "monthly_ks_auc", out_cfg):
        print("\n" + "=" * 50)
        print("[Step 4-1b] Model Score KS/AUC (Overall + Monthly + ByGroup)")
        print("=" * 50)

        combined_output = reports_dir / ("score_eval_combined_" + safe_task_id + ".xlsx")
        all_sheets = {}  # sheet_name -> (DataFrame,)

        for sc in all_score_cols:
            try:
                sc_results = _eval_score_ks_auc(
                    df=df, score_col=sc, label_col=label_col,
                    month_col=month_col, group_cols=group_cols,
                    reports_dir=reports_dir, safe_task_id=safe_task_id,
                    dataset_splits=dataset_splits_list,
                )
                if sc == "model_score":
                    summary["model_auc"] = sc_results.get("overall_auc")
                    summary["model_ks"] = sc_results.get("overall_ks")

                sheets_for_sc = sc_results.get("sheets", [])
                for sheet_name, df_sheet in sheets_for_sc:
                    # Prefix each sheet with score_col name to avoid conflicts
                    safe_sn = (sc + "_" + sheet_name)[:31]
                    all_sheets[safe_sn] = df_sheet

                print(f"  [OK] {sc} AUC={sc_results.get('overall_auc')} KS={sc_results.get('overall_ks')}")
            except Exception as e:
                print(f"{sc} KS/AUC failed: %s" % str(e))

        # Write all sheets into one combined Excel
        if all_sheets:
            with pd.ExcelWriter(str(combined_output), engine="xlsxwriter") as writer:
                for sname, sdf in all_sheets.items():
                    sdf.to_excel(writer, sheet_name=sname, index=False)
            print(f"\n[OK] Combined eval saved: {combined_output}")
            artifacts["model_score_eval"] = str(combined_output)
        else:
            print("\n[WARN] No sheets generated for combined eval")
    else:
        print("\n[Step 4-1b] monthly_ks_auc disabled in config, skipping.")

    # ================================================================
    # 2. external_scores 完整 KS/AUC（整体 + 分月 + 分group）
    # ================================================================
    ext_scores_list = [c for c in (external_scores or []) if c in df.columns]
    if ext_scores_list and (out_cfg is None or is_function_enabled(s4, "monthly_ks_auc", out_cfg)):
        print("\n" + "=" * 50)
        print(f"[Step 4-1c] External Scores KS/AUC ({len(ext_scores_list)} scores)")
        print("=" * 50)
        for sc in ext_scores_list:
            try:
                sc_results = _eval_score_ks_auc(
                    df=df, score_col=sc, label_col=label_col,
                    month_col=month_col, group_cols=group_cols,
                    reports_dir=reports_dir, safe_task_id=safe_task_id,
                    dataset_splits=dataset_splits_list,
                )
                summary["ext_" + sc] = {
                    "auc": sc_results.get("overall_auc"),
                    "ks": sc_results.get("overall_ks"),
                }
                if "combined_excel" in sc_results:
                    artifacts["ext_" + sc + "_eval"] = sc_results["combined_excel"]
            except Exception as e:
                print("  [SKIP] %s failed: %s" % (sc, e))
    else:
        if not ext_scores_list:
            print("\n[Step 4-1c] No external scores found in data, skipping.")

    gc.collect()

    # ================================================================
    # 3. 特征重要性
    # ================================================================
    imp_df = None
    if out_cfg is None or is_function_enabled(s4, "feature_importance", out_cfg):
        importance_path = str(_out_path(reports_dir, s4, "feature_importance", task_id=task_id, out_cfg=out_cfg))
        print("\n" + "=" * 50)
        print("[Step 4-2] Feature Importance")
        print("=" * 50)
        try:
            imp_df = extract_model_importance(model_or_path=model, save_path=importance_path)

            # -- 特征名回映射 --
            if "feature" in imp_df.columns and col_name_map:
                old_names = imp_df["feature"].copy()
                imp_df["feature"] = imp_df["feature"].apply(lambda x: col_name_map.get(x, x))
                renamed_count = (old_names != imp_df["feature"]).sum()
                if renamed_count > 0:
                    print("[OK] 已将 %d 个特征名从 Column_x 映射为原始名称" % renamed_count)

            imp_df.to_csv(importance_path, index=False)
            artifacts["feature_importance"] = importance_path
            gc.collect()

            if not imp_df.empty:
                summary["top_10_features"] = imp_df.head(10)[["feature", "importance_gain"]].to_dict("records")
                print("\nTop 10 Features by Gain:")
                for _, row in imp_df.head(10).iterrows():
                    print("  %s: gain=%s" % (row["feature"], row["importance_gain"]))
        except Exception as e:
            print("Feature importance extraction failed: %s" % str(e))
    else:
        print("\n[Step 4-2] feature_importance disabled in config, skipping.")

    # -- 释放模型内存 --
    del model
    gc.collect()

    # ================================================================
    # 4. 变量每月分箱占比坏率图
    # ================================================================
    if out_cfg is None or is_function_enabled(s4, "monthly_bin_bad_rate", out_cfg):
        bin_bad_rate_path = str(_out_path(reports_dir, s4, "monthly_bin_bad_rate", task_id=task_id, out_cfg=out_cfg))
        print("\n" + "=" * 50)
        print("[Step 4-3] Monthly Bin Bad Rate Charts")
        print("=" * 50)
        try:
            if imp_df is not None and not imp_df.empty:
                top_available = [f for f in list(imp_df["feature"].head(20)) if f in df.columns]
            else:
                top_available = available_features[:20]

            draw_graph_batch(
                data=df, feature_list=top_available, label=label_col,
                month=month_col if (month_col and month_col in df.columns) else None,
                num_bins=num_bins, output_path=bin_bad_rate_path,
            )
            artifacts["monthly_bin_bad_rate"] = bin_bad_rate_path
        except Exception as e:
            print("Monthly bin bad rate charts failed: %s" % str(e))
    else:
        print("\n[Step 4-3] monthly_bin_bad_rate disabled in config, skipping.")

    # ================================================================
    # 5. 过拟合红线检查
    # ================================================================
    if out_cfg is None or is_function_enabled(s4, "overfitting_check", out_cfg):
        print("\n" + "=" * 50)
        print("[Step 4-4] Overfitting Check")
        print("=" * 50)

        model_auc_val = summary.get("model_auc", 0.0)
        model_ks_val = summary.get("model_ks", 0.0)

        auc_exc = thresholds.get("auc_min_excellent", 0.65)
        auc_min = thresholds.get("auc_min_acceptable", 0.55)
        ks_good = thresholds.get("ks_min_good", 0.25)

        if model_auc_val >= auc_exc:
            print("AUC=%.4f >= %.2f -> EXCELLENT" % (model_auc_val, auc_exc))
        elif model_auc_val >= auc_min:
            print("AUC=%.4f >= %.2f -> ACCEPTABLE" % (model_auc_val, auc_min))
        else:
            print("FAIL: AUC=%.4f < %.2f -> NEEDS IMPROVEMENT" % (model_auc_val, auc_min))

        if model_ks_val >= ks_good:
            print("KS=%.4f >= %.2f -> GOOD DISCRIMINATION" % (model_ks_val, ks_good))
        else:
            print("KS=%.4f < %.2f -> WEAK DISCRIMINATION" % (model_ks_val, ks_good))

        summary["thresholds_used"] = {
            "ks_overfitting_gap": thresholds.get("ks_overfitting_gap", 0.05),
            "auc_min_acceptable": auc_min,
            "auc_min_excellent": auc_exc,
            "ks_min_good": ks_good,
        }
    else:
        print("\n[Step 4-4] overfitting_check disabled in config, skipping.")

    # -- 总结 --
    print("\n" + "=" * 60)
    print("Step 4 Complete! Evaluation Summary")
    print("=" * 60)
    auc_v = summary.get("model_auc")
    ks_v = summary.get("model_ks")
    print("Model AUC: %.4f" % auc_v if isinstance(auc_v, (int,float)) else "Model AUC: N/A")
    print("Model KS:  %.4f" % ks_v if isinstance(ks_v, (int,float)) else "Model KS:  N/A")
    print("\nOutput directory: %s" % reports_dir)
    for k, v in artifacts.items():
        print("  [%s] -> %s" % (k, v))

    gc.collect()
    return artifacts, summary


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Strict Model Evaluation")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="target")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True)
    parser.add_argument("--month-col", type=str, default="month")
    parser.add_argument("--project-dir", type=str)
    parser.add_argument("--num-bins", type=int, default=10)

    args = parser.parse_args()
    df = pd.read_csv(args.data_file)

    artifacts, summary = run_evaluation(
        df=df, model_path=args.model_path,
        label_col=args.label_col, feature_cols=args.feature_cols,
        month_col=args.month_col, project_dir=args.project_dir,
        num_bins=args.num_bins,
    )

    print("\nEvaluation Summary:")
    for k, v in summary.items():
        if k != "thresholds_used":
            print("  %s: %s" % (k, str(v)))
