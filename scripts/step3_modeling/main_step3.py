"""
第三步主入口：贝叶斯优化建模 (Bayesian Modeling)

职责（薄编排层）：
  1. CLI 参数解析
  2. 加载数据 + 配置
  3. 调用 split_data / bayesian_modeling 子模块
  4. 打印结果摘要

实际业务逻辑在 bayesian_optimize.py 中。
"""

import argparse
import gc
import os
import sys

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

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for pipeline_config

from bayesian_optimize import _set_cv_params, run_bayesian_modeling, split_data_by_month, split_data_random
from pipeline_config import (
    load_model_params,
    load_evaluation_thresholds,
    make_safe_task_id,
)
try:
    from data_split import load_split_dataset
except ImportError:
    load_split_dataset = None


# ============================================================
# 解析模型配置 → 构建 pbounds + CV 参数
# ============================================================

def _build_pbounds_from_config(model_params: Dict[str, Any]) -> Dict[str, tuple]:
    space = model_params.get("lgbm_search_space", {})
    s = space.get
    nl = s("num_leaves", {})
    return {
        "num_leaves":      (nl.get("min", 2),     nl.get("max", 31)),
        "learning_rate":   (s("learning_rate", {}).get("min", 0.01),  s("learning_rate", {}).get("max", 0.2)),
        "min_child_samples": (s("min_child_samples", {}).get("min", 100), s("min_child_samples", {}).get("max", 256)),
        "subsample":       (s("bagging_fraction", {}).get("min", 0.6),  s("bagging_fraction", {}).get("max", 0.95)),
        "colsample_bytree":(s("feature_fraction", {}).get("min", 0.4),  s("feature_fraction", {}).get("max", 0.9)),
        "reg_alpha":       (s("reg_alpha", {}).get("min", 0.0),      s("reg_alpha", {}).get("max", 1.0)),
        "reg_lambda":      (s("reg_lambda", {}).get("min", 0.0),     s("reg_lambda", {}).get("max", 1.0)),
        "max_depth":       (max(nl.get("min", 2), 2), min(nl.get("max", 31) // 2, 16)),
    }


# ============================================================
# 核心入口
# ============================================================

def run_modeling(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    project_dir: Union[str, Path, None] = None,
    base_dir: Union[str, Path, None] = None,
    month_col: Optional[str] = "month",
    train_months: Optional[List] = None,
    valid_month=None,
    oot_month=None,
    test_size: float = 0.3,
    n_runs: int = 5,
    init_points: int = 10,
    n_iter: int = 30,
    random_state: int = 42,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Step 3 entry point — thin orchestrator."""

    safe_task_id = make_safe_task_id(task_id)
    project_dir = Path(project_dir) if project_dir else SCRIPT_DIR.parent / "output"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    base_dir = base_dir or SCRIPT_DIR.parent.parent
    model_params = load_model_params(project_dir=project_dir, skill_dir=base_dir)
    eval_thresholds = load_evaluation_thresholds(project_dir=project_dir, skill_dir=base_dir)

    bo_cfg = model_params.get("bayesian_optimization", {})
    split_cfg = model_params.get("data_split", {})
    es_cfg = model_params.get("early_stopping", {})
    of_penalty_cfg = model_params.get("overfitting_penalty", {})
    cv_cfg = model_params.get("cross_validation", {})

    cfg_n_runs = bo_cfg.get("n_runs", n_runs)
    cfg_init_points = bo_cfg.get("init_points", init_points)
    cfg_n_iter = bo_cfg.get("n_iter", n_iter)
    cfg_random_state = bo_cfg.get("random_state", random_state)

    es_enabled = es_cfg.get("enabled", True)
    es_rounds = es_cfg.get("patience", 30) if es_enabled else 100

    space = model_params.get("lgbm_search_space", {})
    n_estimators = space.get("n_estimators", {}).get("max", 1000)

    ks_gap_threshold = eval_thresholds.get("ks_overfitting_gap", 0.05)

    # -- 设置全局 CV 参数（供 bayesian_optimize 闭包使用）--
    _set_cv_params(
        ks_overfitting_gap=ks_gap_threshold,
        early_stopping_rounds=es_rounds,
        n_splits=cv_cfg.get("n_splits", 5),
        penalty_factor=of_penalty_cfg.get("penalty_factor", 300),
        n_estimators=n_estimators,
    )

    pbounds = _build_pbounds_from_config(model_params)

    print("  Config: bayesian_opt=%s" % str(bo_cfg))
    print("  Config: early_stopping rounds=%d (enabled=%s)" % (es_rounds, es_enabled))
    print("  Config: data_split=%s" % str(split_cfg))

    # ================================================================
    # 1. 数据拆分
    # ================================================================
    print("\n" + "=" * 60)
    print("[Step 3-1] Data Splitting")
    print("=" * 60)

    val_ratio = split_cfg.get("val_ratio", test_size)
    X_oot = None
    y_oot = None

    # ---- 优先尝试读取已拆好的数据集（每个label独立数据集）----
    dataset_dir = project_dir / "data" / "processed" / "datasets"
    if load_split_dataset is not None and dataset_dir.exists():
        print(f"[Step 3-1] Checking pre-split datasets at: {dataset_dir}")
        df_train = load_split_dataset(label_col, "train", str(dataset_dir))
        df_valid = load_split_dataset(label_col, "valid", str(dataset_dir))
        df_oot = load_split_dataset(label_col, "oot", str(dataset_dir))

        if df_train is not None and df_valid is not None:
            print("[Step 3-1] Pre-split datasets found! Using them directly.")
            X_tr = df_train[feature_cols].values
            y_tr = np.asarray(df_train[label_col])
            X_va = df_valid[feature_cols].values
            y_va = np.asarray(df_valid[label_col])
            if df_oot is not None:
                X_oot = df_oot[feature_cols].values
                y_oot = np.asarray(df_oot[label_col])
                print(f"  Train: {len(X_tr)} | Valid: {len(X_va)} | OOT: {len(X_oot)}")
            else:
                print(f"  Train: {len(X_tr)} | Valid: {len(X_va)} | OOT: None (random split)")

            # ---- Filter out NaN/inf labels before modeling (X NaN is OK for LightGBM) ----
            for arr_name, y_arr in [("y_train", y_tr), ("y_valid", y_va)]:
                mask = ~np.isnan(y_arr) & np.isfinite(y_arr)
                bad_count = (~mask).sum()
                if bad_count > 0:
                    print(f"  [Filter] {arr_name}: removing {bad_count} rows with NaN/inf labels")
            valid_train = ~np.isnan(y_tr) & np.isfinite(y_tr)
            X_tr = X_tr[valid_train]
            y_tr = y_tr[valid_train]
            valid_valid = ~np.isnan(y_va) & np.isfinite(y_va)
            X_va = X_va[valid_valid]
            y_va = y_va[valid_valid]
            if X_oot is not None:
                valid_oot = ~np.isnan(y_oot) & np.isfinite(y_oot)
                X_oot = X_oot[valid_oot]
                y_oot = y_oot[valid_oot]

            # Filter actual inf in features (NaN is OK for LightGBM, only inf is bad)
            for arr_name, X_arr, y_arr in [("X_train", X_tr, y_tr), ("X_valid", X_va, y_va)]:
                inf_mask = np.isinf(X_arr).any(axis=1)
                bad_count = int(inf_mask.sum())
                if bad_count > 0:
                    print(f"  [Filter] {arr_name}: removing {bad_count} rows with inf features")
            X_tr_f = ~np.isinf(X_tr).any(axis=1)
            X_tr = X_tr[X_tr_f]; y_tr = y_tr[X_tr_f]
            X_va_f = ~np.isinf(X_va).any(axis=1)
            X_va = X_va[X_va_f]; y_va = y_va[X_va_f]
            if X_oot is not None:
                X_oot_f = ~np.isinf(X_oot).any(axis=1)
                X_oot = X_oot[X_oot_f]; y_oot = y_oot[X_oot_f]

            print(f"  After filter: Train={len(X_tr)} | Valid: {len(X_va)} | OOT: {len(X_oot) if X_oot is not None else 0}")
        else:
            print("[Step 3-1] Pre-split datasets incomplete, falling back to live split.")
            df_train = df_valid = df_oot = None
    else:
        print("[Step 3-1] No pre-split datasets found, doing live split.")
        df_train = df_valid = df_oot = None

    # ---- 如果没有预拆数据集，执行实时拆分 ----
    if df_train is None or df_valid is None:
        if len(df) > 0:
            print("[Step 3-1] Falling back to live split from input DataFrame.")
            if train_months is not None:
                X_tr, y_tr, X_va, y_va, X_oot, y_oot = split_data_by_month(
                    df=df, label_col=label_col, feature_cols=feature_cols,
                    month_col=month_col, train_months=train_months,
                    val_ratio=val_ratio, random_state=cfg_random_state,
                )
            else:
                X_tr, y_tr, X_va, y_va = split_data_random(
                    df=df, label_col=label_col, feature_cols=feature_cols,
                    val_ratio=val_ratio, random_state=cfg_random_state,
                )
        else:
            print("[Step 3-1] ERROR: No pre-split datasets and input DataFrame is empty!")
            raise ValueError(f"Cannot split data for label {label_col}: no datasets found and no input data.")

    gc.collect()

    # ---- Iron Rule: Step 3+ 用全量数据，但内存限制下采样 ----
    max_train = max(len(X_tr), len(X_va)) if X_oot is None else max(len(X_tr), len(X_va), len(X_oot))
    sampling_threshold = 150_000
    sample_size = 100_000
    feature_count = len(feature_cols)
    if (feature_count >= 300 or max_train >= sampling_threshold) and len(X_tr) > sample_size:
        print(f"[Sampling] Train={len(X_tr)}, Valid={len(X_va)}, Features={feature_count} → downsampling to {sample_size}")
        import random
        idx = list(range(len(X_tr)))
        random.seed(cfg_random_state)
        sample_idx = random.sample(idx, min(sample_size, len(idx)))
        X_tr = X_tr[sample_idx]
        y_tr = y_tr[sample_idx]
        # Also sample validation set proportionally
        val_sample = min(int(sample_size * 0.3), len(X_va))
        val_idx = list(range(len(X_va)))
        random.seed(cfg_random_state + 1)
        val_sample_idx = random.sample(val_idx, min(val_sample, len(val_idx)))
        X_va = X_va[val_sample_idx]
        y_va = y_va[val_sample_idx]
        if X_oot is not None:
            # Sample OOT if too large to fit in memory
            oot_sample_max = 50_000
            if len(X_oot) > oot_sample_max:
                oot_idx = list(range(len(X_oot)))
                random.seed(cfg_random_state + 2)
                oot_sample_idx = random.sample(oot_idx, min(oot_sample_max, len(oot_idx)))
                X_oot = X_oot[oot_sample_idx]
                y_oot = y_oot[oot_sample_idx]
        print(f"  After sampling: Train={len(X_tr)}, Valid={len(X_va)}, OOT={len(X_oot) if X_oot is not None else 0}")
    gc.collect()

    # ================================================================
    # 2. 贝叶斯优化 + 建模
    # ================================================================
    print("\n" + "=" * 60)
    print("[Step 3-2] Bayesian Optimization + LightGBM Modeling")
    print("=" * 60)

    result = run_bayesian_modeling(
        X_train=X_tr, y_train=y_tr,
        X_valid=X_va, y_valid=y_va,
        models_dir=models_dir,
        pbounds=pbounds,
        label_col=label_col,
        safe_task_id=safe_task_id,
        n_runs=cfg_n_runs,
        init_points=cfg_init_points,
        n_iter=cfg_n_iter,
        random_state=cfg_random_state,
        X_oot=X_oot,
        y_oot=y_oot,
        feature_names=feature_cols,
    )

    # ============================================================
    # AGGRESSIVE MEMORY CLEANUP after this label is done
    # ============================================================
    print("\n[Memory] Releasing raw data arrays for label %s..." % label_col)
    del X_tr, y_tr, X_va, y_va
    if X_oot is not None:
        del X_oot, y_oot
    del df
    for _ in range(5):
        gc.collect()

    try:
        import os as _os, psutil as _psutil
        _rss_mb = _psutil.Process(_os.getpid()).memory_info().rss // 1024 // 1024
        print("[Memory] Cleanup complete. RSS ~%d MB" % _rss_mb)
    except Exception:
        print("[Memory] Cleanup complete.")
    # ============================================================

    print("\nStep 3 Complete!")
    return result


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Bayesian Optimization Modeling")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="target")
    parser.add_argument("--feature-cols", type=str, nargs="+", required=True)
    parser.add_argument("--month-col", type=str, default="month")
    parser.add_argument("--train-months", type=str, nargs="+")
    parser.add_argument("--valid-month", type=str)
    parser.add_argument("--oot-month", type=str)
    parser.add_argument("--project-dir", type=str)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--init-points", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=30)

    args = parser.parse_args()
    df = pd.read_csv(args.data_file)

    result = run_modeling(
        df=df,
        label_col=args.label_col,
        feature_cols=args.feature_cols,
        month_col=args.month_col,
        train_months=args.train_months,
        valid_month=args.valid_month,
        oot_month=args.oot_month,
        project_dir=args.project_dir,
        n_runs=args.n_runs,
        init_points=args.init_points,
        n_iter=args.n_iter,
    )

    print("\nOutputs:")
    print("  Model:   %s" % result["model_path"])
    print("  Params:  %s" % result["params_path"])
    for k, v in result["metrics"].items():
        print("  %s: AUC=%s, KS=%s" % (k, v["auc"], v["ks"]))
