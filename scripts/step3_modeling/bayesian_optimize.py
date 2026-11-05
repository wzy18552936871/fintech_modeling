"""
贝叶斯优化 + LightGBM 建模核心模块

从 main_step3.py 中抽取出来，main_step3.py 只负责：
  - 解析 CLI 参数 / 读取数据
  - 调用 run_modeling()
  - 打印结果摘要
"""

from __future__ import print_function, unicode_literals

import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")

# ============================================================
# 全局变量（贝叶斯优化闭包需要）
# ============================================================
X_train_global: Optional[np.ndarray] = None
y_train_global: Optional[np.ndarray] = None
X_valid_global: Optional[np.ndarray] = None
y_valid_global: Optional[np.ndarray] = None
X_oot_global: Optional[np.ndarray] = None
y_oot_global: Optional[np.ndarray] = None
feature_names_global: Optional[List[str]] = None

_cv_params = {
    "ks_overfitting_gap": 0.05,
    "early_stopping_rounds": 100,
    "n_splits": 5,
    "penalty_factor": 300,
    "n_estimators": 1000,
}


def _set_cv_params(
    ks_overfitting_gap: float = 0.05,
    early_stopping_rounds: int = 100,
    n_splits: int = 5,
    penalty_factor: int = 300,
    n_estimators: int = 1000,
) -> None:
    _cv_params["ks_overfitting_gap"] = ks_overfitting_gap
    _cv_params["early_stopping_rounds"] = early_stopping_rounds
    _cv_params["n_splits"] = n_splits
    _cv_params["penalty_factor"] = penalty_factor
    _cv_params["n_estimators"] = n_estimators


# ============================================================
# 指标计算
# ============================================================

def ks_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def evaluate_ks_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_pred_prob)
    ks = ks_score(y_true, y_pred_prob)
    return {"auc": round(auc, 6), "ks": round(ks, 6)}


# ============================================================
# 数据拆分
# ============================================================

def split_data_by_month(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    month_col: str,
    train_months: List,
    val_ratio: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """按月份拆分数据 → (X_train, y_train, X_valid, y_valid, X_oot, y_oot).

    Returns None for OOT arrays if no OOT data available.
    """
    all_cols = list(feature_cols) + [label_col, month_col]
    df_work = df[all_cols].dropna(subset=[label_col]).copy()

    # float64 → float32 降低内存占用
    for col in feature_cols:
        if df_work[col].dtype in (np.float64, np.int64):
            df_work[col] = df_work[col].astype(np.float32)
    gc.collect()

    df_train = df_work[df_work[month_col].isin(train_months)]
    all_months = set(df_work[month_col].unique())
    oot_months_vals = [m for m in all_months if m not in train_months]

    X_all_train = df_train[feature_cols]
    y_all_train = df_train[label_col]

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_all_train, y_all_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_all_train,
    )

    print("  Train+Valid总集(%s): %d rows" % (str(train_months), len(X_all_train)))
    print("  Train(70%%): %d rows" % len(X_train_split))
    print("  Valid(30%%):  %d rows" % len(X_valid_split))

    X_o: Optional[np.ndarray] = None
    y_o: Optional[np.ndarray] = None

    if len(oot_months_vals) > 0:
        df_oot = df_work[~df_work[month_col].isin(train_months)]
        if len(df_oot) > 0:
            print("  OOT(%s):   %d rows" % (str(oot_months_vals), len(df_oot)))
            X_o = df_oot[feature_cols].values.astype(np.float32)
            y_o = df_oot[label_col].values

        del df_oot
    else:
        print("  OOT: no months outside train set")

    del df_work, df_train, X_all_train, y_all_train
    gc.collect()

    return (
        X_train_split.values.astype(np.float32),
        y_train_split.values,
        X_valid_split.values.astype(np.float32),
        y_valid_split.values,
        X_o,
        y_o,
    )


def split_data_random(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    val_ratio: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """随机拆分（无月份列时）."""
    all_cols = list(feature_cols) + [label_col]
    df_work = df[all_cols].dropna(subset=[label_col]).copy()

    for col in feature_cols:
        if df_work[col].dtype in (np.float64, np.int64):
            df_work[col] = df_work[col].astype(np.float32)
    gc.collect()

    X = df_work[feature_cols]
    y = df_work[label_col]

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state, stratify=y,
    )

    print("  Train set: %d rows" % len(X_train_split))
    print("  Valid set: %d rows" % len(X_valid_split))
    print("  (Random split, no month column)")

    del df_work, X, y
    gc.collect()

    return (
        X_train_split.values.astype(np.float32),
        y_train_split.values,
        X_valid_split.values.astype(np.float32),
        y_valid_split.values,
    )


# ============================================================
# 贝叶斯优化目标函数
# ============================================================

def lgb_cv_objective(
    num_leaves: float,
    max_depth: float,
    learning_rate: float,
    min_child_samples: float,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
) -> float:
    """5-fold CV AUC with Train-Valid/OOT KS difference penalty."""
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": _cv_params["n_estimators"],
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "learning_rate": max(min(learning_rate, 1.0), 0.01),
        "min_child_samples": int(min_child_samples),
        "subsample": max(min(subsample, 1.0), 0.1),
        "colsample_bytree": max(min(colsample_bytree, 1.0), 0.1),
        "reg_alpha": max(reg_alpha, 0),
        "reg_lambda": max(reg_lambda, 0),
        "random_state": 42,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(
        n_splits=_cv_params["n_splits"], shuffle=True, random_state=42
    )
    aucs = []
    for train_idx, val_idx in skf.split(X_train_global, y_train_global):
        m = LGBMClassifier(**params)
        m.fit(
            X_train_global[train_idx], y_train_global[train_idx],
            eval_set=[(X_train_global[val_idx], y_train_global[val_idx])],
            feature_name=feature_names_global,
            callbacks=[lgb.callback.early_stopping(_cv_params["early_stopping_rounds"])],
        )
        yp = m.predict_proba(X_train_global[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train_global[val_idx], yp))
        del m

    mean_auc = float(np.mean(aucs))

    # -- Train-Valid/OOT KS 差异惩罚（始终有效） --
    m_full = LGBMClassifier(**params)
    m_full.fit(X_train_global, y_train_global, feature_name=feature_names_global)

    ks_train = ks_score(y_train_global, m_full.predict_proba(X_train_global)[:, 1])

    # 优先用OOT；没有OOT时回退到Valid集，绝不退化为自身
    if X_oot_global is not None:
        ks_ref = ks_score(np.asarray(y_oot_global), m_full.predict_proba(X_oot_global)[:, 1])
        ref_name = "KS_oot"
    else:
        ks_ref = ks_score(y_valid_global, m_full.predict_proba(X_valid_global)[:, 1])
        ref_name = "KS_valid"

    del m_full
    gc.collect()

    ks_diff = abs(ks_train - ks_ref)
    penalty = max(0, (ks_diff - _cv_params["ks_overfitting_gap"])) * _cv_params["penalty_factor"]

    score = mean_auc - penalty
    print("  AUC: %.4f | KS_train: %.4f | %s: %.4f | diff: %.4f | penalty: %.4f | score: %.4f" % (
        mean_auc, ks_train, ref_name, ks_ref, ks_diff, penalty, score))
    return score


# ============================================================
# 核心建模入口
# ============================================================

def run_bayesian_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    models_dir: Path,
    pbounds: Dict[str, Tuple[float, float]],
    label_col: str,
    safe_task_id: str,
    n_runs: int = 5,
    init_points: int = 10,
    n_iter: int = 30,
    random_state: int = 42,
    X_oot: Optional[np.ndarray] = None,
    y_oot: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run Bayesian optimization + training + evaluation, return summary.

    Per-round saves:
      it_{N}_{label}_model.pkl       Best model this round
      it_{N}_{label}_params.json     Params this round
      it_{N}_{label}_eval.json       Eval this round

    Final saves:
      final_model_{label}.pkl        Overall best model
      best_params.json               Optimal params
      round_eval_summary.json        Per-round summary
    """
    global X_train_global, y_train_global, X_valid_global, y_valid_global, X_oot_global, y_oot_global, feature_names_global

    X_train_global = X_train
    y_train_global = y_train
    X_valid_global = X_valid
    y_valid_global = y_valid
    X_oot_global = X_oot
    y_oot_global = y_oot
    feature_names_global = feature_names

    safe_label = label_col.replace(" ", "_").replace("/", "_")
    has_oot = X_oot is not None

    round_records = []  # (round_idx, auc_valid, pkl_path, train_m, valid_m, oot_m, ks_gap)

    for run_idx in range(n_runs):
        print("\n" + "-" * 40)
        print("Bayesian optimization run %d / %d" % (run_idx + 1, n_runs))
        print("-" * 40)

        rng = np.random.RandomState(random_state + run_idx)
        run_seed = rng.randint(-41, 201)
        bo_random_state = random_state + run_seed
        print("  BO seed: base=%d, offset=%d, final=%d" % (random_state, run_seed, bo_random_state))

        optimizer = BayesianOptimization(
            f=lgb_cv_objective,
            pbounds=pbounds,
            random_state=bo_random_state,
        )

        t0 = time.perf_counter()
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        elapsed = time.perf_counter() - t0

        best_params_run = optimizer.max["params"]
        print("Run %d elapsed: %.1fs | Best params: %s" % (run_idx + 1, elapsed, str(best_params_run)))

        # -- 用本轮最优参数在验证集训练 --
        p = dict(best_params_run)
        for k in ("num_leaves", "max_depth", "min_child_samples"):
            p[k] = int(p[k])
        p["objective"] = "binary"
        p["boosting_type"] = "gbdt"
        p["n_estimators"] = _cv_params["n_estimators"]
        p["random_state"] = bo_random_state
        p["n_jobs"] = -1

        run_model = LGBMClassifier(**p)
        run_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            feature_name=feature_names_global,
            callbacks=[lgb.callback.early_stopping(_cv_params["early_stopping_rounds"])],
        )

        m_train = evaluate_ks_auc(y_train, run_model.predict_proba(X_train)[:, 1])
        m_valid = evaluate_ks_auc(y_valid, run_model.predict_proba(X_valid)[:, 1])

        if has_oot:
            try:
                m_oot = evaluate_ks_auc(np.asarray(y_oot), run_model.predict_proba(X_oot)[:, 1])
            except ValueError:
                # OOT has only one class → report nan
                m_oot = {"auc": float("nan"), "ks": float("nan")}
        else:
            m_oot = None

        ks_ref_ks = m_oot["ks"] if m_oot is not None and not np.isnan(m_oot.get("ks", float("nan"))) else m_valid["ks"]
        ks_gap = abs(m_train["ks"] - ks_ref_ks)

        print("Run %d: Train AUC=%.4f KS=%.4f | Valid AUC=%.4f KS=%.4f" % (
            run_idx + 1, m_train["auc"], m_train["ks"], m_valid["auc"], m_valid["ks"]))
        if m_oot is not None:
            print("       OOT  AUC=%.4f KS=%.4f | KS_gap(T-OOT)=%.6f" % (m_oot["auc"], m_oot["ks"], ks_gap))
        else:
            print("       (no OOT)                     | KS_gap(T-Valid)=%.6f" % ks_gap)

        # -- 保存本轮产物 --
        run_model_path = str(models_dir / ("it_%d_%s%s_model.pkl" % (run_idx + 1, safe_label, safe_task_id)))
        joblib.dump(run_model, run_model_path)

        json.dump(p, open(str(models_dir / ("it_%d_%s%s_params.json" % (run_idx + 1, safe_label, safe_task_id))), "w", encoding="utf-8"), indent=2)

        round_eval = {"round": run_idx + 1, "train": m_train, "valid": m_valid, "oot": m_oot, "ks_gap": round(ks_gap, 6)}
        json.dump(round_eval, open(str(models_dir / ("it_%d_%s%s_eval.json" % (run_idx + 1, safe_label, safe_task_id))), "w", encoding="utf-8"), indent=2)

        print("  Saved: it_%d model + params + eval" % (run_idx + 1))

        round_records.append((run_idx + 1, m_valid["auc"], run_model_path, m_train, m_valid, m_oot, ks_gap))

        # -- AGGRESSIVE cleanup of optimizer internals to prevent memory leak --
        # bayesian-optimization accumulates trial history in optimizer.res (DataFrame)
        # and optimizer.space_bounds. Clear these before del to avoid Python 3.7 GC leaks.
        try:
            if hasattr(optimizer, 'res') and isinstance(optimizer.res, pd.DataFrame):
                optimizer.res = pd.DataFrame()
            if hasattr(optimizer, 'space_bounds'):
                optimizer.space_bounds = {}
            if hasattr(optimizer, '_space_bounds'):
                optimizer._space_bounds = {}
            if hasattr(optimizer, 'f'):
                # Sever closure reference to lgb_cv_objective
                optimizer.f = lambda *args, **kwargs: 0.0
        except (AttributeError, TypeError):
            pass
        del optimizer, best_params_run, p, run_model
        for _ in range(3):
            gc.collect()

    # ============================================================
    # MEMORY CLEANUP: Release all heavy objects before summary/final eval
    # ============================================================
    print("\n[Memory] Releasing global data arrays + round model references...")

    # 1. Clear globals (the biggest memory consumers: raw numpy arrays)
    X_train_global = None
    y_train_global = None
    X_valid_global = None
    y_valid_global = None
    X_oot_global = None
    y_oot_global = None
    feature_names_global = None

    # 2. Keep only light metric dicts for summary; discard heavy references
    light_records = []
    for r_idx, auc_v, pkl_p, mt, mv, mo, kg in round_records:
        light_records.append((r_idx, auc_v, pkl_p, mt, mv, mo, kg))
    del round_records

    # 3. Force aggressive GC
    for _ in range(3):
        gc.collect()

    print("[Memory] Cleanup complete.")

    # -- 每轮汇总 --
    print("\n" + "=" * 70)
    print("[Step 3-2.5] Per-Round Evaluation Summary (%d rounds)" % len(light_records))
    print("=" * 70)

    header = "%-6s  %-14s  %-14s"
    if has_oot:
        header += "  %-14s"
    header += "  %-8s"
    print(header % ("Round", "Train(AUC/KS)", "Valid(AUC/KS)",
                    "OOT(AUC/KS)" if has_oot else "(no OOT)", "KS_gap"))

    for r_idx, auc_v, pkl_p, mt, mv, mo, kg in light_records:
        train_s = "%.4f/%.4f" % (mt["auc"], mt["ks"])
        valid_s = "%.4f/%.4f" % (mv["auc"], mv["ks"])
        row = "%-6d  %-14s  %-14s" % (r_idx, train_s, valid_s)
        if mo is not None:
            row += "  %-14s  %.6f" % ("%.4f/%.4f" % (mo["auc"], mo["ks"]), kg)
        else:
            row += "  N/A           %.6f" % kg
        print(row)

    # -- 保存汇总 JSON --
    summary_rows = [{"round": r, "train": mt, "valid": mv, "oot": mo, "ks_gap": kg} for r, _, _, mt, mv, mo, kg in light_records]
    eval_summary_path = str(models_dir / ("round_eval_summary%s.json" % safe_task_id))
    json.dump(summary_rows, open(eval_summary_path, "w", encoding="utf-8"), indent=2)

    # -- 选最优轮次 --
    best_entry = max(light_records, key=lambda x: x[1])
    best_round, best_auc_valid, best_run_pkl_path, _, _, _, _ = best_entry

    del light_records
    gc.collect()

    print("\n>>> Overall best: Round %d (AUC=%.4f)" % (best_round, best_auc_valid))

    # -- 加载最终最佳模型（从磁盘，避免同时驻留多个）--
    best_model = joblib.load(best_run_pkl_path)
    best_params = json.load(open(str(models_dir / ("it_%d_%s%s_params.json" % (best_round, safe_label, safe_task_id))), "r", encoding="utf-8"))

    final_model_path = str(models_dir / ("final_model_%s%s.pkl" % (safe_label, safe_task_id)))
    joblib.dump(best_model, final_model_path)

    # -- 最终评估 --
    print("\n" + "=" * 60)
    print("[Step 3-3] Final Evaluation")
    print("=" * 60)

    def _eval_set(name, X_e, y_e):
        yp = best_model.predict_proba(X_e)[:, 1]
        yt = y_e.values if hasattr(y_e, "values") else np.array(y_e)
        try:
            r = evaluate_ks_auc(yt, yp)
        except ValueError:
            # OOT has only one class → report nan
            print(f"  {name}: AUC=NaN  KS=NaN (single class in y_true)")
            return {"auc": float("nan"), "ks": float("nan")}
        print("  %s: AUC=%.4f  KS=%.4f" % (name, r["auc"], r["ks"]))
        return r

    metrics = {}
    metrics["train"] = _eval_set("Train", pd.DataFrame(X_train), pd.Series(y_train))
    metrics["valid"] = _eval_set("Valid", pd.DataFrame(X_valid), pd.Series(y_valid))
    if has_oot:
        metrics["oot"] = _eval_set("OOT", X_oot, y_oot)

    oot_or_valid = metrics.get("oot", metrics["valid"])
    # Handle NaN KS from single-class OOT
    if np.isnan(oot_or_valid.get("ks", 0)):
        oot_or_valid = metrics["valid"]
    ks_gap_final = abs(metrics["train"]["ks"] - oot_or_valid["ks"])
    gap_label = "OOT" if has_oot and not np.isnan(metrics.get("oot", {}).get("ks", float("nan"))) else "Valid"
    print("\nTrain-%s KS gap: %.4f (threshold: %.2f)" % (gap_label, ks_gap_final, _cv_params["ks_overfitting_gap"]))

    if ks_gap_final > _cv_params["ks_overfitting_gap"]:
        print("WARNING: KS gap exceeds threshold, possible overfitting!")
    else:
        print("KS gap is within acceptable range.")

    # -- 保存最终参数 --
    json.dump(best_params, open(str(models_dir / "best_params.json"), "w", encoding="utf-8"), indent=2)

    # -- 清理全局变量 --
    del best_model
    X_train_global = None
    y_train_global = None
    X_valid_global = None
    y_valid_global = None
    X_oot_global = None
    y_oot_global = None
    feature_names_global = None
    gc.collect()

    return {
        "model_path": final_model_path,
        "params_path": str(models_dir / "best_params.json"),
        "best_params": best_params,
        "metrics": metrics,
        "ks_gap": round(ks_gap_final, 6),
        "round_eval_summary": eval_summary_path,
    }
