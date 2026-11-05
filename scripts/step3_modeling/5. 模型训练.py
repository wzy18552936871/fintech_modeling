"""
独立的模型训练脚本 — 与 main_step3.py 使用同一套配置驱动逻辑
用法: 在 Jupyter notebook 中先准备 X_train, y_train, X_valid, y_valid, X_oot, y_oot，
     然后 %run "5. 模型训练.py"

所有参数从 config/model_param_thresholds.yaml 读取，不再硬编码。
"""
import os
import sys
import json
import time
import joblib
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization

# ---- LightGBM API 版本兼容层 ----
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from lgb_fit_compat import safe_lgbm_fit

try:
    import yaml
except ImportError:
    yaml = None

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置加载（复用 main_step3 的配置逻辑）
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_yaml_safe(path):
    """安全加载 YAML，不存在则返回空字典"""
    if not Path(path).exists():
        print("  [WARN] Config not found: %s" % path)
        return {}
    with open(str(path), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_model_params(base_dir=None):
    """从 model_param_thresholds.yaml 加载模型参数配置"""
    if base_dir is None:
        base_dir = str(Path(SCRIPT_DIR).parent.parent / "config")
    cfg_path = os.path.join(base_dir, "model_param_thresholds.yaml")
    cfg = _load_yaml_safe(cfg_path)

    defaults = {
        "bayesian_optimization": {"n_runs": 5, "init_points": 10, "n_iter": 30,
                                  "random_state": 42, "loss_function": "binary"},
        "lgbm_search_space": {
            "num_leaves": {"min": 2, "max": 31},
            "learning_rate": {"min": 0.01, "max": 0.5},
            "n_estimators": {"min": 500, "max": 1000},
            "feature_fraction": {"min": 0.4, "max": 0.9},
            "bagging_fraction": {"min": 0.4, "max": 0.95},
            "min_child_samples": {"min": 100, "max": 256},
            "reg_alpha": {"min": 0.1, "max": 10.0},
            "reg_lambda": {"min": 0.1, "max": 10.0},
        },
        "data_split": {"oot_months_tail": 4, "val_ratio": 0.3, "random_state": 42},
        "early_stopping": {"enabled": True, "patience": 100},
        "overfitting_penalty": {"ks_threshold": 0.02, "penalty_factor": 300},
        "cross_validation": {"n_splits": 5, "shuffle": True, "random_state": 42},
    }
    if not cfg:
        return defaults

    for section, default_val in defaults.items():
        if section not in cfg:
            cfg[section] = default_val
        elif isinstance(default_val, dict):
            for k, v in default_val.items():
                if k not in cfg[section]:
                    cfg[section][k] = v

    return cfg


# ============================================================
# 配置读取 — 所有参数来自 YAML
# ============================================================
print("=" * 60)
print("加载模型配置 ...")
print("=" * 60)

model_params = load_model_params()

bo_cfg = model_params.get("bayesian_optimization", {})
space = model_params.get("lgbm_search_space", {})
es_cfg = model_params.get("early_stopping", {})
of_penalty_cfg = model_params.get("overfitting_penalty", {})
cv_cfg = model_params.get("cross_validation", {})

# 贝叶斯优化参数
n_runs = bo_cfg.get("n_runs", 5)
init_points = bo_cfg.get("init_points", 10)
n_iter = bo_cfg.get("n_iter", 30)
cfg_random_state = bo_cfg.get("random_state", 42)

# 搜索空间 → pbounds
pbounds_from_cfg = {
    "num_leaves": (space.get("num_leaves", {}).get("min", 2), space.get("num_leaves", {}).get("max", 31)),
    "learning_rate": (space.get("learning_rate", {}).get("min", 0.01), space.get("learning_rate", {}).get("max", 0.5)),
    "min_child_samples": (space.get("min_child_samples", {}).get("min", 100), space.get("min_child_samples", {}).get("max", 256)),
    "subsample": (space.get("bagging_fraction", {}).get("min", 0.4), space.get("bagging_fraction", {}).get("max", 0.95)),
    "colsample_bytree": (space.get("feature_fraction", {}).get("min", 0.4), space.get("feature_fraction", {}).get("max", 0.9)),
    "reg_alpha": (space.get("reg_alpha", {}).get("min", 0.1), space.get("reg_alpha", {}).get("max", 10.0)),
    "reg_lambda": (space.get("reg_lambda", {}).get("min", 0.1), space.get("reg_lambda", {}).get("max", 10.0)),
}
nl = space.get("num_leaves", {})
pbounds_from_cfg["max_depth"] = (max(nl.get("min", 2), 2), min(nl.get("max", 31) // 2, 16))

# 早停参数
es_enabled = es_cfg.get("enabled", True)
es_rounds = es_cfg.get("patience", 100) if es_enabled else 100

# 过拟合惩罚参数
ks_threshold = of_penalty_cfg.get("ks_threshold", 0.02)
penalty_factor = of_penalty_cfg.get("penalty_factor", 300)

# 交叉验证参数
cv_n_splits = cv_cfg.get("n_splits", 5)

# n_estimators
lgbm_n_estimators = space.get("n_estimators", {}).get("max", 1000)

print("  bayesian_optimization: n_runs=%d, init_points=%d, n_iter=%d, random_state(base)=%d" % (
    n_runs, init_points, n_iter, cfg_random_state))
print("  pbounds: %s" % str(pbounds_from_cfg))
print("  early_stopping: rounds=%d (enabled=%s)" % (es_rounds, es_enabled))
print("  overfitting_penalty: threshold=%.4f, factor=%d" % (ks_threshold, penalty_factor))
print("  cross_validation: n_splits=%d" % cv_n_splits)
print("  n_estimators: %d" % lgbm_n_estimators)


# ============================================================
# KS / AUC 评估函数
# ============================================================
def ks_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def evaluate_ks_auc(y_true, y_pred_prob):
    auc = roc_auc_score(y_true, y_pred_prob)
    ks = ks_score(y_true, y_pred_prob)
    return {"auc": round(auc, 6), "ks": round(ks, 6)}


# ============================================================
# 全局变量（供贝叶斯优化闭包使用）
# ============================================================
X_train_g = None
y_train_g = None
X_oot_g = None

# 以下在闭包中引用
ks_overfitting_gap_val = ks_threshold


# ============================================================
# 贝叶斯优化目标函数（5折 CV + 过拟合惩罚）
# ============================================================
def lgb_cv(num_leaves, max_depth, learning_rate, min_child_samples,
           subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": lgbm_n_estimators,
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

    skf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_train_g, y_train_g):
        X_tr, X_val = X_train_g.iloc[train_idx], X_train_g.iloc[val_idx]
        y_tr, y_val = y_train_g.iloc[train_idx], y_train_g.iloc[val_idx]

        model = LGBMClassifier(**params)
        safe_lgbm_fit(model, X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       early_stopping_rounds=es_rounds, verbose=False)

        y_pred = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred))
        del model

    mean_auc = np.mean(aucs)

    # ---- Train-OOT KS 差异惩罚 ----
    model_full = LGBMClassifier(**params)
    safe_lgbm_fit(model_full, X_train_g, y_train_g, verbose=False)
    y_train_pred = model_full.predict_proba(X_train_g)[:, 1]
    ks_train = ks_score(y_train_g, y_train_pred)

    if X_oot_g is not None:
        y_oot_pred = model_full.predict_proba(X_oot_g)[:, 1]
        ks_oot = ks_score(y_oot_g, y_oot_pred)
    else:
        ks_oot = ks_train  # 无 OOT 数据时跳过惩罚

    ks_diff = abs(ks_train - ks_oot)
    penalty = max(0, (ks_diff - ks_overfitting_gap_val)) * penalty_factor

    score = mean_auc - penalty
    print("  AUC: %.4f, KS_train: %.4f, KS_oot: %.4f, KS_diff: %.4f, penalty: %.4f, final score: %.4f" % (
        mean_auc, ks_train, ks_oot, ks_diff, penalty, score))

    return score


# ============================================================
# 主流程 — 多次贝叶斯优化，每轮保存最佳模型
# ============================================================
print("\n" + "=" * 60)
print("开始 %d 轮贝叶斯优化训练 ..." % n_runs)
print("=" * 60)

# ---- X_train, y_train, X_valid, y_valid, X_oot, y_oot 需由外部（notebook）准备 ----
for _var_name in ('X_train', 'y_train', 'X_valid', 'y_valid'):
    if _var_name not in dir():
        print("\n[ERROR] 请先在 Notebook 中准备以下变量:")
        print("  X_train, y_train, X_valid, y_valid")
        print("  （可选）X_oot — 提供 OOT 数据用于过拟合惩罚")
        print("  然后再 %run 此脚本")
        raise SystemExit(1)

# ---- 注入全局变量供闭包使用 ----
X_train_g = X_train
y_train_g = y_train
if 'X_oot' in dir():
    X_oot_g = X_oot
else:
    X_oot_g = None
    print("\n[INFO] 未检测到 X_oot，跳过 OOT 过拟合惩罚")

# ---- 输出目录 ----
out_dir = Path(os.getcwd()) / "models"
out_dir.mkdir(parents=True, exist_ok=True)

round_records = []   # list of (run_idx, auc_valid, pkl_path, json_path)

for run_idx in range(n_runs):
    print("\n" + "-" * 50)
    print("第 %d / %d 轮贝叶斯优化" % (run_idx + 1, n_runs))
    print("-" * 50)

    # ---- 随机种子策略：以配置的 random_state 为 base，每轮加上 [-41, 200] 的随机偏移 ----
    rng = np.random.RandomState(cfg_random_state + run_idx)
    run_seed = rng.randint(-41, 201)
    bo_random_state = cfg_random_state + run_seed
    print("  BO seed: base=%d, offset=%d, final=%d" % (cfg_random_state, run_seed, bo_random_state))

    optimizer = BayesianOptimization(
        f=lgb_cv,
        pbounds=pbounds_from_cfg,
        random_state=bo_random_state,
    )

    t0 = time.perf_counter()
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print("第 %d 轮优化耗时: %.2fs" % (run_idx + 1, time.perf_counter() - t0))

    best_params_run = optimizer.max["params"]

    # ---- 用本轮最优参数训练，在验证集评估 ----
    p = dict(best_params_run)
    p["num_leaves"] = int(p["num_leaves"])
    p["max_depth"] = int(p["max_depth"])
    p["min_child_samples"] = int(p["min_child_samples"])
    p["objective"] = "binary"
    p["boosting_type"] = "gbdt"
    p["n_estimators"] = lgbm_n_estimators
    p["random_state"] = bo_random_state
    p["n_jobs"] = -1

    run_model = LGBMClassifier(**p)
    safe_lgbm_fit(run_model, X_train, y_train,
                   eval_set=[(X_valid, y_valid)],
                   early_stopping_rounds=es_rounds, verbose=False)

    # ---- 对本轮模型做完整评估：Train / Valid / OOT ----
    m_train = evaluate_ks_auc(y_train, run_model.predict_proba(X_train)[:, 1])
    m_valid = evaluate_ks_auc(y_valid, run_model.predict_proba(X_valid)[:, 1])
    if X_oot_g is not None:
        m_oot = evaluate_ks_auc(y_oot, run_model.predict_proba(X_oot)[:, 1])
    else:
        m_oot = None

    print("第 %d 轮评估: Train AUC=%.4f KS=%.4f | Valid AUC=%.4f KS=%.4f" % (
        run_idx + 1, m_train['auc'], m_train['ks'], m_valid['auc'], m_valid['ks']))
    if m_oot is not None:
        print("           OOT  AUC=%.4f KS=%.4f" % (m_oot['auc'], m_oot['ks']))

    auc_v = m_valid['auc']
    ks_gap = abs(m_train['ks'] - (m_oot['ks'] if m_oot else m_valid['ks']))

    # ---- 保存本轮模型 & 参数 & 评估结果 ----
    pkl_path = out_dir / ("it_%d_model.pkl" % (run_idx + 1))
    json_path = out_dir / ("it_%d_params.json" % (run_idx + 1))
    eval_path  = out_dir / ("it_%d_eval.json" % (run_idx + 1))
    joblib.dump(run_model, str(pkl_path))
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)
    # 保存本轮评估结果
    round_eval = {
        "round": run_idx + 1,
        "train": m_train,
        "valid": m_valid,
        "oot": m_oot,
        "ks_gap": round(ks_gap, 6),
    }
    with open(str(eval_path), "w", encoding="utf-8") as f:
        json.dump(round_eval, f, indent=2)

    # ---- 记录：(轮次, valid_auc, pkl路径, 评估字典) ----
    round_records.append((run_idx + 1, auc_v, str(pkl_path), m_train, m_valid, m_oot, ks_gap))
    print("  Saved: %s, %s" % (pkl_path.name, eval_path.name))

    # ---- 释放本轮对象 ----
    del optimizer, best_params_run, p, run_model, y_vp, auc_v
    gc.collect()


# ============================================================
# 比较所有轮次，每轮模型的完整评估汇总
# ============================================================
print("\n" + "=" * 70)
print("每轮模型评估汇总 (%d 轮)" % len(round_records))
print("=" * 70)

header = "%-6s  %-12s  %-12s  %-12s  %-8s"
if X_oot_g is not None:
    header += "  %-12s"
print(header % ("Round", "Train(AUC/KS)", "Valid(AUC/KS)",
                "OOT(AUC/KS)" if X_oot_g else "(skip OOT)", "KS_gap"))

for r_idx, auc_v, pkl_p, mt, mv, mo, kg in round_records:
    train_s = "%.4f/%.4f" % (mt['auc'], mt['ks'])
    valid_s = "%.4f/%.4f" % (mv['auc'], mv['ks'])
    row = "%-6d  %-12s  %-12s" % (r_idx, train_s, valid_s)
    if mo is not None:
        oot_s = "%.4f/%.4f" % (mo['auc'], mo['ks'])
        row += "  %-12s  %.6f" % (oot_s, kg)
    else:
        row += "  N/A             %.6f" % kg
    print(row)

# 按验证集 AUC 选出最佳轮次
best_entry = max(round_records, key=lambda x: x[1])
best_round, best_auc_valid, best_pkl_path, _, _, _, _ = best_entry
print("\n>>> 最终最佳: 第 %d 轮 (Valid AUC=%.4f)" % (best_round, best_auc_valid))

# ---- 加载最终最佳模型 ----
final_model = joblib.load(best_pkl_path)
with open(str(Path(out_dir) / ("it_%d_params.json" % best_round)), "r", encoding="utf-8") as f:
    final_params = json.load(f)

# ---- 保存最终模型 ----
final_pkl = out_dir / "final_model.pkl"
joblib.dump(final_model, str(final_pkl))
print("Final model saved: %s" % final_pkl)

# ---- 保存所有轮次评估汇总表 ----
summary_rows = []
for r_idx, auc_v, pkl_p, mt, mv, mo, kg in round_records:
    row = {"round": r_idx, "train": mt, "valid": mv, "oot": mo, "ks_gap": kg}
    summary_rows.append(row)
summary_path = out_dir / "round_eval_summary.json"
with open(str(summary_path), "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, indent=2)
print("Round eval summary saved: %s" % summary_path)


# ============================================================
# 最终评估
# ============================================================
print("\n" + "=" * 60)
print("最终最佳模型评估")
print("=" * 60)

print("\n====== 训练集评估 ======")
y_train_pred = final_model.predict_proba(X_train)[:, 1]
print(evaluate_ks_auc(y_true=y_train, y_pred_prob=y_train_pred))

print("\n====== 验证集评估 ======")
y_valid_pred = final_model.predict_proba(X_valid)[:, 1]
print(evaluate_ks_auc(y_true=y_valid, y_pred_prob=y_valid_pred))

print("\n====== OOT集评估 ======")
if X_oot_g is not None:
    y_oot_pred = final_model.predict_proba(X_oot)[:, 1]
    print(evaluate_ks_auc(y_true=y_oot, y_pred_prob=y_oot_pred))
else:
    print("[跳过] 未提供 OOT 数据")

print("\n最佳参数:")
for k, v in sorted(final_params.items()):
    print("  %s: %s" % (k, v))

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)