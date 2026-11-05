"""
Pipeline 配置与状态管理（合并 = manifest_utils + common_utils）

职责：
  - manifest.json CRUD & 步骤状态管理
  - YAML 配置文件加载（data_config / model_params / thresholds）
  - 数据源推断、建模任务生成
  - 智能数据读取（smart_read_df）
  - 通用辅助函数（make_safe_task_id, collect_gc, ensure_dirs）
"""

from __future__ import print_function, unicode_literals

import gc
import glob
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================
# YAML 加载器（统一入口）
# ============================================================

def _try_yaml():
    """Try importing PyYAML, fall back to ruamel.yaml"""
    try:
        import yaml
        return yaml
    except ImportError:
        pass
    try:
        from ruamel import yaml
        return yaml
    except ImportError:
        raise RuntimeError("Neither PyYAML nor ruamel.yaml is installed.")


def _load_yaml_file(path: Path, default: Any = None) -> Any:
    """Load a YAML file with graceful fallback."""
    if not path.exists():
        return default if default is not None else {}
    try:
        yaml_mod = _try_yaml()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml_mod.safe_load(f)
        return data or (default if default is not None else {})
    except Exception:
        return default if default is not None else {}


# ============================================================
# 默认模板
# ============================================================

DEFAULT_DATA_CONFIG = {
    "key_col": "",
    "labels": {},
    "data_sources": {},
    "analysis_dims": {
        "month_col": "month",
        "group_by_cols": [],
        "external_scores": [],
    },
    # ⚠️ drop_cols 必须从 data_config.yaml 读取，绝不能写死！
    "drop_cols": [],
}

# ---- 数据源 data_split 默认值 ----
DEFAULT_DATA_SPLIT = {
    "oot_tail_months": 4,
    "val_ratio": 0.3,
    "random_state": 42,
}

DEFAULT_MANIFEST = {
    "project_name": "",
    "skill_version": "2.0",
    "parameters": {
        "label_col": "y",
        "month_col": "month",
        "raw_data_path": "",
        "extra_score_cols": [],
        "feature_cols_source": "wide_table_full.parquet",
        # ⚠️ key_col 和 drop_cols 必须从 data_config.yaml 读取，绝不能写死！
        "key_col": "",
        "drop_cols": [],
        "x_cols": [],
    },
    "data_config_file": "config/data_config.yaml",
    "steps": {
        "step1_eda": {
            "status": "pending",
            "depends_on": [],
            "inputs": [],
            "outputs": [
                "data/processed/wide_table_full.parquet",
                "data/processed/wide_table_sampled.parquet",
                "data/processed/missing_rate.xlsx",
                "data/processed/label_distribution.xlsx",
                "data/processed/psi_monthly.xlsx",
            ],
        },
        "step2_feature_select": {
            "status": "pending",
            "depends_on": ["step1_eda"],
            "inputs": [
                "data/processed/wide_table_sampled.parquet",
                "data/processed/missing_rate.xlsx",
                "data/processed/psi_monthly.xlsx",
            ],
            "outputs": [
                "data/processed/selected_features.csv",
                "data/processed/iv_summary.xlsx",
                "data/processed/correlation_heatmap.png",
            ],
        },
        "step3_modeling": {
            "status": "pending",
            "depends_on": ["step2_feature_select"],
            "inputs": [
                "data/processed/wide_table_full.parquet",
                "data/processed/selected_features.csv",
            ],
            "outputs": [
                "models/final_model.pkl",
                "models/best_params.json",
            ],
        },
        "step4_eval": {
            "status": "pending",
            "depends_on": ["step3_modeling"],
            "inputs": [
                "data/processed/wide_table_full.parquet",
                "data/processed/selected_features.csv",
                "models/final_model.pkl",
            ],
            "outputs": [
                "reports/model_ks_auc_by_month.xlsx",
                "reports/feature_importance.csv",
                "reports/monthly_bin_bad_rate.xlsx",
            ],
        },
        "step5_report": {
            "status": "pending",
            "depends_on": ["step4_eval"],
            "inputs": [
                "data/processed/wide_table_full.parquet",
                "data/processed/selected_features.csv",
                "models/final_model.pkl",
                "reports/feature_importance.csv",
            ],
            "outputs": [
                "reports/final_report.xlsx",
            ],
        },
    },
}

STEP_ORDER = {
    "step1_eda": (1, "Step1: 深度数据探索与特征体检 (Deep EDA)"),
    "step2_feature_select": (2, "Step2: 多维度特征筛选 (Feature Selection)"),
    "step3_modeling": (3, "Step3: 贝叶斯优化建模 (Bayesian Modeling)"),
    "step4_eval": (4, "Step4: 严苛的模型评估 (Strict Evaluation)"),
    "step5_report": (5, "Step5: 结案报告 (Final Report)"),
}


# ============================================================
# Manifest CRUD
# ============================================================

def init_manifest(
    project_dir: Path,
    label_col: str = "y",
    month_col: str = "month",
    raw_data_path: str = "",
    extra_score_cols: Optional[List[str]] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest_path = project_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("parameters", {})
        manifest["parameters"]["label_col"] = label_col
        manifest["parameters"]["month_col"] = month_col
        if raw_data_path:
            manifest["parameters"]["raw_data_path"] = raw_data_path
        if extra_score_cols:
            manifest["parameters"]["extra_score_cols"] = extra_score_cols
    else:
        manifest = deepcopy(DEFAULT_MANIFEST)
        manifest["project_name"] = project_dir.name
        manifest["parameters"]["label_col"] = label_col
        manifest["parameters"]["month_col"] = month_col
        manifest["parameters"]["raw_data_path"] = raw_data_path
        if extra_score_cols:
            manifest["parameters"]["extra_score_cols"] = extra_score_cols

    if data_config:
        params = manifest.setdefault("parameters", {})
        params["key_col"] = data_config.get("key_col", "")
        params["drop_cols"] = data_config.get("drop_cols", params.get("drop_cols", []))
        labels_pool = data_config.get("labels", {})
        if labels_pool:
            params["labels"] = labels_pool
        sources = data_config.get("data_sources", {})
        if sources:
            params["data_sources"] = sources
        dims = data_config.get("analysis_dims", {})
        gb = dims.get("group_by_cols", [])
        ext_scores = dims.get("external_scores", [])
        if gb:
            params["group_by_cols"] = gb
        if ext_scores:
            params["external_scores"] = ext_scores

    _patch_model_labels(manifest, label_col)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_manifest(project_dir: Path) -> Dict[str, Any]:
    manifest_path = project_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {project_dir}. "
            "Please run driver.py to initialize first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(project_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = project_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def mark_step_done(
    manifest: Dict[str, Any],
    step_key: str,
    project_dir: Optional[Path] = None,
) -> None:
    manifest["steps"][step_key]["status"] = "done"
    if project_dir:
        save_manifest(project_dir, manifest)


def mark_step_failed(
    manifest: Dict[str, Any],
    step_key: str,
    error_msg: str = "",
    project_dir: Optional[Path] = None,
) -> None:
    manifest["steps"][step_key]["status"] = "failed"
    if error_msg:
        manifest["steps"][step_key]["error"] = error_msg
    if project_dir:
        save_manifest(project_dir, manifest)


def get_step_status(manifest: Dict[str, Any], step_key: str) -> str:
    return manifest["steps"].get(step_key, {}).get("status", "pending")


def check_step_ready(
    manifest: Dict[str, Any],
    step_key: str,
    project_dir: Path,
) -> Tuple[bool, str]:
    step = manifest["steps"].get(step_key)
    if not step:
        return False, f"Unknown step key: {step_key}"

    for dep in step.get("depends_on", []):
        dep_status = get_step_status(manifest, dep)
        if dep_status != "done":
            dep_name = STEP_ORDER.get(dep, (0, dep))[1]
            return False, (
                f"Dependency {dep_name} is '{dep_status}'. "
                "Please complete it first."
            )

    for inp in step.get("inputs", []):
        inp_path = project_dir / inp
        if not inp_path.exists():
            # ⚠️ 多标签场景下，文件名带任务后缀，用 glob 匹配
            # 支持的模式：selected_features.csv, feature_importance.csv,
            # final_model_*.pkl, score_eval_*.xlsx, monthly_bin_bad_rate.xlsx
            basename = inp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            directory = inp.rsplit("/", 1)[0] if "/" in inp else (
                inp.rsplit("\\", 1)[0] if "\\" in inp else ""
            )

            # 模式1: selected_features.csv → selected_features_*.csv
            if basename == "selected_features.csv":
                glob_pattern = os.path.join(project_dir, directory, "selected_features_*.csv")
            # 模式2: feature_importance.csv → reports/*/feature_importance.csv
            elif basename == "feature_importance.csv" and "reports" in inp:
                glob_pattern = os.path.join(project_dir, "reports", "*", "feature_importance.csv")
            # 模式3: final_model_y.pkl → final_model_*.pkl
            elif basename.startswith("final_model_") and basename.endswith(".pkl"):
                glob_pattern = os.path.join(project_dir, "models", "final_model_*.pkl")
            # 模式4: score_eval_*.xlsx
            elif "score_eval" in basename:
                glob_pattern = os.path.join(project_dir, "reports", "*", "score_eval_*.xlsx")
            else:
                glob_pattern = None

            if glob_pattern is not None:
                matches = list(glob.glob(glob_pattern))
                if not matches:
                    return False, f"Required input file missing: {inp} (tried glob: {glob_pattern})"
            else:
                return False, f"Required input file missing: {inp}"

    return True, "Ready"


def get_pending_steps(manifest: Dict[str, Any]) -> List[str]:
    pending = []
    for step_key in STEP_ORDER:
        status = get_step_status(manifest, step_key)
        if status != "done":
            pending.append(step_key)
    return pending


def get_next_step(manifest: Dict[str, Any]) -> Optional[str]:
    for step_key in STEP_ORDER:
        if get_step_status(manifest, step_key) != "done":
            return step_key
    return None


def get_artifacts_for_step(
    manifest: Dict[str, Any],
    step_key: str,
    project_dir: Path,
) -> Dict[str, Path]:
    outputs = manifest["steps"].get(step_key, {}).get("outputs", [])
    return {Path(out).name: project_dir / out for out in outputs}


def _patch_model_labels(manifest: Dict[str, Any], label_col: str) -> None:
    safe_label = label_col.replace(" ", "_").replace("/", "_")

    for out in manifest["steps"]["step3_modeling"].get("outputs", []):
        if "final_model" in out:
            idx = manifest["steps"]["step3_modeling"]["outputs"].index(out)
            manifest["steps"]["step3_modeling"]["outputs"][idx] = (
                f"models/final_model_{safe_label}.pkl"
            )

    for step_key in ("step4_eval", "step5_report"):
        for i, inp in enumerate(manifest["steps"][step_key].get("inputs", [])):
            if "final_model" in inp:
                manifest["steps"][step_key]["inputs"][i] = (
                    f"models/final_model_{safe_label}.pkl"
                )

    for i, out in enumerate(manifest["steps"]["step5_report"].get("outputs", [])):
        if "final_report" in out:
            manifest["steps"]["step5_report"]["outputs"][i] = (
                f"reports/final_report_{safe_label}.xlsx"
            )


# ============================================================
# 数据表配置加载 (data_config.yaml)
# ============================================================

def load_data_config(
    project_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=base_dir,
        filename="data_config.yaml",
    )
    cfg = _load_yaml_file(target, dict(DEFAULT_DATA_CONFIG))

    merged = dict(DEFAULT_DATA_CONFIG)
    for key in merged:
        if key in cfg:
            merged[key] = cfg[key]

    return merged


def infer_feature_cols(df, data_config: Dict[str, Any], source_key: str = None) -> List[str]:
    if source_key:
        sources = data_config.get("data_sources", {})
        src = sources.get(source_key, {})
        x_cols = src.get("x_cols", [])
    else:
        x_cols = data_config.get("x_cols", [])

    if x_cols:
        return [c for c in x_cols if c in df.columns]

    month_col = data_config.get("analysis_dims", {}).get("month_col", "month")
    group_by = data_config.get("analysis_dims", {}).get("group_by_cols", [])
    ext_scores = data_config.get("analysis_dims", {}).get("external_scores", [])
    drop_cols = data_config.get("drop_cols", [])

    exclude = {month_col}
    exclude.update(group_by)
    exclude.update(ext_scores)
    exclude.update(drop_cols)

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    return feature_cols


def get_modeling_tasks(data_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels_pool = data_config.get("labels", {})
    sources = data_config.get("data_sources", {})

    if not sources:
        global_label_col = data_config.get("label_col", "y")
        return [{
            "source_key": "default",
            "source_name": "默认数据源",
            "label_key": global_label_col,
            "label_col": global_label_col,
            "x_cols": data_config.get("x_cols", []),
            "task_id": "default_" + global_label_col,
        }]

    tasks = []
    for src_key, src_cfg in sources.items():
        src_name = src_cfg.get("name", src_key)
        src_labels = src_cfg.get("labels", [])
        src_x_cols = src_cfg.get("x_cols", [])

        for label_key in src_labels:
            label_def = labels_pool.get(label_key, {})
            label_col = label_def.get("col_name", label_key)

            tasks.append({
                "source_key": src_key,
                "source_name": src_name,
                "label_key": label_key,
                "label_col": label_col,
                "x_cols": src_x_cols,
                "task_id": "%s_%s" % (src_key, label_key),
            })

    return tasks


def get_data_sources(data_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    获取数据源列表（去重，不含label维度）
    用于 PSI/IV 等全量统计，每个数据源只跑一次。
    """
    sources = []
    sources_cfg = data_config.get("data_sources", {})
    for source_key, source_val in sources_cfg.items():
        sources.append({
            "source_key": source_key,
            "source_name": source_val.get("name", source_key),
            "x_cols": source_val.get("x_cols", []),
        })
    return sources


def resolve_label_col(data_config: Dict[str, Any], label_key: str) -> str:
    labels_pool = data_config.get("labels", {})
    label_def = labels_pool.get(label_key, {})
    return label_def.get("col_name", label_key)


def resolve_positive_class(data_config: Dict[str, Any], label_key: str) -> int:
    labels_pool = data_config.get("labels", {})
    label_def = labels_pool.get(label_key, {})
    return label_def.get("positive_class", 1)


# ============================================================
# 智能数据读取
# ============================================================

def smart_read_df(
    data_path,
    dtype=None,
    usecols=None,
    nrows: Optional[int] = None,
):
    import pandas as pd
    ext = Path(str(data_path).split("?")[0]).suffix.lower()

    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else None
        return pd.read_csv(data_path, dtype=dtype, usecols=usecols, nrows=nrows, sep=sep)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(data_path, engine="openpyxl" if ext == ".xlsx" else None)
    elif ext == ".parquet":
        if usecols is not None:
            import warnings
            warnings.warn("usecols 在 parquet 中由 read_parquet(columns=...) 替代")
            return pd.read_parquet(data_path, columns=usecols)
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"不支持的数据格式: {ext} (支持 .csv/.tsv/.xlsx/.xls/.parquet)")


# ============================================================
# 模型参数 & 评估阈值加载
# ============================================================

def get_default_model_params() -> Dict[str, Any]:
    return {
        "bayesian_optimization": {
            "n_runs": 5,
            "init_points": 10,
            "n_iter": 30,
            "random_state": 42,
            "loss_function": "binary",
        },
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
        "data_split": {
            "oot_months_tail": 4,
            "val_ratio": 0.3,
            "random_state": 42,
        },
        "early_stopping": {"enabled": True, "patience": 100},
        "overfitting_penalty": {
            "ks_threshold": 0.02,
            "penalty_factor": 300,
        },
        "cross_validation": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },
    }


def _get_skill_dir(skill_dir, base_dir):
    """Resolve skill_dir: explicit > base_dir fallback."""
    if skill_dir is not None:
        return Path(skill_dir)
    if base_dir is not None:
        return Path(base_dir)
    return resolve_base_dir()


def _resolve_config_path(
    project_dir: Optional[Path],
    skill_dir: Optional[Path],
    filename: str,
) -> Path:
    """Resolve a config file path with priority: project_dir > skill_dir > scripts parent."""
    candidates = []
    if project_dir is not None:
        candidates.append(project_dir / "config" / filename)
    if skill_dir is not None:
        candidates.append(skill_dir / "config" / filename)
    # Fallback to scripts' grandparent (default skill root)
    candidates.append(resolve_base_dir() / "config" / filename)
    for cp in candidates:
        if cp.exists():
            return cp
    # Return the first candidate even if it doesn't exist, so _load_yaml_file returns default
    return candidates[0]


def load_model_params(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    defaults = get_default_model_params()
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="model_param_thresholds.yaml",
    )
    return _load_yaml_file(target, defaults)


# --- Evaluation thresholds ---

def get_default_eval_thresholds() -> Dict[str, Any]:
    return {
        "ks_overfitting_gap": 0.05,
        "auc_min_acceptable": 0.55,
        "auc_min_excellent": 0.65,
        "ks_min_good": 0.25,
    }


def load_evaluation_thresholds(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    defaults = get_default_eval_thresholds()
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="report_thresholds.yaml",
    )
    raw = _load_yaml_file(target, {})
    return raw.get("evaluation", defaults)


# --- Stability thresholds ---

def get_default_stability_thresholds() -> Dict[str, Any]:
    return {
        "psi_max_acceptable": 0.25,
        "psi_warning": 0.10,
    }


def load_stability_thresholds(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    defaults = get_default_stability_thresholds()
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="report_thresholds.yaml",
    )
    raw = _load_yaml_file(target, {})
    return raw.get("stability", defaults)


# --- Monthly evaluation thresholds ---

def get_default_monthly_eval() -> Dict[str, Any]:
    return {
        "auc_min_by_month": 0.51,
        "ks_min_by_month": 0.15,
    }


def load_monthly_evaluation(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    defaults = get_default_monthly_eval()
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="report_thresholds.yaml",
    )
    raw = _load_yaml_file(target, {})
    return raw.get("monthly_evaluation", defaults)


# --- Feature selection thresholds ---

def get_default_feature_select_thresholds() -> Dict[str, Any]:
    return {
        "missing_rate_max": 0.85,
        "psi_max": 0.25,
        "iv_min": 0.01,
        "corr_threshold": 0.80,
        "feat_count_min": 200,
        "feat_count_max": 250,
    }


def load_feature_select_thresholds(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    defaults = get_default_feature_select_thresholds()
    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="feature_select_thresholds.yaml",
    )
    raw = _load_yaml_file(target, {})
    return raw.get("feature_selection", defaults)


# --- Backward-compatible alias ---

def load_thresholds(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    """Backward-compatible alias → evaluation thresholds."""
    return load_evaluation_thresholds(
        project_dir=project_dir, skill_dir=skill_dir, base_dir=base_dir
    )


# ============================================================
# EDA 采样 & 变量配置默认值
# ============================================================

def get_default_sampling_config() -> Dict[str, Any]:
    return {
        "max_full_features": 500,
        "max_full_rows": 200000,
        "sample_size": 100000,
        "random_seed": 42,
        "max_corr_features": 800,
    }


def get_default_variable_eda_config() -> Dict[str, Any]:
    return {
        "min_value_pct_threshold": 0.02,
    }


def load_eda_sampling(
    project_dir: Optional[Union[str, Path]] = None,
    skill_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,  # backward compat alias for skill_dir
) -> Dict[str, Any]:
    cfg = get_default_sampling_config()

    target = _resolve_config_path(
        project_dir=Path(project_dir) if project_dir else None,
        skill_dir=_get_skill_dir(skill_dir, base_dir),
        filename="result_thresholds.yaml",
    )
    raw = _load_yaml_file(target, {})
    step1_eda = raw.get("step1_eda", {}) if isinstance(raw, dict) else {}
    sampling = step1_eda.get("sampling", {}) if isinstance(step1_eda, dict) else {}
    cfg.update(sampling)
    return cfg


# ============================================================
# 通用辅助函数
# ============================================================

def ensure_dirs(project_dir: Path) -> None:
    for sub in [
        "data/raw", "data/processed", "models", "reports",
        "figures", "logs", "config",
    ]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)


def make_safe_task_id(task_id: Optional[str]) -> str:
    if not task_id:
        return ""
    safe = task_id.replace(" ", "_").replace("/", "_")
    return "_" + safe


def resolve_base_dir(script_file: Union[str, Path, None] = None) -> Path:
    if script_file is not None:
        return Path(script_file).resolve().parent.parent
    return Path(__file__).resolve().parent.parent


def safe_json_dump(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def collect_gc(label: str = "") -> None:
    """释放内存并打印简要提示（向后兼容：label有默认值）。"""
    gc.collect()
    if label:
        print(f"[GC] {label}")


def smart_release(*objs, label: str = "") -> None:
    """显式清理传入的对象引用并触发 GC。
    
    用法示例：
        df_full = pd.read_parquet(...)
        ...  # work ...
        smart_release(df_full, label="wide_table released")
        
        cache_dict = {...}
        smart_release(cache_dict, label="cache cleared")
    """
    for obj in objs:
        if isinstance(obj, dict):
            obj.clear()
    gc.collect()
    if label:
        print(f"[GC] {label}")
