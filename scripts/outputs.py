"""
Output configuration utilities for fintech-risk-control pipeline.

Loads result_thresholds.yaml and provides helpers:
  - load_output_config()       → full config dict
  - get_step_config(step_key)  → step-level sub-dict
  - resolve_filename(key, task_id=None, output_dir=None) → full file path
  - is_function_enabled(step_key, func_name) → bool
  - should_generate(step_key, output_key)   → bool
"""

from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------------------------------------------
# Default configuration (fallback when YAML missing or incomplete)
# ------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "run_step1_eda": True,
        "run_step2_feature_select": True,
        "run_step3_modeling": True,
        "run_step4_evaluation": True,
        "run_step5_report": True,
    },
    "step1_eda": {
        "enabled": True,
        "sampling": {
            "max_full_features": 500,
            "max_full_rows": 200000,
            "sample_size": 100000,
            "random_seed": 42,
        },
        "variable_eda": {
            "special_values": [0],
            "compute_unique": False,
            "unique_to_char_threshold": 20,
        },
        "functions": {
            "missing_rate": True,
            "label_distribution": True,
            "eda_basic": True,
            "x_y_cross_eda": True,
            "eda_variable_detail": True,
            "compute_psi": True,
            "compute_iv": True,
            "correlation_check": True,
        },
        "outputs": {
            "wide_table": {"filename": "wide_table", "format": "parquet", "generate": True},
            "preprocessed_data": {"filename": "preprocessed_data", "format": "csv", "generate": True},
            "missing_rate": {"filename": "missing_rate", "format": "xlsx", "generate": True},
            "label_distribution": {"filename": "label_distribution", "format": "xlsx", "generate": True},
            "basic_summary_all": {"filename": "eda_basic_summary_all", "format": "xlsx", "generate": True},
            "basic_summary_by_month": {"filename": "eda_basic_summary_month", "format": "xlsx", "generate": True},
            "basic_summary_by_group": {"filename": "eda_basic_summary_group", "format": "xlsx", "generate": True},
            "variable_eda_numeric": {"filename": "eda_variable_numeric", "format": "xlsx", "generate": True},
            "variable_eda_categorical": {"filename": "eda_variable_categorical", "format": "xlsx", "generate": True},
            "variable_eda_num_to_char_list": {"filename": "eda_num_to_char_vars", "format": "csv", "generate": False},
            "psi_summary": {"filename": "psi_summary", "format": "xlsx", "generate": True},
            "iv_summary": {"filename": "iv_summary", "format": "xlsx", "generate": True},
            "correlation_matrix": {"filename": "correlation_matrix", "format": "xlsx", "generate": True},
            "correlation_heatmap": {"filename": "correlation_heatmap", "format": "png", "generate": False},
        },
    },
    "step2_feature_select": {
        "enabled": True,
        "functions": {
            "hard_filter": True,
            "psi_filter": True,
            "corr_filter": True,
            "corr_dedup": True,
            "generate_heatmap": False,
        },
        "outputs": {
            "selected_features": {"filename": "selected_features", "format": "csv", "generate": True},
        },
    },
    "step3_modeling": {
        "enabled": True,
        "functions": {
            "bayesian_optimization": True,
            "save_per_round": True,
            "early_stopping": True,
        },
        "outputs": {
            "final_model": {"filename": "final_model", "format": "pkl", "generate": True},
            "round_models": {"filename_prefix": "it", "format": "pkl", "generate": True},
            "round_params": {"filename_prefix": "it", "format": "json", "generate": True},
            "best_params_summary": {"filename": "base_params", "format": "json", "generate": True},
        },
    },
    "step4_evaluation": {
        "enabled": True,
        "functions": {
            "model_score_eval": True,
            "monthly_ks_auc": True,
            "feature_importance": True,
            "monthly_bin_bad_rate": True,
            "overfitting_check": True,
        },
        "outputs": {
            "model_ks_auc_by_month": {"filename": "model_ks_auc_by_month", "format": "xlsx", "generate": True},
            "feature_importance": {"filename": "feature_importance", "format": "csv", "generate": True},
            "monthly_bin_bad_rate": {"filename": "monthly_bin_bad_rate", "format": "xlsx", "generate": True},
        },
    },
    "step5_report": {
        "enabled": True,
        "functions": {
            "aggregate_all_tasks": True,
            "model_comparison": True,
            "overall_summary": True,
        },
        "outputs": {
            "final_consolidated_report": {"filename": "final_consolidated_report", "format": "xlsx", "generate": True},
            "task_comparison": {"filename": "task_comparison", "format": "xlsx", "generate": True},
        },
    },
    "general": {
        "verbose": True,
        "save_training_history": True,
        "overwrite_existing": False,
    },
}

# ------------------------------------------------------------------
# Global cache to avoid repeated file reads
# ------------------------------------------------------------------
_config_cache: Optional[Dict[str, Any]] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    """YAML loader with fallback."""
    try:
        import yaml
    except ImportError:
        return {}

    if not path.exists():
        print(f"[outputs] Warning: {path} not found, using defaults.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_output_config(project_dir: Optional[Path] = None, skill_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load result_thresholds.yaml and merge with defaults.

    Priority: YAML file overrides defaults.
    Returns a complete config dict (never None).
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    # Try multiple paths — priority: project_dir > skill_dir
    candidates = []
    if project_dir:
        candidates.append(project_dir / "config" / "result_thresholds.yaml")
    if skill_dir:
        candidates.append(skill_dir / "config" / "result_thresholds.yaml")
    candidates.append(Path(__file__).resolve().parent.parent / "config" / "result_thresholds.yaml")

    yml_config = {}
    for cp in candidates:
        loaded = _load_yaml(cp)
        if loaded:
            yml_config = loaded
            break

    # Deep merge: start from defaults, overlay YAML values
    merged = _deep_merge(_DEFAULT_CONFIG.copy(), yml_config)
    _config_cache = merged
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place on base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def get_step_config(step_key: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the sub-config for a specific step (e.g. 'step4_evaluation')."""
    if config is None:
        config = load_output_config()
    return config.get(step_key, {})


def should_generate(step_key: str, output_key: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Check whether a specific output should be generated.

    Returns True if the output's 'generate' flag is set (default True).
    """
    step_cfg = get_step_config(step_key, config)
    outputs = step_cfg.get("outputs", {})
    out_cfg = outputs.get(output_key, {})
    return out_cfg.get("generate", True)


def is_function_enabled(step_key: str, func_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Check whether a specific sub-function should be executed.

    Priority (从高到低):
      1. step.optional_analysis.<func_name>  (YAML显式配置)
      2. step.functions.<func_name>         (代码默认)
      3. default True (backward compat)

    设计意图：optional_analysis 是用户在 YAML 里手动写的开关，
    优先级应该高于 functions 里的硬编码默认值。
    """
    step_cfg = get_step_config(step_key, config)

    # 优先读取 optional_analysis（用户YAML显式配置）
    opt = step_cfg.get("optional_analysis", {})
    if isinstance(opt, dict) and func_name in opt:
        return bool(opt[func_name])

    # 也支持 alternate key names (compute_correlation / correlation_check)
    alt_names = {
        "correlation_check": ["compute_correlation", "correlation_check"],
        "compute_correlation": ["compute_correlation", "correlation_check"],
    }
    for alt in alt_names.get(func_name, []):
        if alt != func_name and alt in opt:
            return bool(opt[alt])

    # Fall back to functions (默认值)
    funcs = step_cfg.get("functions", {})
    if func_name in funcs:
        return bool(funcs[func_name])

    return True


def resolve_filename(
    step_key: str,
    output_key: str,
    task_id: Optional[str] = None,
    project_dir: Optional[Path] = None,
    subdir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Resolve the full file path for a configured output.

    E.g. resolve_filename('step4_evaluation', 'model_ks_auc_by_month', task_id='source_a_y_good')
         → project_dir/reports/source_a_y_good/model_ks_auc_by_month.xlsx
    """
    if config is None:
        config = load_output_config()

    step_cfg = get_step_config(step_key, config)
    outputs = step_cfg.get("outputs", {})
    out_cfg = outputs.get(output_key, {})

    # Filename base — support both 'filename' and 'filename_prefix'
    if "filename" in out_cfg:
        name = out_cfg["filename"]
    elif "filename_prefix" in out_cfg:
        name = out_cfg["filename_prefix"]
    else:
        name = output_key

    fmt = out_cfg.get("format", "csv")

    # Append task_id suffix if provided
    if task_id:
        safe_tid = task_id.replace("/", "_").replace(" ", "_")
        name = f"{name}_{safe_tid}"

    filename = f"{name}.{fmt}"

    # Resolve directory
    if project_dir:
        if subdir:
            base = Path(project_dir) / subdir
        else:
            base = project_dir
    else:
        base = Path(__file__).resolve().parent.parent / "output"

    return base / filename


def get_pipeline_toggle(step_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Get the pipeline-level on/off for a step."""
    if config is None:
        config = load_output_config()
    pipeline_cfg = config.get("pipeline", {})
    return pipeline_cfg.get(step_name, True)


def reset_cache():
    """Clear config cache (useful for testing or when file changes)."""
    global _config_cache
    _config_cache = None
