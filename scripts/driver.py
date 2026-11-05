"""
风控建模统一驾驶入口 (Driver)

职责:
  - manifest.json 契约驱动编排5步流水线
  - 支持全流程 / 单步执行 / 断点续跑
  - 自动检查前置依赖和输入文件
  - 多标签 / 多数据源建模（get_modeling_tasks）
  - 铁律一: Step3/Step4 用 wide_table_full.parquet (全量)
  - 铁律二: Step2 用 wide_table_sampled.parquet (采样)

用法:
    python driver.py \\
        --project-dir C:/Users/Admin/Desktop/hp_20260513 \\
        --raw-data C:/Users/Admin/Desktop/hp_20260513/data/raw/data.parquet \\
        --label-col p3m2 \\
        --month-col app_date \\
        --steps all

    python driver.py --project-dir ./hp_20260513 --step 3
    python driver.py --project-dir ./hp_20260513 --resume
"""

from __future__ import print_function, unicode_literals

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- 自动切换至 pypc 环境 (pyarrow 必需) ----
_pypc_exe = r"C:\Users\Admin\.conda\envs\pypc\python.exe"
if __name__ == "__main__":
    _have_pyarrow = False
    try:
        import pyarrow  # noqa: F401
        _have_pyarrow = True
    except ImportError:
        pass
    if not _have_pyarrow and Path(_pypc_exe).exists():
        print("[Env] Switching to pypc: %s" % _pypc_exe)
        os.execl(_pypc_exe, _pypc_exe, __file__, *sys.argv[1:])

SCRIPT_DIR = Path(__file__).resolve().parent

# ---- 确保 pipeline_config 可导入 ----
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import (
    init_manifest,
    load_manifest,
    save_manifest,
    mark_step_done,
    mark_step_failed,
    get_step_status,
    check_step_ready,
    get_next_step,
    get_pending_steps,
    STEP_ORDER,
    load_data_config,
    _load_yaml_file,
    DEFAULT_DATA_CONFIG,
    infer_feature_cols,
    get_modeling_tasks,
    smart_read_df,
    collect_gc,
    smart_release,
    ensure_dirs,
    make_safe_task_id,
)


# ======================================================================
# Helper utilities
# ======================================================================


def _import_step_module(step_key):
    """
    动态导入步骤模块并返回其 main_entry 函数。

    step1_eda          -> step1_eda.main_step1
    step2_feature_select -> step2_feature_select.main_step2
    step3_modeling     -> step3_modeling.main_step3
    step4_eval         -> step4_eval.main_step4
    step5_report       -> step5_report.main_step5
    """
    module_map = {
        "step1_eda":          ("step1_eda.main_step1", ["build_wide_table", "run_eda"]),
        "step2_feature_select": ("step2_feature_select.main_step2", ["run_feature_selection"]),
        "step3_modeling":   ("step3_modeling.main_step3", ["run_modeling"]),
        "step4_eval":       ("step4_eval.main_step4", ["run_evaluation"]),
        "step5_report":     ("step5_report.main_step5", ["run_consolidated_report"]),
    }

    if step_key not in module_map:
        raise KeyError("Unknown step key: %s" % step_key)

    mod_path, names = module_map[step_key]
    import importlib
    mod = importlib.import_module(mod_path)
    return {n: getattr(mod, n) for n in names}


def _resolve_month_col(cli_month, manifest, project_dir):
    """
    Resolve month_col 优先级:
      1. CLI arg
      2. project_dir/config/data_config.yaml  -> analysis_dims.month_col
      3. manifest parameters.month_col
      4. fallback 'app_date'
    """
    if cli_month:
        return cli_month

    # Priority 2: project dir data_config.yaml
    try:
        proj_cfg = _load_yaml_file(project_dir / "config" / "data_config.yaml", {})
        dims = proj_cfg.get("analysis_dims") or {}
        v = dims.get("month_col", "")
        if v:
            return v
    except Exception:
        pass

    # Priority 3: manifest parameters.month_col
    params = manifest.get("parameters") or {}
    v = params.get("month_col", "")
    if v:
        return v

    # Default
    return "app_date"


def _infer_months(df, month_col):
    """Return sorted unique months from DataFrame."""
    if month_col not in df.columns:
        return []
    return sorted(df[month_col].dropna().unique().tolist())


def _split_month_list(months, oot_tail=4):
    """Split months into train (all but tail) and OOT (tail)."""
    if len(months) <= oot_tail:
        return list(months), []
    return list(months[:-oot_tail]), list(months[-oot_tail:])


# ======================================================================
# Step 1: Deep EDA + Dual Wide Table
# ======================================================================


def execute_step1(
    project_dir,
    manifest,
    raw_data_path,
    feature_cols,
    extra_score_cols,
    tasks,
):
    """
    Step 1: 数据探索、构建双宽表 (full + sampled)、EDA分析

    铁律二: build_wide_table 生成 wide_table_full.parquet 和
            wide_table_sampled.parquet

    multi-label: 对每个 modeling task 跑一次 EDA，但宽表只建一次。
    """
    print("\n" + "=" * 70)
    print("  >> Step 1: Deep EDA & Dual Wide Table Construction")
    print("=" * 70)

    step_mod = _import_step_module("step1_eda")
    build_wide_table = step_mod["build_wide_table"]
    run_eda = step_mod["run_eda"]

    base_dir = SCRIPT_DIR.parent

    try:
        data_config = load_data_config(project_dir=project_dir)
    except Exception:
        try:
            data_config = load_data_config(base_dir=base_dir)
        except Exception as e:
            print("[Step1] data_config load failed: %s" % e)
            data_config = dict(DEFAULT_DATA_CONFIG)

    month_col = _resolve_month_col(
        manifest.get("parameters", {}).get("month_col"),
        manifest, project_dir,
    )
    manifest.setdefault("parameters", {})["month_col"] = month_col

    dims_cfg = data_config.get("analysis_dims") or {}
    group_by_cols = dims_cfg.get("group_by_cols", [])
    external_scores = dims_cfg.get("external_scores", [])

    # 合并 extra_score_cols: CLI传入 + data_config.external_scores + group_by_cols
    all_extra = list(extra_score_cols) if extra_score_cols else []
    for c in external_scores:
        if c not in all_extra:
            all_extra.append(c)
    for c in group_by_cols:
        if c not in all_extra:
            all_extra.append(c)

    if not tasks:
        label_col = manifest.get("parameters", {}).get("label_col", "y")
        tasks = [{
            "source_key": "default",
            "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
            "label_key": label_col,
            "label_col": label_col,
            "x_cols": [],
            "task_id": "default_" + label_col,
        }]

    # ---- 合并所有任务的 x_cols，构建全局 feature_cols ----
    all_x = set(feature_cols) if feature_cols else set()
    for task in tasks:
        tx = task.get("x_cols") or []
        all_x.update(tx)

    global_x_cfg = data_config.get("x_cols", [])
    if global_x_cfg:
        all_x.update(global_x_cfg)

    if not all_x:
        # 自动推断: 读取原始数据数值列
        try:
            df_sample = smart_read_df(raw_data_path, dtype=str, nrows=1000)
            inferred = infer_feature_cols(df_sample, data_config)
            all_x.update(inferred)
            del df_sample
        except Exception as e:
            print("[Step1] Cannot infer feature cols: %s" % e)

    final_feature_cols = sorted(all_x)
    print("[Step1] Feature candidates: %d" % len(final_feature_cols))
    print("[Step1] Modeling tasks from config:")
    for task in tasks:
        print(f"  - {task['task_id']}: source={task['source_key']}, label={task['label_key']}, x_cols={len(task.get('x_cols',[]))}")

    # ---- 构建双宽表 (只执行一次，包含所有 label 列 + month_col + extra) ----
    processed = project_dir / "data" / "processed"
    full_path = processed / "wide_table_full.parquet"
    sampled_path = processed / "wide_table_sampled.parquet"

    if full_path.exists() and sampled_path.exists():
        print(f"[Step1] wide_table_full.parquet + wide_table_sampled.parquet already exist, skipping build.")
    else:
        try:
            wide_table_path = build_wide_table(
                raw_data_path=raw_data_path,
                feature_cols=final_feature_cols,
                label_col=tasks[0].get("label_col", "y"),
                month_col=month_col,
                extra_cols=all_extra if all_extra else None,
                project_dir=project_dir,
            )
        except Exception as e:
            import traceback
            print("[Step1] build_wide_table failed: %s" % e)
            traceback.print_exc()
            mark_step_failed(manifest, "step1_eda", str(e), project_dir)
            return False

        if not wide_table_path:
            mark_step_failed(
                manifest, "step1_eda",
                "wide_table build returned empty path", project_dir,
            )
            return False

    # ---- 每个 task 执行 EDA (全量数据传入 run_eda) ----
    if not full_path.exists():
        print("[Step1] wide_table_full.parquet NOT found after build!")
        mark_step_failed(manifest, "step1_eda", "wide_table_full missing", project_dir)
        return False

    df_wide = pd.read_parquet(str(full_path))
    all_artifacts = {}

    for task in tasks:
        task_id = task.get("task_id", "")
        label_col = task.get(
            "label_col",
            manifest.get("parameters", {}).get("label_col", "y"),
        )
        src_x_cols = task.get("x_cols") or final_feature_cols

        try:
            # df_base 不需要 copy，run_eda 内部会按需处理
            _, artifacts = run_eda(
                df=df_wide,
                df_base=df_wide,
                label_col=label_col,
                feature_cols=src_x_cols,
                month_col=month_col,
                project_dir=project_dir,
                base_dir=base_dir,
                raw_data_path=raw_data_path,
                extra_cols=all_extra if all_extra else None,
                task_id=task_id,
                group_by_cols=group_by_cols if group_by_cols else None,
            )
            all_artifacts[task_id] = artifacts
        except Exception as e:
            import traceback
            print("[Step1] EDA for task %s failed: %s" % (task_id, e))
            traceback.print_exc()

        collect_gc()

    # 释放宽表内存
    from pipeline_config import smart_release
    smart_release(df_wide, label="Step1 wide_table_full released")

    mark_step_done(manifest, "step1_eda", project_dir)

    print("\n[Step1] Artifacts:")
    for k, v in all_artifacts.items():
        print("  Task %s:" % k)
        for ak, av in v.items():
            print("    [%s] -> %s" % (ak, av))

    return True


# ======================================================================
# Step 2: Feature Selection (投票筛选)
# ======================================================================


def execute_step2(project_dir, manifest, tasks):
    """
    Step 2: 多维度特征筛选

    铁律二: IV/PSI 用 sampled 数据 (wide_table_sampled.parquet)
    """
    print("\n" + "=" * 70)
    print("  >> Step 2: Feature Selection")
    print("=" * 70)

    step_mod = _import_step_module("step2_feature_select")
    run_feature_selection = step_mod["run_feature_selection"]

    base_dir = SCRIPT_DIR.parent

    try:
        data_config = load_data_config(project_dir=project_dir)
    except Exception:
        try:
            data_config = load_data_config(base_dir=base_dir)
        except Exception as e:
            print("[Step2] data_config load failed: %s" % e)
            data_config = dict(DEFAULT_DATA_CONFIG)

    month_col = _resolve_month_col(
        manifest.get("parameters", {}).get("month_col"),
        manifest, project_dir,
    )

    if not tasks:
        label_col = manifest.get("parameters", {}).get("label_col", "y")
        tasks = [{
            "source_key": "default",
            "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
            "label_key": label_col,
            "label_col": label_col,
            "x_cols": [],
            "task_id": "default_" + label_col,
        }]

    processed = project_dir / "data" / "processed"

    # 铁律二: Step2 读 sampled
    sampled_path = processed / "wide_table_sampled.parquet"
    if not sampled_path.exists():
        full_path = processed / "wide_table_full.parquet"
        if full_path.exists():
            print("[Step2] wide_table_sampled.parquet missing, falling back to full.")
            sampled_path = full_path
        else:
            print("[Step2] No wide table found at all!")
            mark_step_failed(manifest, "step2_feature_select", "No wide table", project_dir)
            return False

    df_sampled = pd.read_parquet(str(sampled_path))
    print("[Step2] Sampled data: %d rows x %d cols" % (len(df_sampled), len(df_sampled.columns)))

    for task in tasks:
        task_id = task.get("task_id", "")
        label_col = task.get(
            "label_col",
            manifest.get("parameters", {}).get("label_col", "y"),
        )
        src_x_cols = task.get("x_cols") or []

        # 裁剪到任务需要的特征列（只保留数值类型，排除 category/object）
        if src_x_cols:
            available_x = [c for c in src_x_cols if c in df_sampled.columns
                           and df_sampled[c].dtype.kind in ('i', 'f', 'u')]
        else:
            available_x = infer_feature_cols(df_sampled, data_config)

        if not available_x:
            print("[Step2] No features for task %s, skipping." % task_id)
            continue

        try:
            selected_feats, artifacts = run_feature_selection(
                df=df_sampled,
                df_base=df_sampled,  # 不 copy，直接传引用
                label_col=label_col,
                feature_cols=available_x,
                month_col=month_col,
                project_dir=project_dir,
                base_dir=base_dir,
                task_id=task_id,
            )
            print("[Step2] Task %s: selected %d features." % (task_id, len(selected_feats)))
        except Exception as e:
            import traceback
            print("[Step2] Feature selection for task %s failed: %s" % (task_id, e))
            traceback.print_exc()

    smart_release(df_sampled, label="Step2 wide_table_sampled released")

    mark_step_done(manifest, "step2_feature_select", project_dir)
    return True


# ======================================================================
# Step 3: Bayesian Optimization Modeling
# ======================================================================


def execute_step3(project_dir, manifest, n_runs, tasks):
    """
    Step 3: 贝叶斯优化建模

    铁律一: 建模用全量数据 (wide_table_full.parquet)
    每个 label 先过滤非空行 → 按月份拆分 train/valid/OOT → 各自独立数据集
    """
    print("\n" + "=" * 70)
    print("  >> Step 3: Bayesian Optimization Modeling")
    print("=" * 70)

    step_mod = _import_step_module("step3_modeling")
    run_modeling = step_mod["run_modeling"]

    base_dir = SCRIPT_DIR.parent

    try:
        data_config = load_data_config(project_dir=project_dir)
    except Exception:
        try:
            data_config = load_data_config(base_dir=base_dir)
        except Exception as e:
            print("[Step3] data_config load failed: %s" % e)
            data_config = dict(DEFAULT_DATA_CONFIG)

    month_col = _resolve_month_col(
        manifest.get("parameters", {}).get("month_col"),
        manifest, project_dir,
    )

    if not tasks:
        label_col = manifest.get("parameters", {}).get("label_col", "y")
        tasks = [{
            "source_key": "default",
            "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
            "label_key": label_col,
            "label_col": label_col,
            "x_cols": [],
            "task_id": "default_" + label_col,
        }]

    processed = project_dir / "data" / "processed"

    # Step3: 建模只读需要的列，减少内存占用
    full_path = processed / "wide_table_full.parquet"
    if not full_path.exists():
        print("[Step3] wide_table_full.parquet NOT found!")
        mark_step_failed(manifest, "step3_modeling", "wide_table_full missing", project_dir)
        return False

    # ---- 收集所有 label + 特征列，一次性拆分数据集 ----
    all_labels = []
    task_label_map = {}

    for task in tasks:
        task_id = task.get("task_id", "")
        label_col = task.get(
            "label_col",
            manifest.get("parameters", {}).get("label_col", "y"),
        )
        all_labels.append(label_col)
        task_label_map[task_id] = label_col

    # ---- 调用 data_split 按 label 拆分数据集 ----
    from data_split import split_wide_table_and_save

    # 获取所有选中特征并集
    all_features_set = set()
    for task in tasks:
        feat_path = processed / ("selected_features_" + task.get("task_id", "") + ".csv")
        if not feat_path.exists():
            feat_path = processed / "selected_features.csv"
        if feat_path.exists():
            try:
                feat_df = pd.read_csv(str(feat_path))
                all_features_set.update(feat_df.iloc[:, 0].tolist() if len(feat_df.columns) == 1 else feat_df.columns)
            except Exception:
                pass

    feature_cols_split = list(all_features_set)
    label_cols_split = list(set(all_labels))

    print("\n[Step3] Splitting datasets by label...")
    split_cfg = {}
    try:
        from pipeline_config import load_model_params as _load_model_params
        # Priority: project_dir config > skill base_dir config
        model_params = _load_model_params(project_dir=project_dir, skill_dir=base_dir)
        if not model_params or 'data_split' not in model_params:
            model_params = _load_model_params(skill_dir=base_dir)
        split_cfg = model_params.get("data_split", {})
    except Exception:
        pass

    oot_tail = split_cfg.get("oot_months_tail", 4)
    val_ratio = split_cfg.get("val_ratio", 0.3)

    dataset_dir = processed / "datasets"
    split_wide_table_and_save(
        wide_table_path=str(full_path),
        month_col=month_col,
        label_cols=label_cols_split,
        feature_cols=feature_cols_split,
        output_dir=str(dataset_dir),
        oot_tail_months=oot_tail,
        val_ratio=val_ratio,
        random_state=42,
    )
    print("[Step3] Dataset split complete.")

    # ---- 每个 task 加载对应的数据集进行建模 ----
    for task in tasks:
        task_id = task.get("task_id", "")
        label_col = task_label_map.get(task_id)
        if not label_col:
            continue

        feat_path = processed / ("selected_features_" + task_id + ".csv")
        if not feat_path.exists():
            feat_path = processed / "selected_features.csv"
        if not feat_path.exists():
            print("[Step3] selected_features not found for task %s, skipping." % task_id)
            continue

        try:
            feat_df = pd.read_csv(str(feat_path))
            feature_cols = (
                feat_df.iloc[:, 0].tolist()
                if len(feat_df.columns) == 1
                else list(feat_df.columns)
            )
        except Exception as e:
            print("[Step3] Cannot read selected_features for %s: %s" % (task_id, e))
            continue

        import pyarrow.parquet as pq
        all_parquet_cols = pq.read_schema(str(full_path)).names
        available_feats = [c for c in feature_cols if c in all_parquet_cols]
        if not available_feats:
            print("[Step3] No matching features in data for task %s." % task_id)
            continue

        # 空 DataFrame，run_modeling 会走 load_split_dataset 路径加载对应 label 的数据集
        df_task = pd.DataFrame()
        print(f"[Step3] Task {task_id}: will load from datasets/ for label={label_col}")

        # Skip if final model already exists with correct feature count
        import joblib as _joblib
        _label_short = label_col  # e.g. p4m2
        _models_dir = project_dir / "models"
        _final_pkl = _models_dir / ("final_model_%s_%s.pkl" % (_label_short, task_id))
        if _final_pkl.exists():
            try:
                _existing_model = _joblib.load(str(_final_pkl))
                _existing_nf = _existing_model.booster_.num_feature()
                if _existing_nf == len(available_feats):
                    print(f"[Step3] Skipping {task_id}: final model exists with {_existing_nf} features (matches).")
                    del _existing_model
                    gc.collect()
                    continue
            except Exception:
                pass  # Fall through to retrain

        try:
            result = run_modeling(
                df=df_task,
                label_col=label_col,
                feature_cols=available_feats,
                month_col=month_col,
                project_dir=project_dir,
                base_dir=base_dir,
                n_runs=n_runs,
                task_id=task_id,
            )
            if result:
                print("[Step3] Task %s modeling complete." % task_id)
        except Exception as e:
            import traceback
            print("[Step3] Modeling for task %s failed: %s" % (task_id, e))
            traceback.print_exc()

        # ============================================================
        # AGGRESSIVE MEMORY CLEANUP after each label
        # ============================================================
        del df_task
        del feat_df
        print("\n[Memory] Aggressive cleanup after label %s..." % task_id)

        # Clear lightgbm internal caches
        try:
            import lightgbm as _lgbm
            del _lgbm
        except Exception:
            pass

        # Destroy imported step module to release LGBMClassifier refs
        if "step3_modeling.main_step3" in sys.modules:
            mod_ref = sys.modules["step3_modeling.main_step3"]
            if hasattr(mod_ref, "bayesian_optimize"):
                bo = mod_ref.bayesian_optimize
                # Reset globals inside bayesian_optimize
                for attr in ["X_train_global", "y_train_global", "X_valid_global",
                            "y_valid_global", "X_oot_global", "y_oot_global",
                            "feature_names_global"]:
                    if hasattr(bo, attr):
                        setattr(bo, attr, None)

        for _ in range(5):
            gc.collect()

        import os
        proc_mem_kb = 0
        try:
            psutil = __import__("psutil")
            proc_mem_kb = psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024
        except Exception:
            pass
        print("[Memory] Cleanup complete. RSS ~%d MB\n" % proc_mem_kb)
        # ============================================================

    mark_step_done(manifest, "step3_modeling", project_dir)
    return True


# ======================================================================
# Step 4: Strict Model Evaluation
# ======================================================================


def execute_step4(project_dir, manifest, tasks):
    """
    Step 4: 严苛模型评估 (KS/AUC/PSI)

    铁律一: 评估用全量数据 (wide_table_full.parquet)
    """
    print("\n" + "=" * 70)
    print("  >> Step 4: Strict Model Evaluation")
    print("=" * 70)

    step_mod = _import_step_module("step4_eval")
    run_evaluation = step_mod["run_evaluation"]

    # Prefer project dir config over skill dir
    _cfg_dir = project_dir
    try:
        data_config = load_data_config(project_dir=project_dir)
    except Exception:
        try:
            _cfg_dir = SCRIPT_DIR.parent
            data_config = load_data_config(base_dir=_cfg_dir)
        except Exception as e:
            print("[Step4] data_config load failed: %s" % e)
            _cfg_dir = SCRIPT_DIR.parent
            data_config = dict(DEFAULT_DATA_CONFIG)

    month_col = _resolve_month_col(
        manifest.get("parameters", {}).get("month_col"),
        manifest, project_dir,
    )

    # 读取 external_scores 和 group_by_cols (for Step4 score evaluation)
    dims_cfg = data_config.get("analysis_dims", {})
    raw_ext = dims_cfg.get("external_scores", [])
    # 支持两种格式：YAML list 或字符串形式的 ['a','b',...]
    if isinstance(raw_ext, str):
        try:
            import ast
            external_scores_list = ast.literal_eval(raw_ext)
        except Exception:
            external_scores_list = [raw_ext.strip("[]' \"")]
    else:
        external_scores_list = list(raw_ext)

    group_cols_list = list(dims_cfg.get("group_by_cols", []))

    if not tasks:
        label_col = manifest.get("parameters", {}).get("label_col", "y")
        tasks = [{
            "source_key": "default",
            "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
            "label_key": label_col,
            "label_col": label_col,
            "x_cols": [],
            "task_id": "default_" + label_col,
        }]

    processed = project_dir / "data" / "processed"
    models_dir = project_dir / "models"

    # Step4: 评估只读需要的列（model_score + external_scores + label + month + group_cols）
    full_path = processed / "wide_table_full.parquet"
    if not full_path.exists():
        print("[Step4] wide_table_full.parquet NOT found!")
        mark_step_failed(manifest, "step4_eval", "wide_table_full missing", project_dir)
        return False

    # Get column names from parquet schema (cheap, no data load)
    import pyarrow.parquet as pq
    all_parquet_cols = pq.read_schema(str(full_path)).names

    # ---- 收集所有任务需要的列并集 + 元信息，一次性读取宽表 ----
    all_eval_cols_set = set([month_col])
    task_info = []

    for task in tasks:
        task_id = task.get("task_id", "")
        label_col = task.get(
            "label_col",
            manifest.get("parameters", {}).get("label_col", "y"),
        )
        safe_label = label_col.replace("/", "_").replace(" ", "_")
        safe_task_id = make_safe_task_id(task_id)

        # 定位模型文件（支持多种命名）
        model_path_candidates = [
            models_dir / ("final_model_" + safe_label + "_" + safe_task_id + ".pkl"),
            models_dir / ("final_model_" + safe_label + safe_task_id + ".pkl"),
            models_dir / ("final_model_" + safe_label + ".pkl"),
            models_dir / ("best_lgbm_" + safe_label + ".pkl"),
            models_dir / ("it_1_" + safe_label + "_" + safe_task_id + "_model.pkl"),
            models_dir / ("it_1_" + safe_label + safe_task_id + "_model.pkl"),
            models_dir / ("final_model_" + safe_task_id + ".pkl"),
        ]

        model_path = None
        for cand in model_path_candidates:
            if cand.exists():
                model_path = cand
                break

        # fallback: scan models dir for .pkl files containing task_id or label
        if model_path is None:
            pkl_files = list(models_dir.glob("*.pkl"))
            for pf in pkl_files:
                if task_id in str(pf.name) or safe_label in str(pf.name):
                    model_path = pf
                    print(f"[Step4] Found model via scan: {pf.name}")
                    break

        if model_path is None:
            print("[Step4] Model not found for task %s, skipping." % task_id)
            continue

        # 读取 selected_features
        feat_path = processed / ("selected_features_" + task_id + ".csv")
        if not feat_path.exists():
            feat_path = processed / "selected_features.csv"

        feature_cols = []
        if feat_path.exists():
            try:
                feat_df = pd.read_csv(str(feat_path))
                if not feat_df.empty:
                    feature_cols = (
                        feat_df.iloc[:, 0].tolist()
                        if len(feat_df.columns) == 1
                        else list(feat_df.columns)
                    )
            except Exception:
                pass

        # 收集该任务需要的列（month_col 必须包含，否则 step4 按月 KS/AUC 会被跳过）
        task_cols = set(feature_cols) | {label_col, month_col} | set(external_scores_list) | set(group_cols_list)
        all_eval_cols_set.update(task_cols)

        task_info.append({
            "task_id": task_id,
            "label_col": label_col,
            "model_path": str(model_path),
            "feature_cols": feature_cols,
            "cols": list(task_cols),
        })

    if not task_info:
        print("[Step4] No valid tasks to evaluate.")
        mark_step_done(manifest, "step4_eval", project_dir)
        return True

    # 一次性读取（只读真正需要的列并集）
    all_eval_cols = [c for c in all_eval_cols_set if c in all_parquet_cols]
    df_shared = pd.read_parquet(str(full_path), columns=all_eval_cols)
    print(f"[Step4] Shared load: {len(df_shared)} rows x {len(df_shared.columns)} cols")

    # 遍历任务，每个任务从共享数据裁剪副本
    for info in task_info:
        my_cols = [c for c in info["cols"] if c in df_shared.columns]
        df_task = df_shared[my_cols].copy()
        print(f"[Step4] Task {info['task_id']}: using {len(df_task)} rows x {len(df_task.columns)} cols")

        try:
            run_evaluation(
                df=df_task,
                model_path=info["model_path"],
                label_col=info["label_col"],
                feature_cols=info["feature_cols"],
                month_col=month_col,
                project_dir=project_dir,
                base_dir=_cfg_dir,
                task_id=info["task_id"],
                external_scores=external_scores_list,
                group_cols=group_cols_list,
            )
            print("[Step4] Task %s evaluation complete." % info["task_id"])
        except Exception as e:
            import traceback
            print("[Step4] Evaluation for task %s failed: %s" % (info["task_id"], e))
            traceback.print_exc()

        del df_task
        collect_gc()

    # 释放共享宽表
    from pipeline_config import smart_release
    smart_release(df_shared, label="Step4 wide_table_full released")

    mark_step_done(manifest, "step4_eval", project_dir)
    return True


# ======================================================================
# Step 5: Final Report
# ======================================================================


def execute_step5(project_dir, manifest, tasks):
    """Step 5: 结案报告 (Final Report Generation)."""
    print("\n" + "=" * 70)
    print("  >> Step 5: Final Report Generation")
    print("=" * 70)

    step_mod = _import_step_module("step5_report")
    run_consolidated_report_fn = step_mod["run_consolidated_report"]

    # Prefer project dir config over skill dir
    _cfg_dir = SCRIPT_DIR.parent
    try:
        data_config = load_data_config(project_dir=project_dir)
    except Exception:
        try:
            data_config = load_data_config(base_dir=_cfg_dir)
        except Exception as e:
            print("[Step5] data_config load failed: %s" % e)
            data_config = dict(DEFAULT_DATA_CONFIG)

    if not tasks:
        label_col = manifest.get("parameters", {}).get("label_col", "y")
        tasks = [{
            "source_key": "default",
            "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
            "label_key": label_col,
            "label_col": label_col,
            "x_cols": [],
            "task_id": "default_" + label_col,
        }]

    # 多任务聚合报告 (run_consolidated_report loads data_config internally)
    try:
        run_consolidated_report_fn(
            project_dir=project_dir,
            base_dir=_cfg_dir,
        )
    except Exception as e:
        import traceback
        print("[Step5] Consolidated report failed (non-fatal): %s" % e)
        traceback.print_exc()
        mark_step_failed(manifest, "step5_report", str(e), project_dir)
        return False

    mark_step_done(manifest, "step5_report", project_dir)
    return True


# ======================================================================
# Step Scheduler
# ======================================================================

_STEP_TO_EXECUTOR = {
    "step1_eda":          lambda m, p: execute_step1(p, m, _get_raw(m), _get_feature_cols(m, p), _get_extra_score_cols(m, p), _build_tasks(m, p)),
    "step2_feature_select": lambda m, p: execute_step2(p, m, _build_tasks(m, p)),
    "step3_modeling":   lambda m, p: execute_step3(p, m, 5, _build_tasks(m, p)),
    "step4_eval":       lambda m, p: execute_step4(p, m, _build_tasks(m, p)),
    "step5_report":     lambda m, p: execute_step5(p, m, _build_tasks(m, p)),
}


def _get_raw(manifest):
    return manifest.get("parameters", {}).get("raw_data_path", "")


def _get_feature_cols(manifest, project_dir):
    """Get feature cols from manifest or data_config — prefer project dir config."""
    x_cols = manifest.get("parameters", {}).get("x_cols", [])
    if x_cols:
        return list(x_cols)

    # Priority 1: project dir's own data_config.yaml
    try:
        dc = load_data_config(project_dir=Path(project_dir))
        if dc.get("x_cols"):
            return list(dc["x_cols"])
        sources = dc.get("data_sources", {})
        for src_key, src_cfg in sources.items():
            if src_cfg.get("x_cols"):
                return list(src_cfg["x_cols"])
    except Exception:
        pass

    # Priority 2: skill dir default config
    try:
        base_dir = SCRIPT_DIR.parent
        dc = load_data_config(base_dir=base_dir)
        if dc.get("x_cols"):
            return list(dc["x_cols"])
    except Exception:
        pass

    return []


def _get_extra_score_cols(manifest, project_dir=None):
    """Get extra score cols from manifest or data_config — prefer project dir config."""
    extra = manifest.get("parameters", {}).get("extra_score_cols", [])
    if extra:
        return list(extra)

    # Priority 1: project dir's own data_config.yaml
    if project_dir is not None:
        try:
            dc = load_data_config(project_dir=Path(project_dir))
            dims = dc.get("analysis_dims", {})
            ext = dims.get("external_scores", [])
            if ext:
                return list(ext)
        except Exception:
            pass
    # Priority 2: skill dir default config
    try:
        base_dir = SCRIPT_DIR.parent
        dc = load_data_config(base_dir=base_dir)
        dims = dc.get("analysis_dims", {})
        ext = dims.get("external_scores", [])
        if ext:
            return list(ext)
    except Exception:
        pass

    return []


def _build_tasks(manifest, project_dir=None):
    """Build modeling tasks from data_config — prefer project dir config over skill dir."""
    # Priority 1: project dir's own data_config.yaml (user overrides)
    if project_dir is not None:
        try:
            dc = load_data_config(project_dir=Path(project_dir))
            tasks = get_modeling_tasks(dc)
            if tasks:
                return tasks
        except Exception as e:
            print(f"[Driver] Loading tasks from project config failed: {e}")

    # Priority 2: skill dir default config
    try:
        base_dir = SCRIPT_DIR.parent
        dc = load_data_config(base_dir=base_dir)
        tasks = get_modeling_tasks(dc)
        if tasks:
            return tasks
    except Exception:
        pass

    label_col = manifest.get("parameters", {}).get("label_col", "y")
    return [{
        "source_key": "default",
        "source_name": "\u9ed8\u8ba4\u6570\u636e\u6e90",
        "label_key": label_col,
        "label_col": label_col,
        "x_cols": [],
        "task_id": "default_" + label_col,
    }]


def run_single_step(step_key, manifest, project_dir):
    """Run a single step by key name."""
    executor = _STEP_TO_EXECUTOR.get(step_key)
    if executor is None:
        print("[Driver] Unknown step key: %s" % step_key)
        return False

    ready, reason = check_step_ready(manifest, step_key, project_dir)
    if not ready:
        print("[Driver] Step %s not ready: %s" % (step_key, reason))
        return False

    status = get_step_status(manifest, step_key)
    if status == "done":
        print("[Driver] Step %s already done, skipping." % step_key)
        return True

    try:
        result = executor(manifest, project_dir)
        return bool(result)
    except Exception as e:
        import traceback
        print("[Driver] Step %s failed: %s" % (step_key, e))
        traceback.print_exc()
        mark_step_failed(manifest, step_key, str(e), project_dir)
        return False


def run_single_step_by_number(step_num, manifest, project_dir):
    """Run a single step by number (1-5)."""
    for step_key, (num, name) in STEP_ORDER.items():
        if num == step_num:
            return run_single_step(step_key, manifest, project_dir)
    print("[Driver] Invalid step number: %d (valid: 1-5)" % step_num)
    return False


def run_all_steps(manifest, project_dir, steps_filter=None):
    """Execute all or specified steps in order."""
    all_ok = True

    if steps_filter:
        for step_key in steps_filter:
            if step_key not in STEP_ORDER:
                print("[Driver] Unknown step key: %s, skipping." % step_key)
                continue
            ok = run_single_step(step_key, manifest, project_dir)
            if not ok:
                all_ok = False
                print("[Driver] Step %s failed. Stopping." % step_key)
                break
    else:
        for step_key in STEP_ORDER:
            ok = run_single_step(step_key, manifest, project_dir)
            if not ok:
                all_ok = False
                print("[Driver] Step %s failed. Stopping." % step_key)
                break

    return all_ok


def resume_pipeline(manifest, project_dir):
    """Resume from the next pending/failed step."""
    next_step = get_next_step(manifest)
    if next_step is None:
        print("[Driver] All steps complete! Nothing to resume.")
        return True

    step_num, step_name = STEP_ORDER.get(next_step, (0, next_step))
    print("[Driver] Resuming from step %d: %s" % (step_num, step_name))

    all_ok = True
    found = False
    for step_key in STEP_ORDER:
        if step_key == next_step:
            found = True
        if found:
            ok = run_single_step(step_key, manifest, project_dir)
            if not ok:
                all_ok = False
                print("[Driver] Step %s failed. Stopping." % step_key)
                break

    return all_ok


# ======================================================================
# CLI Entry Point
# ======================================================================


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="\u98ce\u63a7\u5efa\u6a21\u7edf\u4e00\u9a71\u9a76\u5165\u53e3 (Driver) \u2014 manifest.json \u59d4\u5a43\u9a71\u52a8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # \u5168\u6d41\u7a0b\u6267\u884c (\u9996\u6b21\u8fd0\u884c)
  python driver.py --project-dir ./hp_20260513 \\
      --raw-data ./data/raw/data.parquet \\
      --label-col p3m2 --month-col app_date --steps all

  # \u53ea\u8dd1\u7b2c3\u6b65
  python driver.py --project-dir ./hp_20260513 --step 3

  # \u65ad\u70b9\u7eed\u8dd1
  python driver.py --project-dir ./hp_20260513 --resume

  # \u6307\u5b9a\u6b65\u9aa4\u5217\u8868
  python driver.py --project-dir ./hp_20260513 \\
      --steps step1_eda,step3_modeling
        """,
    )
    parser.add_argument(
        "--project-dir", type=str, required=True,
        help="\u9879\u76ee\u6839\u76ee\u5f55 (manifest.json \u6240\u5728\u76ee\u5f55)",
    )
    parser.add_argument(
        "--raw-data", type=str, default="",
        help="\u539f\u59cb\u6570\u636e\u6587\u4ef6\u8def\u5f84 (\u9996\u6b21\u8fd0\u884c\u65f6\u9700\u8981)",
    )
    parser.add_argument(
        "--label-col", type=str, default=None,
        help="\u6807\u7b7e\u5217\u540d (\u9ed8\u8ba4 y, \u4ec5\u521d\u59cb\u5316 manifest \u65f6\u751f\u6548)",
    )
    parser.add_argument(
        "--month-col", type=str, default=None,
        help="\u6708\u4efd\u5217\u540d (\u9ed8\u8ba4 app_date)",
    )
    parser.add_argument(
        "--feature-cols", type=str, nargs="+", default=None,
        help="\u6307\u5b9a\u7279\u5f81\u5217 (\u9ed8\u8ba4\u81ea\u52a8\u63a8\u65ad)",
    )
    parser.add_argument(
        "--extra-score-cols", type=str, nargs="+", default=None,
        help="\u989d\u5916\u8bc4\u5206\u5217 (\u5916\u90e8\u5206\u6570)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help='Execution mode: "all" / comma-separated step names',
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run single step by number (1-5)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="\u4ece\u4e0a\u6b21\u5931\u8d25\u5904\u6062\u590d\u6267\u884c",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="\u8d1d\u53f6\u65af\u4f18\u5316\u8fed\u4ee3\u6b21\u6570 (\u9ed8\u8ba4 5)",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point."""
    args = parse_args(argv)

    project_dir = Path(args.project_dir).resolve()
    base_dir = SCRIPT_DIR.parent  # skill root

    # Ensure directory structure
    ensure_dirs(project_dir)

    t0 = time.time()

    # ---- Initialize / load manifest ----
    if not (project_dir / "manifest.json").exists():
        print("[Driver] Initializing manifest.json ...")

        extra_cols = args.extra_score_cols or []
        data_config_loaded = {}

        # Priority 1: Try to load from project dir first (user has already edited it)
        proj_cfg_path = project_dir / "config" / "data_config.yaml"
        if proj_cfg_path.exists():
            try:
                data_config_loaded = load_data_config(project_dir=project_dir)
            except Exception as e:
                print("[Driver] Failed to load project data_config.yaml: %s" % e)

        # Priority 2: Fall back to skill dir default config
        if not data_config_loaded:
            try:
                skill_base = SCRIPT_DIR.parent
                data_config_loaded = load_data_config(base_dir=skill_base)
            except Exception:
                pass

        # Copy data_config.yaml to project dir only if neither exists there
        proj_cfg_path = project_dir / "config" / "data_config.yaml"
        skill_cfg_path = base_dir / "config" / "data_config.yaml"
        if not proj_cfg_path.exists() and skill_cfg_path.exists():
            import shutil
            proj_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(skill_cfg_path), str(proj_cfg_path))
            print("[Driver] Copied data_config.yaml to project dir.")

        manifest = init_manifest(
            project_dir=project_dir,
            label_col=args.label_col or "y",
            month_col=args.month_col or None,
            raw_data_path=args.raw_data,
            extra_score_cols=extra_cols if extra_cols else None,
            data_config=data_config_loaded if data_config_loaded else None,
        )
    else:
        manifest = load_manifest(project_dir)

        # Update raw_data_path if provided and missing in manifest
        if args.raw_data and not manifest.get("parameters", {}).get("raw_data_path"):
            manifest["parameters"]["raw_data_path"] = args.raw_data
            save_manifest(project_dir, manifest)

        # Override label_col / month_col if explicitly passed
        if args.label_col:
            manifest.setdefault("parameters", {})["label_col"] = args.label_col
        if args.month_col:
            manifest.setdefault("parameters", {})["month_col"] = args.month_col

    print("[Driver] Project: %s" % project_dir.name)
    print("[Driver] Manifest: %s" % (project_dir / "manifest.json"))

    # ---- Print current step status ----
    print("\n[Driver] Step Status:")
    for step_key in STEP_ORDER:
        status = get_step_status(manifest, step_key)
        _, name = STEP_ORDER[step_key]
        icon = {"done": "[OK]", "failed": "[X]", "pending": "[ ]"}.get(status, "?")
        print("  %s %s: %s" % (icon, name, status))

    # ---- Determine execution mode ----
    if args.resume:
        success = resume_pipeline(manifest, project_dir)
    elif args.step is not None:
        success = run_single_step_by_number(args.step, manifest, project_dir)
    elif args.steps == "all":
        success = run_all_steps(manifest, project_dir)
    elif args.steps:
        # Comma-separated step names
        steps_list = [s.strip() for s in args.steps.split(",")]
        success = run_all_steps(manifest, project_dir, steps_filter=steps_list)
    else:
        # Default: resume if failed steps, otherwise full pipeline
        pending = get_pending_steps(manifest)
        if not pending:
            print("[Driver] All steps already done!")
            success = True
        else:
            has_failed = any(
                get_step_status(manifest, s) == "failed" for s in pending
            )
            if has_failed:
                print("[Driver] Found failed steps, resuming...")
                success = resume_pipeline(manifest, project_dir)
            else:
                print("[Driver] Running full pipeline (steps all)")
                success = run_all_steps(manifest, project_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    if success:
        print("[Driver] [OK] All requested steps completed successfully in %.1fs" % elapsed)
    else:
        print("[Driver] [X] Pipeline finished with errors in %.1fs" % elapsed)
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
