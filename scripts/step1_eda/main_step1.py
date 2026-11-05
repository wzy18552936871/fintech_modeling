"""
第一步主入口：深度探索性数据分析 (Deep EDA)

职责（薄编排层）：
  1. CLI 参数解析 / build_wide_table 调用
  2. 依次调用子模块：预处理 → 缺失率 → 标签分布 → 基础统计
     → X变量交叉Y的EDA(lift/IV) → 变量明细 → 全量报告 Excel

实际业务逻辑在各子模块中。
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

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for pipeline_config

from data_preprocess import explore_and_convert
from missing_rate import compute_missing_rate
from eda_basic import run_basic_eda as run_eda_basic_stats
from eda_variable_detail import run_variable_eda as run_eda_variable_detail
from x_y_cross_eda import run_x_y_cross_eda
from outputs import load_output_config, is_function_enabled, resolve_filename
from pipeline_config import make_safe_task_id, collect_gc

# ---- PSI + Correlation in Step1; IV moved to Step2 ----
try:
    from psi_calc import calculate_psi
except ImportError:
    calculate_psi = None
try:
    from correlation import plot_correlation_heatmap
except ImportError:
    plot_correlation_heatmap = None


# ============================================================
# Label Distribution (inline from step2's 4.标签分布.py)
# ============================================================

def calc_label_dist(
    df: pd.DataFrame,
    label_col: str,
    month_col: Optional[str] = None,
    output_excel_path: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute label distribution stats and optionally save to Excel."""
    result: Dict[str, Any] = {}

    total = len(df)
    bad_count = df[label_col].sum() if df[label_col].dtype in (int, float) else (df[label_col] == 1).sum()
    good_count = total - bad_count
    bad_rate = bad_count / total if total > 0 else 0.0

    result["total"] = total
    result["good_count"] = int(good_count)
    result["bad_count"] = int(bad_count)
    result["bad_rate"] = round(float(bad_rate), 6)

    # By month
    if month_col and month_col in df.columns:
        grouped = df.groupby(month_col).agg(
            total=(label_col, "count"),
            bad=(label_col, "sum"),
        ).reset_index()
        grouped["bad_rate"] = grouped["bad"] / grouped["total"]
        result["by_month"] = grouped

    if output_excel_path:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except ImportError:
            engine = None
        with pd.ExcelWriter(str(output_excel_path), engine=engine) as writer:
            summary_df = pd.DataFrame([
                {"metric": "total", "value": result["total"]},
                {"metric": "good_count", "value": result["good_count"]},
                {"metric": "bad_count", "value": result["bad_count"]},
                {"metric": "bad_rate", "value": result["bad_rate"]},
            ])
            summary_df.to_excel(writer, sheet_name="Label_Summary", index=False)
            if "by_month" in result:
                result["by_month"].to_excel(writer, sheet_name="By_Month", index=False)

    print(f"  Total={result['total']}, Bad={result['bad_count']} ({result['bad_rate']:.4%})")
    return result


# ============================================================
# 构建 Wide Table（共享宽表）
# ============================================================

def _save_dual_wide_tables(
    df_full: pd.DataFrame,
    keep_cols: List[str],
    processed: Path,
    max_features: int = 500,
    max_rows: int = 200000,
    sample_size: int = 100000,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    铁律二：双宽表设计
    - wide_table_full.parquet   : 始终全量，Step3建模用
    - wide_table_sampled.parquet: ≥500变量或≥20万行时10万样本，EDA/PSI/IV/相关性用
    只保留需要的列（特征+标签+月份），不保留无关列。
    """
    # 裁剪到需要的列
    df_full_trimmed = df_full[[c for c in keep_cols if c in df_full.columns]].copy()

    full_path = processed / "wide_table_full.parquet"
    sampled_path = processed / "wide_table_sampled.parquet"

    x_count = len(keep_cols)
    row_count = len(df_full_trimmed)

    need_sample = (x_count >= max_features) or (row_count > max_rows)

    if not need_sample:
        # 两个文件内容相同，只写一份再复制
        df_full_trimmed.to_parquet(str(full_path), index=False, engine="pyarrow")
        import shutil
        shutil.copy2(str(full_path), str(sampled_path))
        print(f"  [WideTable] Full + Sampled (same data): {row_count} rows, {x_count} cols")
    else:
        # 全量写入
        df_full_trimmed.to_parquet(str(full_path), index=False, engine="pyarrow")
        print(f"  [WideTable] Full: {row_count} rows, {x_count} cols -> {full_path.name}")
        gc.collect()

        # 采样写入
        n = min(sample_size, row_count)
        df_sampled = df_full_trimmed.sample(n=n, replace=False, random_state=seed)
        df_sampled.to_parquet(str(sampled_path), index=False, engine="pyarrow")
        print(f"  [WideTable] Sampled: {n} rows, {x_count} cols -> {sampled_path.name}")

    return str(full_path), str(sampled_path)


def build_wide_table(
    raw_data_path: str,
    feature_cols: List[str],
    label_col: str = "dummy",
    month_col: str = "app_date",
    extra_cols: Optional[List[str]] = None,
    project_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    铁律二：构建双宽表，列裁剪
    - wide_table_full.parquet   : 全量行 + 需要的列（建模/缺失率/标签分布用）
    - wide_table_sampled.parquet: 采样行 + 需要的列（PSI/IV/相关性/X-Y交叉用）
    
    需要的列 = x_cols + labels + month_col + group_by_cols + external_scores
    """
    if project_dir is None:
        project_dir = SCRIPT_DIR.parent / "output"
    processed = Path(project_dir) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    full_path = processed / "wide_table_full.parquet"
    sampled_path = processed / "wide_table_sampled.parquet"

    print("\n" + "=" * 60)
    print("[Step 1] Building Dual Wide Tables (列裁剪 + 双宽表)")
    print("=" * 60)
    print("  Raw data: %s" % raw_data_path)

    from pipeline_config import smart_read_df, load_data_config

    # 加载 data_config.yaml 获取完整的列信息
    from pipeline_config import smart_read_df, load_data_config

    try:
        data_cfg = load_data_config(project_dir=project_dir)
        print("  [Config] Loaded data_config.yaml")
    except Exception:
        data_cfg = {}
    if not data_cfg:
        try:
            data_cfg = load_data_config(base_dir=base_dir)
            print("  [Config] Loaded data_config.yaml from skill dir")
        except Exception as e:
            print(f"  [Config] Failed to load: {e}")
            data_cfg = {}
        data_cfg = {}

    # 读取原始数据
    try:
        df_raw = smart_read_df(raw_data_path, dtype=str)
    except Exception as e:
        print(f"  Wide table build failed: {e}")
        return ""

    if "month" in df_raw.columns:
        df_raw["month"] = pd.to_numeric(df_raw["month"], errors="coerce")
    # ⚠️ drop_cols 必须从配置文件读取，绝不能写死！
    for col in data_cfg.get("drop_cols", []):
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)
    gc.collect()

    # 构建 keep_cols: x_cols + labels + month_col + group_by_cols + external_scores
    keep_cols = list(feature_cols)  # x_cols

    # 添加所有标签列
    labels_pool = data_cfg.get("labels", {})
    for lbl_key, lbl_def in labels_pool.items():
        col_name = lbl_def.get("col_name", lbl_key)
        if col_name not in keep_cols:
            keep_cols.append(col_name)

    # 添加 month_col
    dims_cfg = data_cfg.get("analysis_dims", {})
    mc = dims_cfg.get("month_col", month_col)
    if mc and mc not in keep_cols:
        keep_cols.append(mc)

    # 添加 group_by_cols
    for c in dims_cfg.get("group_by_cols", []):
        if c not in keep_cols:
            keep_cols.append(c)

    # 添加 external_scores
    for c in dims_cfg.get("external_scores", []):
        if c not in keep_cols:
            keep_cols.append(c)

    # 如果 extra_cols 传入也加上
    if extra_cols:
        for c in extra_cols:
            if c not in keep_cols:
                keep_cols.append(c)

    # 只保留实际存在的列
    keep_cols = [c for c in keep_cols if c in df_raw.columns]

    print(f"  Keeping {len(keep_cols)} columns (x_cols + labels + month + group_by + external_scores)")

    # 先裁剪列，再做类型转换
    use_cols = list(set(keep_cols))
    try:
        df_clean, _dtype_map = explore_and_convert(df_raw, use_cols)
    except Exception as e:
        print(f"  Wide table convert note: {e} (saving raw as-is)")
        df_clean = df_raw

    # 再次裁剪到需要的列（explore_and_convert可能返回更多列）
    final_cols = [c for c in keep_cols if c in df_clean.columns]
    df_trimmed = df_clean[final_cols].copy()
    collect_gc()

    # 调用 _save_dual_wide_tables 生成两个文件
    try:
        _save_dual_wide_tables(df_trimmed, final_cols, processed)
        print(f"  [OK] wide_table_full.parquet: {len(df_trimmed)} rows x {len(final_cols)} cols")
    except Exception as e:
        print(f"  Wide table save failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

    return str(full_path)


# ============================================================
# 核心 EDA 编排
# ============================================================

def run_eda(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    month_col: Optional[str] = "app_date",
    project_dir: Union[str, Path, None] = None,
    base_dir: Union[str, Path, None] = None,
    raw_data_path: Optional[str] = None,
    extra_cols: Optional[List[str]] = None,
    task_id: Optional[str] = None,
    group_by_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parameters
    ----------
    df : pd.DataFrame        数据
    df_base : pd.DataFrame   基准数据（PSI 用）
    label_col : str          标签列
    feature_cols : list      特征列
    month_col : str          月份列
    project_dir : str        输出目录
    base_dir : str           skill 根目录
    raw_data_path : str      原始数据路径（None=wide table已建）
    extra_cols : list        额外评分列
    task_id : str            任务ID（文件名后缀）
    group_by_cols : list     分组列

    Returns
    -------
    df_out : pd.DataFrame    处理后数据
    artifacts : dict         产物路径映射
    """
    safe_task_id = make_safe_task_id(task_id)
    project_dir = Path(project_dir) if project_dir else SCRIPT_DIR.parent / "output"
    processed = project_dir / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    base_dir = base_dir or SCRIPT_DIR.parent.parent

    try:
        out_cfg = load_output_config(project_dir=project_dir, skill_dir=base_dir)
    except Exception:
        out_cfg = None

    s1 = "step1_eda"

    artifacts: Dict[str, str] = {}

    # -- 先自动推断特征列（避免后面再做一次） --
    if feature_cols is None:
        exclude = {label_col}
        if month_col:
            exclude.add(month_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    # -- 只保留当前任务需要的列，避免全表拷贝（内存优化） --
    keep_cols = list(feature_cols) + [label_col]
    if month_col and month_col not in keep_cols:
        keep_cols.append(month_col)
    if group_by_cols:
        for c in group_by_cols:
            if c not in keep_cols:
                keep_cols.append(c)
    # 只保留实际存在的列
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    # -- 过滤 group_by_cols，只保留实际存在的列 --
    if group_by_cols:
        group_by_cols = [c for c in group_by_cols if c in df_out.columns]
    else:
        group_by_cols = []

    print("=" * 60)
    print("第一步：深度 EDA (%s)" % task_id if task_id else "第一步：深度 EDA")
    print("=" * 60)
    print("Features: %d" % len(feature_cols))

    # -- 检查 label_col 是否存在 --
    if label_col not in df_out.columns:
        print(f"WARNING: Label column '{label_col}' NOT found in data.")
        print(f"  Available columns (first 30): {list(df_out.columns[:30])}")

    # ================================================================
    # 1. 缺失率统计（全量、不分label，文件存在则跳过）
    # ================================================================
    mr_base_path = processed / "missing_rate.xlsx"
    mr_done = mr_base_path.exists() and mr_base_path.stat().st_size > 0

    if (out_cfg is None or is_function_enabled(s1, "missing_rate", out_cfg)) and not mr_done:
        mr_path = str(mr_base_path)
        print("\n" + "=" * 50)
        print("[Step 1-1] Missing Rate")
        print("=" * 50)
        try:
            compute_missing_rate(
                df=df_out, output_excel_path=mr_path,
                date_col=None, group_cols=None,  # 全量计算（df_out已是裁剪后的全量），不分组
                feature_cols=feature_cols,
            )
            artifacts["missing_rate"] = mr_path
        except Exception as e:
            print("Missing rate failed: %s" % str(e))
    else:
        if mr_done:
            artifacts["missing_rate"] = str(mr_base_path)
            print("\n[Step 1-1] Missing Rate already done, skipping.")
        else:
            print("\n[Step 1-1] missing_rate disabled, skipping.")

    # ================================================================
    # 2. 标签分布
    # ================================================================
    if out_cfg is None or is_function_enabled(s1, "label_distribution", out_cfg):
        lb_path = str(processed / ("label_distribution" + safe_task_id + ".xlsx"))
        print("\n" + "=" * 50)
        print("[Step 1-2] Label Distribution")
        print("=" * 50)
        try:
            if label_col not in df_out.columns:
                raise KeyError(f"Label column '{label_col}' not found in data.")
            calc_label_dist(
                df=df_out, label_col=label_col, month_col=month_col,
                output_excel_path=lb_path, feature_cols=feature_cols,
            )
            artifacts["label_distribution"] = lb_path
        except Exception as e:
            print("Label distribution failed: %s" % str(e))
    else:
        print("\n[Step 1-2] label_distribution disabled, skipping.")

    # ================================================================
    # 3. 基础 EDA 统计
    # ================================================================
    if out_cfg is None or is_function_enabled(s1, "eda_basic", out_cfg):
        eda_path = processed / ("eda_basic" + safe_task_id + ".xlsx")
        print("\n" + "=" * 50)
        print("[Step 1-3] EDA Basic Statistics")
        print("=" * 50)
        try:
            from pipeline_config import load_eda_sampling
            eda_sampling = load_eda_sampling(project_dir=project_dir, skill_dir=base_dir)
            eda_cfg_basic = {"sampling": eda_sampling}
            basic_results = run_eda_basic_stats(
                df=df_out, label_col=label_col,
                month_col=month_col, group_cols=group_by_cols,
                eda_cfg=eda_cfg_basic,
            )
            # Guard: at least one sheet with data
            has_data = any(not v.empty for v in basic_results.values()) if basic_results else False
            if has_data:
                try:
                    import openpyxl  # noqa: F401
                    engine = "openpyxl"
                except ImportError:
                    engine = None
                with pd.ExcelWriter(str(eda_path), engine=engine) as writer:
                    for sheet_name, df_s in basic_results.items():
                        if not df_s.empty:
                            df_s.to_excel(writer, sheet_name=sheet_name, index=False)
                artifacts["eda_basic"] = str(eda_path)
            else:
                print("  No data to write for EDA Basic (empty results).")
        except Exception as e:
            print("EDA basic stats failed: %s" % str(e))
    else:
        print("\n[Step 1-3] eda_basic disabled, skipping.")

    # ================================================================
    # 4. X变量交叉Y的详细 EDA（lift / IV）
    # ================================================================
    if out_cfg is None or is_function_enabled(s1, "x_y_cross_eda", out_cfg):
        cross_path = str(processed / ("x_y_cross_eda" + safe_task_id + ".xlsx"))
        print("\n" + "=" * 50)
        print("[Step 1-4] X-Y Cross EDA (Lift / IV)")
        print("=" * 50)
        try:
            from pipeline_config import load_eda_sampling, get_default_variable_eda_config
            eda_sampling = load_eda_sampling(project_dir=project_dir, skill_dir=base_dir)
            var_eda_defaults = get_default_variable_eda_config()
            eda_cfg_cross = {"sampling": eda_sampling, "variable_eda": var_eda_defaults}

            # Guard: skip if label column missing
            if label_col not in df_out.columns:
                print(f"  Label column '{label_col}' NOT found in data, skipping X-Y Cross EDA.")
            else:
                from eda_variable_detail import _maybe_sample
                exclude_x = {label_col}
                if month_col:
                    exclude_x.add(month_col)
                x_cols_cross = [c for c in feature_cols if c in df_out.columns and c not in exclude_x]

                if not x_cols_cross:
                    print("  No feature columns for X-Y cross EDA, skipping.")
                else:
                    # 对 X+Y 一起抽样（保证 label 对齐）
                    df_x_for_sample = df_out[x_cols_cross]
                    df_sampled_x, is_full = _maybe_sample(df_x_for_sample, eda_cfg_cross)
                    if is_full:
                        df_for_eda = df_out[x_cols_cross + [label_col]].copy()
                    else:
                        sample_n = eda_sampling.get("sample_size", 100000)
                        seed = eda_sampling.get("random_seed", 42)
                        indices = df_x_for_sample.sample(
                            n=min(sample_n, len(df_x_for_sample)),
                            replace=False, random_state=seed
                        ).index
                        df_for_eda = df_out.loc[indices, x_cols_cross + [label_col]].copy()

                    cross_df, bin_details = run_x_y_cross_eda(
                        df=df_for_eda, label_col=label_col,
                        feature_cols=x_cols_cross, eda_cfg=eda_cfg_cross,
                    )
                    # 保存
                    try:
                        import openpyxl  # noqa: F401
                        engine = "openpyxl"
                    except ImportError:
                        engine = None
                    with pd.ExcelWriter(str(cross_path), engine=engine) as writer:
                        cross_df.to_excel(writer, sheet_name="X_Y_Cross_Eda", index=False)
                        # 新增 Sheet：数值型变量分箱明细
                        if bin_details:
                            bin_df = pd.DataFrame(bin_details)
                            bin_df.to_excel(writer, sheet_name="Bin_Details", index=False)
                    artifacts["x_y_cross_eda"] = cross_path
        except Exception as e:
            import traceback
            print("X-Y Cross EDA failed: %s" % str(e))
            traceback.print_exc()
    else:
        print("\n[Step 1-4] x_y_cross_eda disabled, skipping.")

    collect_gc()

    # ================================================================
    # 5. 变量明细 EDA（可选，数据量大时跳过）
    # ================================================================
    if out_cfg is None or is_function_enabled(s1, "eda_variable_detail", out_cfg):
        detail_path = processed / ("eda_variable_detail" + safe_task_id + ".xlsx")
        print("\n" + "=" * 50)
        print("[Step 1-5] EDA Variable Detail")
        print("=" * 50)
        try:
            from pipeline_config import load_eda_sampling, get_default_variable_eda_config
            eda_sampling = load_eda_sampling(project_dir=project_dir, skill_dir=base_dir)
            var_eda_defaults = get_default_variable_eda_config()
            # Merge variable_eda config from output_config if available
            s1_cfg = out_cfg.get("step1_eda", {}) if out_cfg else {}
            ve = s1_cfg.get("variable_eda", {})
            var_eda_defaults["compute_unique"] = ve.get("compute_unique", False)
            var_eda_defaults["special_values"] = ve.get("special_values", [0])
            var_eda_defaults["unique_to_char_threshold"] = ve.get("unique_to_char_threshold", 20)
            eda_cfg_detail = {"sampling": eda_sampling, "variable_eda": var_eda_defaults}

            var_results = run_eda_variable_detail(
                df=df_out, label_col=label_col, feature_cols=feature_cols,
                month_col=month_col, eda_cfg=eda_cfg_detail,
            )
            # Save to Excel with separate sheets (guard: at least one sheet must have data)
            has_num = not var_results["numeric"].empty
            has_cat = not var_results["categorical"].empty
            if has_num or has_cat:
                try:
                    import openpyxl  # noqa: F401
                    engine = "openpyxl"
                except ImportError:
                    engine = None
                with pd.ExcelWriter(str(detail_path), engine=engine) as writer:
                    if has_num:
                        var_results["numeric"].to_excel(writer, sheet_name="Numeric", index=False)
                    if has_cat:
                        var_results["categorical"].to_excel(writer, sheet_name="Categorical", index=False)
                artifacts["eda_variable_detail"] = str(detail_path)

            # If compute_unique is on and there are num-to-char conversions,
            # save the conversion list separately
            if var_results.get("num_to_char_vars"):
                nc_df = pd.DataFrame(var_results["num_to_char_vars"],
                                     columns=["variable", "unique_count"])
                nc_path = processed / ("eda_num_to_char_vars" + safe_task_id + ".csv")
                nc_df.to_csv(str(nc_path), index=False)
                artifacts["eda_num_to_char_vars"] = str(nc_path)
                print("  num-to-char vars saved: %s" % nc_path)
        except Exception as e:
            import traceback
            print("EDA variable detail failed: %s" % str(e))
            traceback.print_exc()
    else:
        print("\n[Step 1-5] eda_variable_detail disabled, skipping.")

    collect_gc()

    # ================================================================
    # 6. 逐月 PSI 计算（Step1，**基于sampled数据**、不分label）
    # base=第n-1月, compare=第n月，每个数据源只需算一次 → 文件存在且非空则跳过
    # ================================================================
    psi_base_path = processed / "psi_monthly.xlsx"
    psi_done = psi_base_path.exists() and psi_base_path.stat().st_size > 0

    if (out_cfg is None or is_function_enabled(s1, "compute_psi", out_cfg)) and not psi_done:
        print("\n" + "=" * 50)
        print("[Step 1-6] Monthly PSI Calculation (sampled data)")
        print("=" * 50)
        try:
            from psi_calc import calculate_monthly_psi
            from pipeline_config import load_feature_select_thresholds, load_eda_sampling
            thresholds_cfg = load_feature_select_thresholds(project_dir=project_dir, skill_dir=base_dir) if base_dir else {}
            psi_bins = thresholds_cfg.get("psi_bin_num", 10)

            # 使用 sampled 数据计算 PSI（铁律二）— 直接读取 wide_table_sampled.parquet
            sampled_psi_path = processed / "wide_table_sampled.parquet"
            if sampled_psi_path.exists():
                df_psi = pd.read_parquet(str(sampled_psi_path))
                print(f"  [PSI] Loaded wide_table_sampled.parquet: {len(df_psi)} rows, {len(df_psi.columns)} cols")
            else:
                # fallback: 若 sampled 文件不存在，使用 df_out
                df_psi = df_out
                print(f"  [PSI] Fallback to full data: {len(df_psi)} rows")

            psi_path_str = str(processed / "psi_monthly.xlsx")
            psi_result_summary, psi_result_detail = calculate_monthly_psi(
                df=df_psi,
                month_col=month_col,
                features=feature_cols,
                bin_num=psi_bins,
                output_excel_path=psi_path_str,
            )
            artifacts["psi_summary"] = psi_path_str
            print(f"  Monthly PSI saved: {psi_path_str}")
            collect_gc()
        except Exception as e:
            import traceback
            print(f"Monthly PSI calculation failed: {e}")
            traceback.print_exc()
    else:
        if psi_done:
            print("\n[Step 1-6] Monthly PSI already done, skipping.")
        else:
            print("\n[Step 1-6] Monthly PSI disabled, skipping.")

    # ================================================================
    # 7. 相关性分析（已在Step1）
    # ================================================================
    if out_cfg is None or is_function_enabled(s1, "correlation_check", out_cfg):
        corr_path = str(processed / ("correlation_matrix" + safe_task_id + ".xlsx"))
        corr_img_path = str(processed / ("correlation_heatmap" + safe_task_id + ".png"))
        print("\n" + "=" * 50)
        print("[Step 1-7] Correlation Analysis (sampled data)")
        print("=" * 50)
        try:
            from pipeline_config import load_eda_sampling
            eda_sampling_corr = load_eda_sampling(project_dir=project_dir, skill_dir=base_dir) if base_dir else {}

            # 使用 sampled 数据计算相关性（铁律二）— 直接读取 wide_table_sampled.parquet
            sampled_corr_path = processed / "wide_table_sampled.parquet"
            if sampled_corr_path.exists():
                df_corr = pd.read_parquet(str(sampled_corr_path))
                print(f"  [Corr] Loaded wide_table_sampled.parquet: {len(df_corr)} rows, {len(df_corr.columns)} cols")
            else:
                # fallback: 若 sampled 文件不存在，使用 df_out
                df_corr = df_out
                print(f"  [Corr] Fallback to full data: {len(df_corr)} rows")

            x_count_corr = len(feature_cols)

            # 限制变量数：相关性矩阵为 O(n²)，630变量=40万格子
            max_corr_features = eda_sampling_corr.get("max_corr_features", 300)
            if x_count_corr > max_corr_features:
                import random as _rcorr
                _rcorr.seed(eda_sampling_corr.get("random_seed", 42))
                feature_cols_corr = _rcorr.sample(feature_cols, max_corr_features)
            else:
                feature_cols_corr = feature_cols

            # Compute correlation matrix on numeric features
            num_cols = [c for c in feature_cols_corr if df_corr[c].dtype in ("float64", "int64")]
            if len(num_cols) >= 2:
                corr_matrix = df_corr[num_cols].corr().abs()

                # Save correlation matrix to Excel
                try:
                    import openpyxl  # noqa: F401
                    engine = "openpyxl"
                except ImportError:
                    engine = None
                with pd.ExcelWriter(str(corr_path), engine=engine) as writer:
                    corr_matrix.to_excel(writer, sheet_name="Correlation_Matrix")
                    artifacts["correlation_matrix"] = corr_path
                    print(f"  Correlation matrix saved: {corr_path}")

                # Save heatmap image if seaborn available
                if plot_correlation_heatmap is not None:
                    try:
                        heat_feats = num_cols[:min(len(num_cols), 80)]
                        plot_correlation_heatmap(
                            df_corr, heat_feats, save_path=corr_img_path,
                            annot=False,
                        )
                        artifacts["correlation_heatmap"] = corr_img_path
                        print(f"  Correlation heatmap saved: {corr_img_path}")
                    except Exception as he:
                        print(f"  Heatmap plot failed (non-blocking): {he}")
            else:
                print("  [SKIP] Insufficient numeric features for correlation.")
        except Exception as e:
            import traceback
            print(f"Correlation analysis failed: {e}")
            traceback.print_exc()
    else:
        print("\n[Step 1-7] Correlation disabled, skipping.")

    collect_gc()

    # ================================================================
    # 双宽表保存（铁律二）
    # ================================================================
    wide_full_path = processed / "wide_table_full.parquet"
    wide_sampled_path = processed / "wide_table_sampled.parquet"
    if not wide_full_path.exists() or not wide_sampled_path.exists():
        print("\n" + "=" * 50)
        print("[Step 1-8] Save Dual Wide Tables")
        print("=" * 50)
        try:
            _save_dual_wide_tables(df_out, keep_cols, processed)
            artifacts["wide_table_full"] = str(wide_full_path)
            artifacts["wide_table_sampled"] = str(wide_sampled_path)
            print("  Saved: wide_table_full.parquet + wide_table_sampled.parquet")
        except Exception as e:
            print(f"Wide table save failed: {e}")

    # ================================================================
    # 8. 数据集拆分 → Train / Valid / OOT parquet（Step 3直接读取）
    # 只拆一次，包含全部特征。每个label各一套train/valid/oot。
    # data_split参数以第一个数据源为准
    # ================================================================
    try:
        from pipeline_config import load_data_config
        from data_split import split_wide_table_and_save, get_source_split_config
        dataset_dir = str(processed / "datasets")

        # 检查是否已有数据集
        existing = list(Path(dataset_dir).glob("dataset_*.parquet")) if Path(dataset_dir).exists() else []
        if existing:
            print("\n[Step 1-8] Datasets already exist (%d files), skipping." % len(existing))
        else:
            print("\n" + "=" * 50)
            print("[Step 1-8] Split Train / Valid / OOT Datasets")
            print("=" * 50)

            # 读取所有label列名（去重）
            data_config_map = load_data_config(project_dir=project_dir, base_dir=base_dir)
            all_label_cols = []
            seen_labels = set()
            for src_val in data_config_map.get("data_sources", {}).values():
                for lbl_key in src_val.get("labels", []):
                    label_info = data_config_map.get("labels", {}).get(lbl_key, {})
                    col_name = label_info.get("col_name", lbl_key)
                    if col_name not in seen_labels:
                        all_label_cols.append(col_name)
                        seen_labels.add(col_name)

            # 用全部特征列（source_all的x_cols）
            base_source = data_config_map.get("data_sources", {}).get("source_all", {})
            all_features = base_source.get("x_cols", feature_cols) if base_source else feature_cols

            # 取第一个有配置的data_split参数
            first_cfg = {}
            for src_val in data_config_map.get("data_sources", {}).values():
                cfg_candidate = get_source_split_config(src_val)
                if cfg_candidate:
                    first_cfg = cfg_candidate
                    break
            if not first_cfg:
                first_cfg = {"oot_tail_months": 4, "val_ratio": 0.3, "random_state": 42}

            # 使用全量宽表进行数据集拆分
            full_p = str(processed / "wide_table_full.parquet")
            split_results = split_wide_table_and_save(
                wide_table_path=full_p,
                month_col=month_col,
                label_cols=all_label_cols,
                feature_cols=all_features,
                output_dir=dataset_dir,
                **first_cfg,
            )
            if split_results:
                artifacts["datasets"] = dataset_dir
                for key, fpath in split_results.items():
                    print("  Split %s: %s" % (key, fpath))
    except Exception as e:
        import traceback
        print("\n[Step 1-8] Dataset split failed (non-blocking): %s" % e)
        traceback.print_exc()

    # -- 总结 --
    print("\n" + "=" * 60)
    print("Step 1 Complete!")
    print("=" * 60)
    for k, v in artifacts.items():
        print("  [%s] -> %s" % (k, v))

    collect_gc()
    return df_out, artifacts


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Deep EDA")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="target")
    parser.add_argument("--feature-cols", type=str, nargs="+")
    parser.add_argument("--month-col", type=str, default="month")
    parser.add_argument("--project-dir", type=str)

    args = parser.parse_args()
    df = pd.read_csv(args.data_file)

    df_out, artifacts = run_eda(
        df=df, df_base=df.copy(),
        label_col=args.label_col, feature_cols=args.feature_cols,
        month_col=args.month_col, project_dir=args.project_dir,
    )

    print("\nArtifacts saved:")
    for k, v in artifacts.items():
        print("  %s: %s" % (k, v))
