"""
第二步主入口：投票式特征筛选 (Vote-based Feature Selection)

职责（薄编排层）：
  1. CLI 参数解析
  2. 加载配置/阈值（含 var_keep_list / var_drop_list 人工干预）
  3. Step2 计算 IV
  4. 复用 Step1 PSI / 相关性矩阵结果
  5. 投票筛选：缺失率 + IV + PSI + 相关性，通过票数门槛保留特征
  6. 人工干预：var_keep_list 强制保留、var_drop_list 强制删除
  7. 输出 selected_features.csv + 投票详情表

用法：
    python main_step2.py --data-file raw.csv --base-data-file base.csv ...
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
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for common_utils

from pipeline_config import load_feature_select_thresholds, make_safe_task_id, collect_gc, _load_yaml_file
from outputs import is_function_enabled, load_output_config

# ---- IV 计算模块（Step2 自己计算） ----
try:
    from iv_calc import calculate_iv_with_bins
except ImportError:
    calculate_iv_with_bins = None

# ---- 投票筛选模块 ----
from vote_filter import vote_filter


# ============================================================
# 核心入口
# ============================================================

def run_feature_selection(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    month_col: Optional[str] = "month",
    project_dir: Union[str, Path, None] = None,
    base_dir: Union[str, Path, None] = None,
    task_id: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    投票式特征筛选。

    Parameters
    ----------
    df : pd.DataFrame       待筛选数据（wide table）
    df_base : pd.DataFrame  基准数据集（当前未使用，保留签名兼容性）
    label_col : str         标签列
    feature_cols : list     候选特征（None=自动推断）
    month_col : str         月份分组列
    project_dir : str       项目输出目录
    base_dir : str          skill 根目录（加载配置）
    task_id : str           任务 ID（用于文件名后缀）

    Returns
    -------
    selected_features : list[str]
    artifacts : dict
    """
    safe_task_id = make_safe_task_id(task_id)
    project_dir = Path(project_dir) if project_dir else SCRIPT_DIR.parent / "output"
    processed = project_dir / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 0. 加载阈值配置（含投票参数 + 人工干预列表）
    # ================================================================
    base_dir = base_dir or SCRIPT_DIR.parent.parent

    thresholds = load_feature_select_thresholds(project_dir=project_dir, skill_dir=base_dir)
    full_yaml = thresholds

    # 重新加载完整 YAML 以读取人工干预列表等顶层字段
    from pipeline_config import _load_yaml_file as _ylf
    _yaml_candidates = []
    if project_dir is not None:
        _yaml_candidates.append(Path(project_dir) / "config" / "feature_select_thresholds.yaml")
    if base_dir is not None:
        _yaml_candidates.append(Path(base_dir) / "config" / "feature_select_thresholds.yaml")
    for _yc in _yaml_candidates:
        if _yc.exists():
            full_yaml = _ylf(_yc, {})
            break
    else:
        full_yaml = {}

    # 加载 result 开关配置
    try:
        out_cfg = load_output_config(project_dir=project_dir, skill_dir=base_dir)
    except Exception:
        out_cfg = None

    missing_rate_max = thresholds.get("missing_rate_max", 0.85)
    iv_min = thresholds.get("iv_min", 0.01)
    psi_max = thresholds.get("psi_max", 0.25)
    corr_threshold_val = thresholds.get("corr_threshold", 0.80)

    feat_min = thresholds.get("feat_count_min", 200)
    feat_max = thresholds.get("feat_count_max", 250)
    vote_pass_count = thresholds.get("vote_pass_count", 3)

    # ---- 人工干预列表（从顶层 YAML 读取） ----
    var_keep_list_raw = full_yaml.get("var_keep_list", [])
    var_drop_list_raw = full_yaml.get("var_drop_list", [])

    # 兼容字符串格式（用户可能写 "a,b,c"）
    def _parse_list(raw):
        if isinstance(raw, str):
            return [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, list):
            return raw if raw else []
        return []

    var_keep_list = _parse_list(var_keep_list_raw)
    var_drop_list = _parse_list(var_drop_list_raw)

    # ================================================================
    # 确定特征列
    # ================================================================
    if feature_cols is None:
        exclude = {label_col}
        if month_col:
            exclude.add(month_col)
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]

    # -- Iron Rule 2: driver.py 已经传入 wide_table_sampled.parquet 的数据 --
    # Step2 不再在内部做二次采样，直接使用传入的 sampled 数据
    print(f"  [IronRule2] Using sampled data from driver: {len(df)} rows")

    # -- 只保留当前任务需要的列（内存优化） --
    keep_cols = list(feature_cols) + [label_col]
    if month_col and month_col not in keep_cols:
        keep_cols.append(month_col)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df_base = df_base[keep_cols] if df_base is not None else df

    print("=" * 60)
    print("第二步：投票式特征筛选 (Vote-based Feature Selection)")
    print("=" * 60)
    print("Candidate features: %d" % len(feature_cols))
    print("Vote pass threshold: >= %d conditions" % vote_pass_count)
    if var_keep_list:
        print("var_keep_list (force keep): %d features" % len(var_keep_list))
    if var_drop_list:
        print("var_drop_list (force drop): %d features" % len(var_drop_list))

    artifacts = {}

    # ================================================================
    # 1. Step2 计算 IV（全量、不分label，文件存在则跳过）
    # ================================================================
    iv_base_path = processed / "iv_summary.xlsx"
    iv_done = iv_base_path.exists() and iv_base_path.stat().st_size > 0

    iv_df = None
    if iv_done:
        print("\n[Step 2-1] IV already computed, loading cached result.")
        try:
            iv_df = pd.read_excel(str(iv_base_path))
            artifacts["iv"] = str(iv_base_path)
            print(f"  [OK] Loaded cached IV from {iv_base_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load cached iv_summary.xlsx: {e}")
    else:
        iv_path = str(processed / "iv_summary.xlsx")
        print("\n" + "=" * 50)
        print("[Step 2-1] IV Calculation")
        print("=" * 50)
        try:
            if calculate_iv_with_bins is not None:
                iv_result = calculate_iv_with_bins(
                    data=df,
                    label_col=label_col,
                    features=feature_cols,
                    positive=1,
                    group_cols=None,  # 全量计算，不分组
                    bin_method="chimerge",
                    output_excel=iv_path,
                )
                if iv_result is not None:
                    if isinstance(iv_result, tuple):
                        iv_df = iv_result[0]
                    else:
                        iv_df = iv_result
                    artifacts["iv"] = iv_path
                    print(f"  [OK] IV calculated and saved to {iv_path}")
                collect_gc()
            else:
                print("  [WARN] iv_calc module not available, skipping IV calculation")
        except Exception as e:
            import traceback
            print(f"IV calculation failed: {e}")
            traceback.print_exc()

    # ================================================================
    # 2. 加载 Step1 缺失率结果（全量、不分label）
    # ================================================================
    missing_rate_path = str(processed / "missing_rate.xlsx")

    missing_rate_df = None
    if Path(missing_rate_path).exists():
        try:
            missing_rate_df = pd.read_excel(missing_rate_path)
            print("\n[OK] Reused missing_rate.xlsx from Step1")
        except Exception:
            print("\n[WARN] Failed to read missing_rate.xlsx, skipping MR vote")

    # ================================================================
    # 3. 加载 Step1 逐月 PSI 结果（可选，受 psi_filter 开关控制）
    # ================================================================
    psi_enabled = is_function_enabled("step2_feature_select", "psi_filter", out_cfg)
    psi_path = str(processed / "psi_monthly.xlsx")

    psi_df = None
    if psi_enabled and Path(psi_path).exists():
        try:
            psi_raw = pd.read_excel(psi_path, sheet_name="Monthly_PSI")
            # 兼容两种 PSI 格式：
            # 1) 长表: feature + psi_value
            # 2) 宽表: feature + 月份列 (202509, 202510, ...) → 取每月最大PSI
            if "psi_value" in psi_raw.columns:
                psi_df = psi_raw.groupby("feature")["psi_value"].max().reset_index()
            else:
                # 宽表格式：第一列是 feature，其余都是月份 PSI 值
                feat_col = [c for c in psi_raw.columns if c in ("feature", "variable")][0]
                month_cols = [c for c in psi_raw.columns if c != feat_col]
                psi_vals = psi_raw[month_cols].apply(lambda row: row.dropna().max(), axis=1)
                psi_df = pd.DataFrame({feat_col: psi_raw[feat_col], "psi_value": psi_vals})
            artifacts["psi"] = psi_path
            print("[OK] Reused psi_monthly.xlsx from Step1 for PSI vote")
            collect_gc()
        except Exception as e:
            print("[WARN] Failed to read psi_monthly.xlsx: %s" % str(e))
    elif not psi_enabled:
        print("[Step 2-3] PSI filter disabled by config, skipping.")

    # ================================================================
    # 4. 加载 Step1 相关性矩阵（可选，受 corr_filter 开关控制）
    # ================================================================
    corr_enabled = is_function_enabled("step2_feature_select", "corr_filter", out_cfg)

    corr_path = str(processed / ("correlation_matrix" + safe_task_id + ".xlsx"))
    if not Path(corr_path).exists():
        corr_path = str(processed / "correlation_matrix.xlsx")

    corr_matrix_df = None
    if corr_enabled and Path(corr_path).exists():
        try:
            corr_matrix_df = pd.read_excel(corr_path, sheet_name="Correlation_Matrix", header=0, index_col=0)
            artifacts["correlation"] = corr_path
            print("[OK] Reused correlation_matrix.xlsx from Step1 for Corr vote")
        except Exception as e:
            print("[WARN] Failed to read correlation_matrix.xlsx: %s" % str(e))
    elif not corr_enabled:
        print("[Step 2-4] Correlation filter disabled by config, skipping.")

    # ================================================================
    # 5. 投票筛选
    # ================================================================
    print("\n" + "=" * 50)
    print("[Step 2-2] Vote-based Feature Filtering")
    print("=" * 50)

    try:
        selected, vote_detail_df = vote_filter(
            feature_list=feature_cols,
            missing_rate_df=missing_rate_df,
            psi_df=psi_df,
            iv_df=iv_df,
            corr_matrix_df=corr_matrix_df,
            missing_rate_max=missing_rate_max,
            psi_threshold=psi_max,
            iv_min=iv_min,
            corr_threshold=corr_threshold_val,
            vote_pass_count=vote_pass_count,
            var_keep_list=var_keep_list,
            var_drop_list=var_drop_list,
        )
    except Exception as e:
        import traceback
        print("Vote filter failed: %s" % str(e))
        traceback.print_exc()
        # Fallback: 全部保留
        selected = list(feature_cols)
        vote_detail_df = None


    # ================================================================
    # 5. Auto-Tune: 自适应调节 IV/PSI 阈值使选中数量落入 [feat_min, feat_max]
    # ================================================================
    if vote_detail_df is not None:
        print()
        print('[Auto-Tune] Checking selection count vs target range [%d, %d]...' % (feat_min, feat_max))

        if not (feat_min <= len(selected) <= feat_max):
            print('[Auto-Tune] Outside target range. Searching for better IV/PSI thresholds...')

            import numpy as np_num
            # Load IV and PSI lookup tables
            iv_lookup_dict = iv_df.groupby('variable')['info_value'].max().to_dict()
            if psi_df is not None:
                feat_col_psi = 'feature' if 'feature' in psi_df.columns else 'variable'
                val_col_psi = 'psi_value' if 'psi_value' in psi_df.columns else 'psi'
                psi_lookup_dict = psi_df.groupby(feat_col_psi)[val_col_psi].max().to_dict()
            else:
                psi_lookup_dict = None

            # For each feature, get (iv, psi)
            feature_iv = []
            feature_psi = []
            for f in feature_cols:
                v_iv = iv_lookup_dict.get(f, 0.0)
                v_psi = psi_lookup_dict.get(f, 0.0) if psi_lookup_dict else 0.0
                feature_iv.append(v_iv)
                feature_psi.append(v_psi)

            # Current counts per threshold
            def count_at(iv_t, psi_t):
                c = 0
                for j in range(len(feature_cols)):
                    passes_iv = feature_iv[j] >= iv_t
                    passes_psi = feature_psi[j] < psi_t
                    if passes_iv and passes_psi:
                        c += 1
                return c

            # Binary search on IV first (keep PSI fixed)
            target = (feat_min + feat_max) // 2
            best_iv = iv_min
            best_count = count_at(iv_min, psi_max)

            # Search IV in [0.0005, 0.05]
            iv_candidates = np_num.linspace(0.0005, 0.05, 100)
            for iv_c in iv_candidates:
                c = count_at(iv_c, psi_max)
                if feat_min <= c <= feat_max:
                    best_iv = float(iv_c)
                    best_count = c
                    break

            if not (feat_min <= best_count <= feat_max):
                # Try adjusting both IV and PSI
                psi_candidates = np_num.linspace(0.1, 0.5, 50)
                for iv_c in iv_candidates:
                    found = False
                    for psi_c in psi_candidates:
                        c = count_at(iv_c, psi_c)
                        if feat_min <= c <= feat_max:
                            best_iv = float(iv_c)
                            psi_max = float(psi_c)
                            best_count = c
                            found = True
                            break
                    if found:
                        break

            if feat_min <= best_count <= feat_max and abs(best_iv - iv_min) > 1e-6:
                print('[Auto-Tune] Found: iv_min=%.4f, psi_max=%.4f -> ~%d features' % (best_iv, psi_max, best_count))
                # Re-run vote_filter with new thresholds
                selected, vote_detail_df = vote_filter(
                    feature_list=feature_cols,
                    missing_rate_df=missing_rate_df,
                    psi_df=psi_df,
                    iv_df=iv_df,
                    corr_matrix_df=corr_matrix_df,
                    missing_rate_max=missing_rate_max,
                    psi_threshold=psi_max,
                    iv_min=best_iv,
                    corr_threshold=corr_threshold_val,
                    vote_pass_count=vote_pass_count,
                    var_keep_list=var_keep_list,
                    var_drop_list=var_drop_list,
                )
                print('[Auto-Tune] Final selected: %d features (target [%d, %d])' % (len(selected), feat_min, feat_max))
            else:
                print('[Auto-Tune] Could not find thresholds within target range. Using original results (%d features).' % len(selected))
        else:
            print('[Auto-Tune] Already within target. No adjustment needed.')

    # ================================================================
    # 6. 保存投票详情表（供人工审核）
    # ================================================================
    if vote_detail_df is not None:
        vote_detail_path = str(processed / ("vote_detail" + safe_task_id + ".xlsx"))
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except ImportError:
            engine = None
        with pd.ExcelWriter(vote_detail_path, engine=engine) as writer:
            vote_detail_df.to_excel(writer, sheet_name="Vote_Detail")
        artifacts["vote_detail"] = vote_detail_path
        print("\n[OK] Vote detail saved to %s" % vote_detail_path)

    # ================================================================
    # 7. 保存最终名单
    # ================================================================
    selected_path = str(processed / ("selected_features" + safe_task_id + ".csv"))
    pd.DataFrame({"feature": selected}).to_csv(selected_path, index=False)
    artifacts["selected_features"] = selected_path

    collect_gc()

    print("\n" + "=" * 60)
    print("Step 2 Complete! Final selected features: %d (target: [%d, %d])" % (
        len(selected), feat_min, feat_max))
    if not (feat_min <= len(selected) <= feat_max):
        print("NOTE: Selected count outside target range; consider adjusting vote_pass_count or thresholds.")
    print("=" * 60)

    return selected, artifacts


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Vote-based Feature Selection")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--base-data-file", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="target")
    parser.add_argument("--month-col", type=str, default="month")
    parser.add_argument("--feature-cols", type=str, nargs="+")
    parser.add_argument("--project-dir", type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.data_file)
    df_base = pd.read_csv(args.base_data_file)

    selected, artifacts = run_feature_selection(
        df=df, df_base=df_base,
        label_col=args.label_col,
        feature_cols=args.feature_cols,
        month_col=args.month_col,
        project_dir=args.project_dir,
    )

    print("\nArtifacts saved:")
    for k, v in artifacts.items():
        print("  %s: %s" % (k, v))
