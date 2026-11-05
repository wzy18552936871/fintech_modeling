"""
X变量交叉Y的详细 EDA 统计模块

对每个特征 X 与标签 Y 交叉计算：
  - 数值型: null_lift + 有值部分 10 分箱 lift
  - 字符型: top10 取值 lift
  - 字符型: 超过阈值占比的取值，按 lift 排序的 top5 / bot5

输出格式示例（字符串列）：
  兰州 : 1200 : 2.6 || 武汉 : 1400 : 2.4 || ...
"""
from __future__ import print_function, unicode_literals

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Lift 计算核心
# ============================================================

def _lift_from_bad_rate(group_bad_rate: float, overall_bad_rate: float) -> float:
    """单个分组的 lift = group_bad_rate / overall_bad_rate."""
    if overall_bad_rate <= 0 or group_bad_rate < 0:
        return np.nan
    return round(group_bad_rate / overall_bad_rate, 6)


def _format_value_pair(val, cnt, lift):
    """格式化单个 值:数量:Lift 对"""
    return "%s : %d : %.2f" % (str(val), int(cnt), float(lift))


def _join_pairs(pairs_list, max_items=10):
    """将多个 值:数量:Lift 用 || 拼接"""
    return " || ".join(pairs_list[:max_items])


# ============================================================
# 数值型变量：null lift + 10分箱 lift
# ============================================================

def _stat_num_cross_y(col_series, y_series, overall_bad_rate, eda_cfg):
    """
    对单个数值型变量计算：
      null_lift, bin1_lift ~ bin10_lift

    Returns dict with keys:
      variable, type, null_lift, bin1_lift~bin10_lift,
      null_top5 (placeholder), bot5 (placeholder)
    """
    total = len(col_series)
    is_null = col_series.isna()
    null_cnt = int(is_null.sum())

    # -- null lift --
    if null_cnt > 0:
        null_bad_rate = y_series[is_null].mean()
        null_lift_val = _lift_from_bad_rate(null_bad_rate, overall_bad_rate)
    else:
        null_lift_val = np.nan

    # -- 有值部分 10 分箱 lift --
    clean_mask = ~is_null
    clean_y = y_series[clean_mask]
    clean_col = col_series[clean_mask]

    bins = [np.nan] * 10
    bin_edges_info = []
    if len(clean_col) > 0:
        try:
            # 用 pandas qcut 做等频分箱，容错处理
            unique_vals = clean_col.nunique()
            if unique_vals <= 10:
                # 唯一值很少，直接用 value_counts 分组
                combined_cv = pd.concat([clean_col, clean_y], axis=1)
                combined_cv.columns = ["x_val", "y_val"]
                grouped = combined_cv.groupby("x_val")["y_val"].mean()
                # 扩展为 10 个 bin（不够的用 NaN）
                for i, (v, br) in enumerate(sorted(grouped.items(), key=lambda x: float(x[0]))):
                    if i < 10:
                        bins[i] = _lift_from_bad_rate(float(br), overall_bad_rate)
                        cnt_i = int((clean_col == v).sum())
                        bin_edges_info.append({
                            "bin_idx": i + 1,
                            "bin_label": str(v),
                            "sample_count": cnt_i,
                        })
            else:
                # 等频分箱（qcut）——解决极端偏态分布导致等宽分箱大量bin为空的问题
                min_v2, max_v2 = float(clean_col.min()), float(clean_col.max())
                if min_v2 == max_v2:
                    bins[0] = _lift_from_bad_rate(float(clean_y.mean()), overall_bad_rate)
                else:
                    try:
                        col_binned = pd.qcut(
                            clean_col, q=10, duplicates="drop"
                        )
                        # pandas 1.3.x compat: use .unique() instead of .categories
                        unique_cats = sorted(col_binned.unique(), key=lambda x: float(x.left))
                        actual_bins = len(unique_cats)
                        for b_idx, cat_label in enumerate(unique_cats):
                            mask_b = col_binned == cat_label
                            cnt_i = int(mask_b.sum())
                            bin_edges_info.append({
                                "bin_idx": b_idx + 1,
                                "bin_label": str(cat_label),
                                "sample_count": cnt_i,
                            })
                            if b_idx < 10 and mask_b.sum() > 0:
                                br = float(clean_y[mask_b].mean())
                                bins[b_idx] = _lift_from_bad_rate(br, overall_bad_rate)
                    except Exception:
                        # fallback to equal-width if qcut also fails
                        bin_edges = np.linspace(min_v2, max_v2, 11)
                        col_binned = pd.cut(
                            clean_col, bins=bin_edges,
                            include_lowest=True, labels=["bin%d" % i for i in range(1, 11)]
                        )
                        for b_idx in range(10):
                            b_name = "bin%d" % (b_idx + 1)
                            mask_b = col_binned == b_name
                            cnt_i = int(mask_b.sum())
                            low_e = bin_edges[b_idx]
                            high_e = bin_edges[b_idx + 1]
                            bin_edges_info.append({
                                "bin_idx": b_idx + 1,
                                "bin_label": f"[{low_e:.2f},{high_e:.2f}]",
                                "sample_count": cnt_i,
                            })
                            if mask_b.sum() > 0:
                                br = float(clean_y[mask_b].mean())
                                bins[b_idx] = _lift_from_bad_rate(br, overall_bad_rate)
        except Exception:
            pass  # 全留 NaN

    result = {
        "variable": col_series.name,
        "type": "num",
        "null_cnt": null_cnt,
        "null_lift": null_lift_val if not np.isnan(null_lift_val) else None,
    }
    for i, b in enumerate(bins):
        result["bin%d_lift" % (i + 1)] = round(b, 6) if not np.isnan(b) else None
    result["_bin_details"] = bin_edges_info
    return result


# ============================================================
# 字符型变量：top10 lift + top5/bot5 lift（含阈值过滤）
# ============================================================

def _stat_cat_cross_y(col_series, y_series, overall_bad_rate, eda_cfg):
    """
    对单个字符型变量计算：
      - top10_lift: 频数最高的 10 个取值的 lift，按 count 排序
      - threshold_filtered_top5 / bot5: 超过阈值占比的取值，按 lift 排序

    Returns dict
    """
    var_cfg = eda_cfg.get("variable_eda", {})
    min_pct = var_cfg.get("min_value_pct_threshold", 0.02)  # 默认 2%
    total = len(col_series)
    min_count = max(1, int(total * min_pct))

    # value_counts + 对应标签坏率
    vc = col_series.value_counts()
    combined = pd.concat([col_series, y_series], axis=1).groupby(col_series)[y_series.name].agg(["mean", "count"])
    combined.columns = ["bad_rate", "count"]

    # -- 全部取值的 lift --
    all_lifts = []
    for val, row in combined.iterrows():
        br = float(row["bad_rate"])
        cnt = int(row["count"])
        lft = _lift_from_bad_rate(br, overall_bad_rate)
        all_lifts.append({"val": val, "cnt": cnt, "lift": lft})

    # -- top10 按频数排序 --
    top10_sorted = sorted(all_lifts, key=lambda x: x["cnt"], reverse=True)[:10]
    top10_pairs = [_format_value_pair(r["val"], r["cnt"], r["lift"]) for r in top10_sorted]

    # -- 阈值过滤后按 lift 排序的 top5 / bot5 --
    threshold_filtered = [r for r in all_lifts if r["cnt"] >= min_count]
    lift_sorted = sorted(threshold_filtered, key=lambda x: float(x["lift"]) if not np.isnan(x["lift"]) else -1e9, reverse=True)

    top5_pairs = [_format_value_pair(r["val"], r["cnt"], r["lift"]) for r in lift_sorted[:5]]
    bot5_pairs = [_format_value_pair(r["val"], r["cnt"], r["lift"]) for r in lift_sorted[-5:][::-1] if lift_sorted]

    # -- IV 近似值（WoE + IV）--
    iv_val = _compute_iv_simple(col_series, y_series)

    return {
        "variable": col_series.name,
        "type": "char",
        "unique_cnt": int(vc.shape[0]),
        "top10_by_count_lift": _join_pairs(top10_pairs, max_items=10),
        "threshold_top5_lift": _join_pairs(top5_pairs, max_items=5),
        "threshold_bot5_lift": _join_pairs(bot5_pairs, max_items=5),
        "iv_approx": round(iv_val, 6) if iv_val is not None else None,
    }


# ============================================================
# 简单 IV 计算
# ============================================================

def _compute_iv_simple(col_series, y_series):
    """
    简化的 IV 计算：对字符型变量按取值分组算 WoE/IV。
    数值型如果有大量唯一值则返回 None。
    """
    combined = pd.concat([col_series, y_series], axis=1).dropna()
    if len(combined) == 0:
        return None

    total_good = (combined[y_series.name] == 0).sum()
    total_bad = (combined[y_series.name] == 1).sum()

    if total_good == 0 or total_bad == 0:
        return None

    group_stats = combined.groupby(combined[col_series.name])[y_series.name].agg(["count", "sum"])
    group_stats["bad"] = group_stats["sum"]
    group_stats["good"] = group_stats["count"] - group_stats["bad"]

    # 平滑
    group_stats["good"] += 1
    group_stats["bad"] += 1
    total_good += group_stats.shape[0]
    total_bad += group_stats.shape[0]

    group_stats["pct_good"] = group_stats["good"] / total_good
    group_stats["pct_bad"] = group_stats["bad"] / total_bad

    woe = np.log(group_stats["pct_bad"] / group_stats["pct_good"])
    iv = (group_stats["pct_bad"] - group_stats["pct_good"]) * woe

    return float(iv.sum())


# ============================================================
# 统一入口
# ============================================================

def run_x_y_cross_eda(df, label_col, feature_cols, eda_cfg):
    """
    对每个特征 X 交叉 Y 计算 lift / IV 等指标。

    Parameters
    ----------
    df : pd.DataFrame     数据（包含 label_col）
    label_col : str       标签列名（0/1）
    feature_cols : list   需要分析的特征列表
    eda_cfg : dict        step1_eda 配置

    Returns
    -------
    result_df : DataFrame  每行一个变量，包含 lift / IV 等统计量
    bin_details : list     数值型变量的分箱明细（边界+样本量）
    """
    print("=" * 60)
    print("[X-Y Cross EDA] Starting ...")
    print("=" * 60)

    y = df[label_col].astype(float)
    overall_bad_rate = float(y.mean())
    print("  Overall bad rate: %.4f" % overall_bad_rate)

    if not feature_cols:
        exclude = {label_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    print("  Variables to analyze: %d" % len(feature_cols))

    rows = []
    all_bin_details = []

    # -- 区分 num / char --
    num_cols = []
    cat_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        try:
            is_num = np.issubdtype(df[col].dtype, np.number)
        except (TypeError, AttributeError):
            # CategoricalDtype incompatible with old numpy.isSubdtype
            is_num = pd.api.types.is_numeric_dtype(df[col])
        if is_num:
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # -- 数值型 --
    print("  Computing %d numeric vars ..." % len(num_cols))
    for col in tqdm(num_cols, desc="num cross-y"):
        s = df[col]
        try:
            row = _stat_num_cross_y(s, y, overall_bad_rate, eda_cfg)
            # 收集分箱明细
            details = row.pop("_bin_details", [])
            for bd in details:
                bd["variable"] = col
                bd["type"] = "num"
            all_bin_details.extend(details)
            rows.append(row)
        except Exception as e:
            # 失败时留一行空记录
            rows.append({
                "variable": col, "type": "num",
                "null_lift": None,
            })

    # -- 字符型 --
    print("  Computing %d categorical vars ..." % len(cat_cols))
    for col in tqdm(cat_cols, desc="cat cross-y"):
        s = df[col]
        try:
            row = _stat_cat_cross_y(s, y, overall_bad_rate, eda_cfg)
            rows.append(row)
        except Exception as e:
            rows.append({
                "variable": col, "type": "char",
                "top10_by_count_lift": None,
                "threshold_top5_lift": None,
                "threshold_bot5_lift": None,
                "iv_approx": None,
            })

    result_df = pd.DataFrame(rows)
    print("  [X-Y Cross EDA] Done. %d variables processed." % len(result_df))
    return result_df, all_bin_details
