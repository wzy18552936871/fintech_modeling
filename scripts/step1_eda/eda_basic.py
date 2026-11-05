"""
基础 EDA 统计模块

输出三个维度的汇总：
  1. 全样本：tot_cnt, y_cnt, y_pct, x_var_cnt
  2. 分月  ：month, tot_cnt, y_cnt, y_pct, x_var_cnt
  3. 分Group: group, tot_cnt, y_cnt, y_pct, x_var_cnt

所有统计由 eda_config 控制（来自 result_thresholds.yaml → step1_eda）
"""
from __future__ import print_function, unicode_literals
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 内部辅助：条件判断 + 随机抽样
# ============================================================
def _maybe_sample(df, eda_cfg):
    """
    铁律一：基础 EDA 必须使用全量宽表，不再采样。
    """
    print("  [Basic EDA] Full data: %d vars, %d rows" % (len(df.columns), len(df)))
    return df, True


# ============================================================
# 1. 全样本统计
# ============================================================
def summary_all(df_ed, label_col, total_x_vars):
    """返回一行记录：tot_cnt, y_cnt, y_pct, x_var_cnt"""
    tot = len(df_ed)
    y_cnt = int(df_ed[label_col].sum()) if df_ed[label_col].dtype in (np.int64, np.float64, np.int32, float) else int((df_ed[label_col] == 1).sum())
    row = pd.Series({
        "tot_cnt": tot,
        "y_cnt": y_cnt,
        "y_pct": round(y_cnt / tot, 6) if tot > 0 else 0.0,
        "x_var_cnt": total_x_vars,
    })
    return pd.DataFrame([row])


# ============================================================
# 2. 分月统计
# ============================================================
def summary_by_month(df_ed, label_col, month_col, total_x_vars):
    """按月返回：month, tot_cnt, y_cnt, y_pct, x_var_cnt"""
    g = df_ed.groupby(month_col)
    rows = []
    for m, grp in g:
        tot = len(grp)
        y_cnt = int(grp[label_col].sum()) if grp[label_col].dtype in (np.int64, np.float64, np.int32, float) else int((grp[label_col] == 1).sum())
        rows.append({
            "month": m,
            "tot_cnt": tot,
            "y_cnt": y_cnt,
            "y_pct": round(y_cnt / tot, 6) if tot > 0 else 0.0,
            "x_var_cnt": total_x_vars,
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


# ============================================================
# 3. 分 Group 统计（group_cols 从 data_config.yaml → analysis_dims.group_by_cols）
# ============================================================
def summary_by_group(df_ed, label_col, group_cols, total_x_vars):
    """按每个分组列分别返回：{dim}, tot_cnt, y_cnt, y_pct, x_var_cnt"""
    frames = []
    for dim in group_cols:
        if dim not in df_ed.columns:
            continue
        g = df_ed.groupby(dim)
        rows = []
        for gv, grp in g:
            tot = len(grp)
            y_cnt = int(grp[label_col].sum()) if grp[label_col].dtype in (np.int64, np.float64, np.int32, float) else int((grp[label_col] == 1).sum())
            rows.append({
                "group_value": gv,
                "tot_cnt": tot,
                "y_cnt": y_cnt,
                "y_pct": round(y_cnt / tot, 6) if tot > 0 else 0.0,
                "x_var_cnt": total_x_vars,
            })
        if rows:
            rdf = pd.DataFrame(rows)
            rdf.insert(0, "group_dim", dim)
            frames.append(rdf)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# 统一入口
# ============================================================
def run_basic_eda(df, label_col, month_col, group_cols, eda_cfg):
    """
    Parameters
    ----------
    df : pd.DataFrame 原始数据
    label_col : str 标签列名
    month_col : str 月份列名
    group_cols : list[str] 分组维度列表
    eda_cfg : dict result_thresholds.yaml 中 step1_eda 配置

    Returns
    -------
    results : dict
        {'all': DataFrame, 'by_month': DataFrame, 'by_group': DataFrame}
    """
    print("=" * 60)
    print("[Basic EDA] Starting ...")
    print("=" * 60)

    # ---- 确保 month_col / label_col 在数据中 ----
    keep = [label_col]
    if month_col and month_col in df.columns:
        keep.append(month_col)
    for gc in group_cols:
        if gc in df.columns and gc not in keep:
            keep.append(gc)

    # ---- 计算真实特征数量（排除 label + month + group_cols）----
    exclude_cols = {label_col}
    if month_col:
        exclude_cols.add(month_col)
    for gc in group_cols:
        exclude_cols.add(gc)
    total_x_vars = len([c for c in df.columns if c not in exclude_cols])

    # ---- 条件判断 + 随机抽样 ----
    df_s, _ = _maybe_sample(df[keep], eda_cfg)

    r_all = summary_all(df_s, label_col, total_x_vars)
    print("  All-summary:\n" + str(r_all))

    r_month = summary_by_month(df_s, label_col, month_col, total_x_vars) if (month_col and month_col in df_s.columns) else pd.DataFrame()
    if not r_month.empty:
        print("  Month-summary shape: %s" % str(r_month.shape))

    r_group = summary_by_group(df_s, label_col, group_cols, total_x_vars) if group_cols else pd.DataFrame()
    if not r_group.empty:
        print("  Group-summary shape: %s" % str(r_group.shape))

    print("  [Basic EDA] Done.")
    return {"all": r_all, "by_month": r_month, "by_group": r_group}
