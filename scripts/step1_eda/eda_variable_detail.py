"""
变量详细 EDA 统计模块

数值型 (num) 输出列：
  variable, type(num), total_cnt, null_cnt, null_rate,
  special_value_cols (如 val_0_cnt, val_0_rate, val_-9_cnt, ...),
  min, pct1, pct10, pct25, pct50, pct75, pct90, pct99, max

字符型 (char) 输出列：
  variable, type(char), total_cnt, null_cnt, null_rate,
  special_value_cols,
  top1, top2, ..., top5    （格式 "值: cnt"）
  bot1, bot2, ..., bot5

如果 compute_unique = True，额外追加列：
  num: unique_cnt, most_frequent_val  (格式 "值: cnt")
  char: unique_cnt, most_frequent_val
  num → char 转换列表（unique < threshold 的变量转 char 统计）

PSI / Corr 由 caller 控制开关，这里只提供数据基础。
"""
from __future__ import print_function, unicode_literals
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 内部辅助：条件判断 + 随机抽样
# ============================================================
def _maybe_sample(df, eda_cfg):
    """
    按条件决定是否全量跑 EDA；不满足条件时做简单随机抽样（非分层）。  

    规则（全部可配置，来自 step1_eda.sampling）：
      - var_num < max_full_features AND 样本量 <= max_full_rows  → 全量
      - 否则 → 随机抽取 sample_size 条
    """
    sampling = eda_cfg.get("sampling", {})
    max_features = sampling.get("max_full_features", 500)
    max_rows = sampling.get("max_full_rows", 200000)
    sample_n = sampling.get("sample_size", 100000)
    seed = sampling.get("random_seed", 42)

    x_count = len(df.columns)
    row_count = len(df)

    if (x_count < max_features) and (row_count <= max_rows):
        print("  [VarEDA] Full data: %d vars, %d rows" % (x_count, row_count))
        return df, True
    else:
        # 简单随机抽样（非分层）
        n = min(sample_n, row_count)
        df_s = df.sample(n=n, replace=False, random_state=seed)
        print("  [VarEDA] Random sample: %d / %d rows (%d vars)" % (n, row_count, x_count))
        return df_s, False


# ============================================================
# 数值型变量统计
# ============================================================
def _stat_numeric(col_series, special_vals, do_unique):
    """单列 num 变量的统计，返回 dict"""
    total = len(col_series)
    s_null = col_series.isna().sum()

    r = {
        "total_cnt": total,
        "null_cnt": int(s_null),
        "null_rate": round(float(s_null) / total, 6) if total > 0 else 0.0,
    }

    # 特殊值 cnt
    for sv in special_vals:
        v_key = "val_%s_cnt" % str(sv).replace("-", "minus")
        r[v_key] = int((col_series == sv).sum())

    clean = col_series.dropna()
    if len(clean) == 0:
        # 全缺失，分位数填 NaN
        for qk in ("min", "pct1", "pct10", "pct25", "pct50", "pct75", "pct90", "pct99", "max"):
            r[qk] = np.nan
    else:
        r["min"] = float(clean.min())
        r["pct1"]  = float(np.percentile(clean.values, 1))
        r["pct10"] = float(np.percentile(clean.values, 10))
        r["pct25"] = float(np.percentile(clean.values, 25))
        r["pct50"] = float(np.percentile(clean.values, 50))
        r["pct75"] = float(np.percentile(clean.values, 75))
        r["pct90"] = float(np.percentile(clean.values, 90))
        r["pct99"] = float(np.percentile(clean.values, 99))
        r["max"]   = float(clean.max())

    if do_unique:
        vc = col_series.value_counts(dropna=False)
        r["unique_cnt"] = len(vc)
        top_val = vc.index[0]
        top_cnt = int(vc.iloc[0])
        r["most_frequent_val"] = "%s : %d" % (str(top_val), top_cnt)

    return r


# ============================================================
# 字符型变量统计
# ============================================================
def _stat_categorical(col_series, special_vals, do_unique):
    """单列 char 变量的统计，返回 dict"""
    total = len(col_series)
    s_null = col_series.isna().sum()

    r = {
        "total_cnt": total,
        "null_cnt": int(s_null),
        "null_rate": round(float(s_null) / total, 6) if total > 0 else 0.0,
    }

    for sv in special_vals:
        v_key = "val_%s_cnt" % str(sv).replace("-", "minus")
        r[v_key] = int((col_series == sv).sum())

    vc = col_series.value_counts(dropna=False)

    # top 5 / bot 5
    n_top = min(5, len(vc))
    for i in range(n_top):
        v = str(vc.index[i])
        c = int(vc.iloc[i])
        r["top%d" % (i + 1)] = "%s : %d" % (v, c)
    for i in range(n_top, 5):
        r["top%d" % (i + 1)] = ""

    n_bot = min(5, len(vc))
    vc_rev = vc.iloc[::-1]
    for i in range(n_bot):
        v = str(vc_rev.index[i])
        c = int(vc_rev.iloc[i])
        r["bot%d" % (i + 1)] = "%s : %d" % (v, c)
    for i in range(n_bot, 5):
        r["bot%d" % (i + 1)] = ""

    if do_unique:
        r["unique_cnt"] = len(vc)
        top_val = vc.index[0]
        top_cnt = int(vc.iloc[0])
        r["most_frequent_val"] = "%s : %d" % (str(top_val), top_cnt)

    return r


# ============================================================
# 统一入口
# ============================================================
def run_variable_eda(df, label_col, month_col, eda_cfg, feature_cols=None):
    """
    Parameters
    ----------
    df : pd.DataFrame  原始数据（包含所有变量）
    label_col : str    标签列名
    month_col : str    月份列名
    eda_cfg : dict     step1_eda 配置
    feature_cols : list, optional  特征列白名单（None=自动推断）

    Returns
    -------
    results : dict
        {
            'numeric': DataFrame,          # 数值型变量统计
            'categorical': DataFrame,      # 字符型变量统计
            'num_to_char_vars': list[str], # unique<阈值被转char的变量（空列表表示无）
        }
    """
    var_cfg = eda_cfg.get("variable_eda", {})
    special_vals = var_cfg.get("special_values", [0])       # 默认 [0]
    do_unique = var_cfg.get("compute_unique", False)
    unique_thresh = var_cfg.get("unique_to_char_threshold", 20)

    print("=" * 60)
    print("[Variable EDA] Starting ...")
    print("=" * 60)

    # ---- 排除 label + month，只统计 X 变量 ----
    exclude = {label_col}
    if month_col and month_col in df.columns:
        exclude.add(month_col)

    if feature_cols is not None:
        x_cols = [c for c in feature_cols if c in df.columns and c not in exclude]
    else:
        x_cols = [c for c in df.columns if c not in exclude]

    if not x_cols:
        print("  No feature columns to analyze.")
        return {"numeric": pd.DataFrame(), "categorical": pd.DataFrame(), "num_to_char_vars": []}

    # ---- 条件判断 + 随机抽样（仅对 X 列） ----
    df_x = _maybe_sample(df[x_cols], eda_cfg)[0]

    # ---- 自动分类 num / char ----
    num_cols_raw = df_x.select_dtypes(include=np.number).columns.tolist()
    char_cols = [c for c in x_cols if c not in num_cols_raw]

    # ---- unique 检查 → num→char 转换（仅在 do_unique=True 时） ----
    converted_vars = []
    if do_unique:
        print("  Checking unique counts for %d numeric vars ..." % len(num_cols_raw))
        num_cols_final = []
        for c in tqdm(num_cols_raw, desc="unique check"):
            u = df_x[c].nunique()
            if u < unique_thresh:
                converted_vars.append((c, u))
                char_cols.append(c)
            else:
                num_cols_final.append(c)
        num_cols = num_cols_final
    else:
        num_cols = num_cols_raw

    print("  Numeric vars: %d, Categorical vars: %d" % (len(num_cols), len(char_cols)))

    # ---- 统计每个变量 ----
    # 确定共同列名（特殊值动态生成）
    extra_col_keys = []
    for sv in special_vals:
        extra_col_keys.append("val_%s_cnt" % str(sv).replace("-", "minus"))

    common_base = ["variable", "type", "total_cnt", "null_cnt", "null_rate"] + list(extra_col_keys)

    # 数值型
    print("  Computing numeric vars ...")
    num_rows = []
    for c in tqdm(num_cols, desc="num vars"):
        row = {"variable": c, "type": "num"}
        row.update(_stat_numeric(df_x[c], special_vals, do_unique))
        num_rows.append(row)
    df_num = pd.DataFrame(num_rows)

    # 字符型（含被转char的）
    print("  Computing categorical vars ...")
    cat_rows = []
    for c in tqdm(char_cols, desc="cat vars"):
        row = {"variable": c, "type": "char"}
        row.update(_stat_categorical(df_x[c], special_vals, do_unique))
        cat_rows.append(row)
    df_cat = pd.DataFrame(cat_rows)

    print("  [Variable EDA] Done.")
    return {
        "numeric": df_num,
        "categorical": df_cat,
        "num_to_char_vars": converted_vars,
    }


# ============================================================
# PSI 统计（可选，由开关控制）
# ============================================================
def run_psi_all(df, label_col, month_col, base_df, eda_cfg, psi_bin_num=10):
    """
    计算所有变量逐月 PSI。

    默认逻辑（无用户特殊指定时）：
      - 第一个月作为基准月，不计算PSI
      - 第二个月开始，以上一个月为base_df，当月为compare_df

    Parameters
    ----------
    df : pd.DataFrame        全量数据
    label_col : str          标签列（排除）
    month_col : str          月份列
    base_df : pd.DataFrame   保留参数名（兼容旧接口），逐月模式下不使用
    eda_cfg : dict           配置
    psi_bin_num : int        分箱数

    Returns
    -------
    psi_df : DataFrame  含 variable, month, prev_month, psi_value 四列
    """
    print("=" * 60)
    print("[PSI All Variables Monthly] Starting ...")
    print("=" * 60)

    exclude = {label_col}
    if month_col and month_col in df.columns:
        exclude.add(month_col)
    x_cols = [c for c in df.columns if c not in exclude]

    # 仅数值型变量算 PSI
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c in x_cols]

    if not month_col or month_col not in df.columns:
        print("  No month column, skipping monthly PSI.")
        return pd.DataFrame()

    # 获取有序月份列表
    sorted_months = sorted(df[month_col].dropna().unique())
    print(f"  Months found: {sorted_months}")
    print(f"  Numeric variables for PSI: {len(num_cols)}")

    if len(sorted_months) < 2:
        print("  Need at least 2 months for monthly PSI, skipping.")
        return pd.DataFrame()

    rows = []
    # 逐月比较：month[i] vs month[i-1]
    prev_month_data = None
    prev_month_name = None

    for month_idx, m in enumerate(sorted_months):
        month_df = df[df[month_col] == m]

        if month_idx == 0:
            # 第一个月作为基准，不计算PSI，只缓存数据
            print(f"  Month {m}: Setting as baseline (no PSI computed)")
            prev_month_data = month_df.copy()
            prev_month_name = m
            continue

        # 从第二个月开始，以上一月为base，当月为compare
        print(f"  Month {m}: Comparing vs previous month {prev_month_name} ...")
        curr_month_data = month_df

        for col in tqdm(num_cols, desc=f"PSI {m}"):
            if col not in prev_month_data.columns or col not in curr_month_data.columns:
                continue

            base_s = prev_month_data[col].dropna()
            curr_s = curr_month_data[col].dropna()

            if len(base_s) == 0 or len(curr_s) == 0:
                continue

            try:
                bins = _equal_width_bins(base_s, psi_bin_num)
                psi_val = _psi_by_bins(base_s, curr_s, bins)
                rows.append({
                    "variable": col,
                    "month": m,
                    "prev_month": prev_month_name,
                    "psi_value": round(psi_val, 6),
                })
            except Exception:
                # 某个变量计算失败，跳过
                continue

        # 更新基准为当前月
        prev_month_data = curr_month_data
        prev_month_name = m

    psi_df = pd.DataFrame(rows)
    print(f"  [PSI] Done, {len(psi_df)} records")
    return psi_df


def _equal_width_bins(series, n_bins):
    """按等宽生成 bin edges"""
    vals = series.values.astype(float)
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


def _psi_by_bins(base_s, curr_s, bins):
    """基于相同分箱边界计算 PSI"""
    base_hist, _ = np.histogram(base_s.values, bins=bins)
    curr_hist, _ = np.histogram(curr_s.values, bins=bins)

    # 加 1e-6 平滑
    base_pct = (base_hist + 1e-6) / (len(base_hist) + 1e-6 * len(base_hist))
    curr_pct = (curr_hist + 1e-6) / (len(curr_hist) + 1e-6 * len(curr_hist))

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


# ============================================================
# 相关性矩阵（可选，由开关控制）
# ============================================================
def run_correlation(df, label_col, month_col, eda_cfg):
    """计算数值型变量之间的 Pearson 相关系数矩阵"""
    print("=" * 60)
    print("[Correlation Matrix] Starting ...")
    print("=" * 60)

    exclude = {label_col}
    if month_col and month_col in df.columns:
        exclude.add(month_col)
    x_cols = [c for c in df.columns if c not in exclude]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c in x_cols]

    # 相关矩阵可能非常大，先抽样
    df_s, _ = _maybe_sample(df[num_cols], eda_cfg)

    corr_df = df_s.corr(method="pearson")
    print("  [Correlation] Done, shape: %s" % str(corr_df.shape))
    return corr_df
