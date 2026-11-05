import gc
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2_contingency

#=========================== 定义特征PSI统计函数 ===================================
def chi_merge(data, col, target, max_bins=10):
    data = data[[col, target]].dropna().copy()
    data = data.sort_values(by=col)
    freq_table = data.groupby([col, target]).size().unstack(fill_value=0)

    while len(freq_table) > max_bins:
        chi_values = []
        for i in range(len(freq_table) - 1):
            table = freq_table.iloc[i:i + 2]
            try:
                chi2, _, _, _ = chi2_contingency(table)
            except ValueError:
                # 零频单元格，给一个很大的chi2避免合并这一对
                chi2 = 1e10
            chi_values.append(chi2)
        min_index = chi_values.index(min(chi_values))
        freq_table.iloc[min_index] = freq_table.iloc[min_index] + freq_table.iloc[min_index + 1]
        freq_table = freq_table.drop(freq_table.index[min_index + 1])

    bins = sorted(freq_table.index.tolist())
    bins = [-np.inf] + bins + [np.inf]
    bins = sorted(set(bins))
    return bins

def calculate_psi(
    base_df,
    compare_df,
    output_excel_path,
    feature_cols=None,
    bin_num=10,
    bin_method='equal_width',
    base_weight_col=None,
    compare_weight_col=None,
    target_col='target'  # bin_method为chi时，才需要赋值
):
    """
    计算多个特征的 PSI（Population Stability Index）并输出到 Excel，
    并将所有特征分箱明细合并写入同一个sheet

    参数：
    ----------
    base_df : pd.DataFrame，基准数据集
    compare_df : pd.DataFrame，比对数据集
    output_excel_path : str，结果输出路径
    feature_cols : list[str]，要统计的特征，默认全部数值型特征
    bin_num : int，分箱数量（默认 10）
    bin_method : str，分箱方式：'equal_width' | 'equal_freq' | 'chi'
    base_weight_col : str or None，基准数据的权重列
    compare_weight_col : str or None，比对数据的权重列
    target_col : str，用于卡方分箱的目标列
    """

    if feature_cols is None:
        feature_cols = [col for col in base_df.select_dtypes(include=np.number).columns]

    psi_list = []
    all_details = []

    # 使用 tqdm 进度条
    # 过滤非数值列，避免直方图报错
    numeric_feature_cols = []
    for c in feature_cols:
        if c in base_df.columns and pd.api.types.is_numeric_dtype(base_df[c]):
            numeric_feature_cols.append(c)
    print("Starting PSI calculation for %d features (numeric only, skipped non-numeric)..." % len(numeric_feature_cols))
    for col in tqdm(numeric_feature_cols, desc="计算PSI"):
        base_data = base_df[[col]].copy()
        compare_data = compare_df[[col]].copy()

        base_data['weight'] = base_df[base_weight_col] if (base_weight_col and base_weight_col in base_df.columns) else 1.0
        compare_data['weight'] = compare_df[compare_weight_col] if (compare_weight_col and compare_weight_col in compare_df.columns) else 1.0

        base_data = base_data.dropna(subset=[col])
        compare_data = compare_data.dropna(subset=[col])

        if base_data[col].nunique() <= 1:
            print("[WARN] Feature %s has constant values in base data, skipping" % col)
            continue

        # 分箱
        if bin_method == 'equal_width':
            bins_edges = np.histogram_bin_edges(base_data[col], bins=bin_num)
            bins_edges = np.unique(bins_edges)
        elif bin_method == 'equal_freq':
            try:
                base_data['bin'] = pd.qcut(base_data[col], q=bin_num, duplicates='drop')
                bins_edges = sorted(set(
                    [interval.left for interval in base_data['bin'].cat.categories] +
                    [base_data['bin'].cat.categories[-1].right]
                ))
            except Exception as e:
                print("[WARN] Feature %s equal-freq binning failed: %s" % (col, str(e)))
                continue
        elif bin_method == 'chi':
            bins_edges = chi_merge(base_df[[col, target_col]].dropna(), col, target_col, max_bins=bin_num)
            bins_edges = np.unique(bins_edges)
        else:
            raise ValueError("分箱方法必须是 'equal_width', 'equal_freq' 或 'chi'")

        # Execute bin mapping
        if bin_method != 'equal_freq':
            base_data['bin'] = pd.cut(base_data[col], bins=bins_edges, right=True, include_lowest=True)
            compare_data['bin'] = pd.cut(compare_data[col], bins=bins_edges, right=True, include_lowest=True)

        # Print distribution info
        print("\nFeature %s bin edges: %s" % (col, str(bins_edges)))
        print("Base dist:\n%s" % str(base_data['bin'].value_counts(dropna=False).to_dict()))
        print("Compare dist:\n%s" % str(compare_data['bin'].value_counts(dropna=False).to_dict()))

        # 计算分布（raw counts）
        base_dist = base_data.groupby('bin', observed=False)['weight'].sum()
        compare_dist = compare_data.groupby('bin', observed=False)['weight'].sum()

        # ---- Normalize to percentages (fix: avoid sum=0 -> 0/0 -> NaN -> PSI=0) ----
        total_base = base_dist.sum()
        total_compare = compare_dist.sum()

        epsilon = 1e-4
        if total_base == 0:
            total_base = epsilon * len(base_dist)
        if total_compare == 0:
            total_compare = epsilon * len(compare_dist)

        base_pct = (base_dist / total_base).fillna(epsilon).replace(0, epsilon)
        compare_pct = (compare_dist / total_compare).fillna(epsilon).replace(0, epsilon)

        # 对齐索引
        all_bins = sorted(set(base_pct.index).union(compare_pct.index))
        base_pct = base_pct.reindex(all_bins, fill_value=epsilon)
        compare_pct = compare_pct.reindex(all_bins, fill_value=epsilon)

        # PSI 计算
        psi_vals = (base_pct - compare_pct) * np.log(base_pct / compare_pct)
        total_psi = psi_vals.sum()

        print("Feature %s PSI value: %.6f\n" % (col, total_psi))

        psi_list.append({
            'feature': col,
            'psi': round(total_psi, 6)
        })

        detail_df = pd.DataFrame({
            'feature': col,
            'bin': [str(b) for b in all_bins],
            'base_dist': base_pct.values,
            'compare_dist': compare_pct.values,
            'psi_per_bin': psi_vals.values
        })
        all_details.append(detail_df)

    # Write results to Excel
    df_result = pd.DataFrame(psi_list)
    df_detail_all = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, sheet_name='PSI_Summary', index=False)
        df_detail_all.to_excel(writer, sheet_name='Bins_Details', index=False)

    print("[OK] PSI calculation done, saved to: %s" % output_excel_path)

    del all_details, psi_list, df_result, df_detail_all
    gc.collect()


#=========================== 复用工具函数 ===================================
def get_bins_auto(base_data, col, bin_num=10):
    """
    对单列数据做等宽分箱，返回 bins_edges。
    base_data: 包含目标列的 DataFrame
    col: 列名
    bin_num: 分箱数
    """
    bins_edges = np.histogram_bin_edges(base_data[col], bins=bin_num)
    bins_edges = np.unique(bins_edges)
    return bins_edges


def get_psi_detail(feature, bins_edges, base_values, compare_values, epsilon=1e-4):
    """
    根据分箱边界、两组数据计算 PSI 明细和总 PSI。

    返回: (detail_df, total_psi)
    """
    base_s = pd.Series(base_values)
    compare_s = pd.Series(compare_values)

    base_bin = pd.cut(base_s, bins=bins_edges, right=True, include_lowest=True)
    compare_bin = pd.cut(compare_s, bins=bins_edges, right=True, include_lowest=True)

    # 分布
    base_dist = base_bin.value_counts().sort_index()
    compare_dist = compare_bin.value_counts().sort_index()

    # Normalize
    total_base = base_dist.sum()
    total_compare = compare_dist.sum()
    if total_base == 0:
        total_base = epsilon * len(base_dist)
    if total_compare == 0:
        total_compare = epsilon * len(compare_dist)

    base_pct = (base_dist / total_base).fillna(epsilon).replace(0, epsilon)
    compare_pct = (compare_dist / total_compare).fillna(epsilon).replace(0, epsilon)

    # 对齐
    all_bins = sorted(set(base_pct.index).union(compare_pct.index))
    base_pct = base_pct.reindex(all_bins, fill_value=epsilon)
    compare_pct = compare_pct.reindex(all_bins, fill_value=epsilon)

    psi_vals = (base_pct - compare_pct) * np.log(base_pct / compare_pct)
    total_psi = float(psi_vals.sum())

    detail_df = pd.DataFrame({
        'feature': feature,
        'bin': [str(b) for b in all_bins],
        'base_dist': base_pct.values,
        'compare_dist': compare_pct.values,
        'psi_per_bin': psi_vals.values
    })

    return detail_df, total_psi


#=========================== 逐月 PSI 计算函数 ===================================
def calculate_monthly_psi(
    df: pd.DataFrame,
    month_col: str,
    features: list,
    bin_num: int = 10,
    output_excel_path: str = None
):
    """
    逐月 PSI 计算：base=第n-1月, compare=第n月（内存优化版）

    参数:
        df: 完整数据（全量、不分label）
        month_col: 月份列名（如 'app_date'，函数内部会取前6位转为YYYYMM）
        features: 特征列列表
        bin_num: 分箱数
        output_excel_path: 输出excel路径

    返回:
        psi_summary_df, bins_detail_df
    """
    if month_col not in df.columns:
        print(f"[WARN] month_col '{month_col}' not found in df columns. Skipping monthly PSI.")
        return pd.DataFrame(), pd.DataFrame()

    # 用 numpy array 代替 DataFrame 切片，大幅减少内存
    month_arr = np.array(df[month_col].astype(str).str[:6], dtype='U6')
    months = sorted(np.unique(month_arr))
    if len(months) < 2:
        print("[WARN] 月份数不足2，无法计算逐月PSI")
        return pd.DataFrame(), pd.DataFrame()

    # 预计算每个月份的 mask（bool array，比DataFrame切片省内存）
    month_masks = {m: month_arr == m for m in months}

    psi_list = []

    numeric_features = [f for f in features if df[f].dtype in ['int64','float64','int32','float32']]
    print(f"Starting monthly PSI for {len(numeric_features)} numeric features across {len(months)} months...")

    feat_idx = 0
    for feature in tqdm(numeric_features, desc="Monthly PSI"):
        try:
            feat_values = df[feature].values  # numpy array, no copy

            # 用首月数据做全局分箱
            first_month = months[0]
            base_for_bins = feat_values[month_masks[first_month]]
            base_for_bins = base_for_bins[~np.isnan(base_for_bins)] if np.issubdtype(feat_values.dtype, np.floating) else base_for_bins
            if len(base_for_bins) == 0:
                continue

            try:
                bins = np.unique(np.histogram_bin_edges(base_for_bins, bins=bin_num))
            except Exception:
                continue

            # 逐对比较：n-1月 vs n月
            for i in range(1, len(months)):
                base_month = months[i-1]
                compare_month = months[i]

                base_data = feat_values[month_masks[base_month]]
                compare_data = feat_values[month_masks[compare_month]]

                if np.issubdtype(feat_values.dtype, np.floating):
                    base_data = base_data[~np.isnan(base_data)]
                    compare_data = compare_data[~np.isnan(compare_data)]

                if len(base_data) == 0 or len(compare_data) == 0:
                    continue

                try:
                    _, psi_val = get_psi_detail(
                        feature, bins, base_data, compare_data
                    )
                except Exception:
                    continue

                psi_list.append({
                    'feature': feature,
                    'base_month': base_month,
                    'compare_month': compare_month,
                    'psi_value': round(psi_val, 6),
                })
        except Exception:
            continue

        feat_idx += 1
        # 每25个特征 gc 一次（更频繁）
        if feat_idx % 25 == 0:
            gc.collect()

    df_result = pd.DataFrame(psi_list) if psi_list else pd.DataFrame()

    if output_excel_path:
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            if not df_result.empty and 'compare_month' in df_result.columns:
                pivot_df = df_result.pivot_table(
                    index='feature',
                    columns='compare_month',
                    values='psi_value',
                    aggfunc='mean'
                )
                month_cols = sorted(pivot_df.columns)
                pivot_df = pivot_df[month_cols]
                pivot_df.to_excel(writer, sheet_name='Monthly_PSI')
            else:
                df_result.to_excel(writer, sheet_name='Monthly_PSI', index=False)
        print(f"[OK] Monthly PSI saved to: {output_excel_path}")

    result = df_result
    del psi_list, df_result, month_masks
    gc.collect()

    return result, pd.DataFrame()

