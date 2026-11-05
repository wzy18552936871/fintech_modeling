"""
相关性去重：对高度相关的特征对，仅保留 IV 较高的那个。
"""
from __future__ import annotations
import gc
import pandas as pd
import numpy as np


def corr_dedup(
    df,
    feature_list,
    iv_df=None,
    corr_threshold=0.80,  # 与 feature_select_thresholds.yaml 一致
):
    """
    计算特征间相关性矩阵，剔除冗余特征（相关系数 > threshold 时保留 IV 高者）。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据表。
    feature_list : list[str]
        待评估的特征列表。
    iv_df : pd.DataFrame, optional
        IV 汇总表，需含 'variable'/'feature' 和 'info_value' 列。
    corr_threshold : float
        相关性去重阈值。

    Returns
    -------
    list[str]
        去重后的特征列表。
    """
    available = [f for f in feature_list if f in df.columns]
    if len(available) < 2:
        return available

    # ---- 构建 IV 映射 ----
    iv_map = {}
    if iv_df is not None:
        feat_col = 'variable' if 'variable' in iv_df.columns else 'feature'
        for _, row in iv_df.iterrows():
            iv_map[str(row[feat_col])] = float(row['info_value'])

    # ---- 计算相关性矩阵 (分批,避免OOM) ----
    print("Computing correlation matrix for %d features..." % len(available))

    # 只取数值列做相关系数的样本 (子采样降低内存)
    df_corr = df[available].copy()
    for c in df_corr.columns:
        try:
            df_corr[c] = df_corr[c].astype(np.float32)
        except Exception:
            pass

    # 如果特征>200,子采样5000行加速
    if len(df_corr) > 5000:
        df_corr = df_corr.sample(5000, random_state=42)

    corr_matrix = df_corr.corr(min_periods=100).abs()
    del df_corr
    gc.collect()

    # ---- 遍历上三角，找高相关对 ----
    to_drop = set()
    n = len(available)
    for i in range(n):
        if available[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                f1, f2 = available[i], available[j]
                iv1 = iv_map.get(f1, 0.0)
                iv2 = iv_map.get(f2, 0.0)
                # 保留 IV 高的，IV 相同则剔除靠后的
                if iv1 >= iv2:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)
                break

    del corr_matrix
    gc.collect()

    result = [f for f in available if f not in to_drop]
    print("Correlation dedup (>= %.2f): dropped %d redundant features" % (corr_threshold, len(to_drop)))
    print("[OK] After dedup: %d features\n" % len(result))
    return result
