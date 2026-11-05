# """
# 硬性过滤：根据缺失率、PSI、IV 等硬指标剔除劣质特征。
# """
# from __future__ import annotations
# import pandas as pd
# import numpy as np
#
#
# def hard_filter(
#     feature_list,
#     missing_rate_df=None,
#     psi_df=None,
#     iv_df=None,
#     missing_rate_max=0.85,
#     psi_threshold=None,
#     iv_min=0.001,
# ):
#     """
#     根据缺失率、PSI、IV 三项硬指标过滤特征。
#
#     Parameters
#     ----------
#     feature_list : list[str]
#         待筛选的特征列表。
#     missing_rate_df : pd.DataFrame, optional
#         缺失率表，需含 'feature' 和 'missing_rate' 列。
#     psi_df : pd.DataFrame, optional
#         PSI 汇总表，需含 'feature' 和 'psi' 列。
#     iv_df : pd.DataFrame, optional
#         IV 汇总表，需含 'variable'/'feature' 和 'info_value' 列。
#     missing_rate_max : float
#         缺失率上限，超过则剔除。
#     psi_threshold : float or None
#         PSI 上限，超过则剔除；若为 None 则使用 PSI 85 分位点动态计算。
#     iv_min : float
#         IV 下限，低于此值视为弱特征，剔除。
#
#     Returns
#     -------
#     list[str]
#         通过硬性过滤的特征列表。
#     """
#     passed = set(feature_list)
#
#     # ---- 缺失率过滤 ----
#     if missing_rate_df is not None and len(passed) > 0:
#         mr = missing_rate_df.copy()
#         feat_col = 'feature' if 'feature' in mr.columns else 'variable'
#         max_mr = mr.groupby(feat_col)['missing_rate'].max().reset_index()
#         failed = set(max_mr[max_mr['missing_rate'] > missing_rate_max][feat_col])
#         failed &= passed
#         print("[emoji] 缺失率过滤 (> %.2f): 剔除 %d 个特征" % (missing_rate_max, len(failed)))
#         passed -= failed
#
#     # ---- PSI 过滤 ----
#     if psi_df is not None and len(passed) > 0:
#         psi = psi_df.copy()
#         feat_col = 'feature' if 'feature' in psi.columns else 'variable'
#         if psi_threshold is None:
#             psi_threshold = float(psi['psi'].quantile(0.85))
#             print("[emoji] PSI 85 分位点动态阈值: %.6f" % psi_threshold)
#         failed = set(psi[psi['psi'] > psi_threshold][feat_col])
#         failed &= passed
#         print("[emoji] PSI 过滤 (> %.6f): 剔除 %d 个特征" % (psi_threshold, len(failed)))
#         passed -= failed
#
#     # ---- IV 过滤 ----
#     if iv_df is not None and len(passed) > 0:
#         iv = iv_df.copy()
#         feat_col = 'variable' if 'variable' in iv.columns else 'feature'
#         iv[feat_col] = iv[feat_col].astype(str)
#         failed = set(iv[iv['info_value'] < iv_min][feat_col])
#         failed &= passed
#         print("[emoji] IV 过滤 (< %.2f): 剔除 %d 个特征" % (iv_min, len(failed)))
#         passed -= failed
#
#     result = sorted(passed)
#     print("\n[OK] 硬性过滤完成: %d → %d 个特征\n" % (len(feature_list), len(result)))
#     return result
import pandas as pd
import numpy as np


def hard_filter(
        feature_list,
        missing_rate_df=None,
        psi_df=None,
        iv_df=None,
        missing_rate_max=0.85,
        psi_threshold=None,
        iv_min=0.001,
):
    """
    合并缺失率、PSI、IV 三项硬指标，统一过滤特征。
    """
    # 1. 初始化基础 DataFrame
    merged_df = pd.DataFrame({'feature': feature_list})

    # 2. 依次合并缺失率、PSI、IV 数据
    if missing_rate_df is not None:
        feat_col = 'feature' if 'feature' in missing_rate_df.columns else 'variable'
        # 取最大值以防同一特征有多条记录
        mr = missing_rate_df.groupby(feat_col)['missing_rate'].max().reset_index()
        mr.rename(columns={feat_col: 'feature'}, inplace=True)
        merged_df = merged_df.merge(mr, on='feature', how='left')

    if psi_df is not None:
        feat_col = 'feature' if 'feature' in psi_df.columns else 'variable'
        psi = psi_df.groupby(feat_col)['psi'].max().reset_index()
        psi.rename(columns={feat_col: 'feature'}, inplace=True)
        merged_df = merged_df.merge(psi, on='feature', how='left')

    if iv_df is not None:
        feat_col = 'variable' if 'variable' in iv_df.columns else 'feature'
        iv = iv_df.groupby(feat_col)['info_value'].max().reset_index()
        iv.rename(columns={feat_col: 'feature'}, inplace=True)
        merged_df = merged_df.merge(iv, on='feature', how='left')

    # 3. 动态计算 PSI 阈值（如果未指定）
    if psi_threshold is None and 'psi' in merged_df.columns:
        psi_threshold = float(merged_df['psi'].quantile(0.85))
        print("[emoji] PSI 85 分位点动态阈值: %.6f" % psi_threshold)

    # 4. 统一应用过滤条件（利用 Pandas 向量化布尔索引）
    keep_mask = pd.Series(True, index=merged_df.index)

    if 'missing_rate' in merged_df.columns:
        mask_mr = merged_df['missing_rate'] <= missing_rate_max
        print("[emoji] 缺失率过滤 (> %.2f): 剔除 %d 个特征" % (missing_rate_max, (~mask_mr).sum()))
        keep_mask &= mask_mr

    if 'psi' in merged_df.columns and psi_threshold is not None:
        mask_psi = merged_df['psi'] <= psi_threshold
        print("[emoji] PSI 过滤 (> %.6f): 剔除 %d 个特征" % (psi_threshold, (~mask_psi).sum()))
        keep_mask &= mask_psi

    if 'info_value' in merged_df.columns:
        mask_iv = merged_df['info_value'] >= iv_min
        print("[emoji] IV 过滤 (< %.2f): 剔除 %d 个特征" % (iv_min, (~mask_iv).sum()))
        keep_mask &= mask_iv

    # 5. 提取通过筛选的特征并返回
    result = merged_df.loc[keep_mask, 'feature'].tolist()
    print("\n[OK] 硬性过滤完成: %d → %d 个特征\n" % (len(feature_list), len(result)))

    return result