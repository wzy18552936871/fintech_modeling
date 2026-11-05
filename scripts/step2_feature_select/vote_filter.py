"""
投票筛选：基于多个条件对每个特征打分投票，通过票数门槛决定是否保留。

核心逻辑：
  - 缺失率 (missing_rate): 缺失率 <= 阈值 → 通过
  - IV (info_value):       IV >= 阈值 → 通过
  - PSI:                   PSI <= 阈值 → 通过
  - 相关性 (corr):         不与其他高相关特征冲突 → 通过
  - 若某项 EDA 结果不存在（如 psi/corr），该条件默认全部通过
  - vote_pass_count = N：每个特征需 >= N 票才保留
  - var_keep_list: 投票不满足仍强制保留
  - var_drop_list: 投票满足仍强制删除
"""
from __future__ import annotations
import gc
import pandas as pd
import numpy as np


def vote_filter(
    feature_list,
    missing_rate_df=None,
    psi_df=None,
    iv_df=None,
    corr_matrix_df=None,
    missing_rate_max=0.85,
    psi_threshold=None,
    iv_min=0.001,
    corr_threshold=0.80,
    vote_pass_count=3,
    var_keep_list=None,
    var_drop_list=None,
):
    """
    投票筛选：每个特征在多个条件上获得"通过/不通过"投票，达到门槛即保留。

    Parameters
    ----------
    feature_list : list[str]
        待筛选的特征列表。
    missing_rate_df : pd.DataFrame, optional
        缺失率表，需含 'feature'/'variable' 和 'missing_rate' 列。
    psi_df : pd.DataFrame, optional
        PSI 汇总表，需含 'feature'/'variable' 和 'psi_value' 列。
        如果为 None，PSI 投票默认全部通过。
    iv_df : pd.DataFrame, optional
        IV 汇总表，需含 'variable'/'feature' 和 'info_value' 列。
    corr_matrix_df : pd.DataFrame, optional
        相关性矩阵 DataFrame（索引和列均为特征名）。若为 None 则 corr 条件默认全部通过。
    missing_rate_max : float
        缺失率上限，超过则该项不通过（missing_rate < threshold → 通过）。
    psi_threshold : float or None
        PSI 上限，超过则该项不通过（psi < threshold → 通过）；None 时使用默认 0.25。
    iv_min : float
        IV 下限，低于此值该项不通过（iv >= threshold → 通过）。
    corr_threshold : float
        相关性去重阈值。
    vote_pass_count : int
        通过门槛：>= 此票数则保留。
    var_keep_list : list[str], optional
        强制保留的特征列表（投票不够也保留）。
    var_drop_list : list[str], optional
        强制删除的特征列表（投票够了也删掉）。

    Returns
    -------
    list[str]
        通过投票筛选的特征列表。
    pd.DataFrame
        每个特征的投票详情表（用于报告输出）。
    """
    var_keep = set(var_keep_list) if var_keep_list else set()
    var_drop = set(var_drop_list) if var_drop_list else set()

    features = list(feature_list)
    n_active_conditions = 0  # 实际参与投票的条件数（不含默认通过的条件）

    # ================================================================
    # 1. 缺失率投票
    #    规则：missing_rate < missing_rate_max → 通过
    # ================================================================
    mr_pass = pd.Series(True, index=features, name="_mr")
    if missing_rate_df is not None:
        mr = missing_rate_df.copy()
        feat_col_mr = 'feature' if 'feature' in mr.columns else 'variable'
        max_mr = mr.groupby(feat_col_mr)['missing_rate'].max().reset_index()
        max_mr.columns = ['feature', 'missing_rate']
        lookup = max_mr.set_index('feature')['missing_rate'].to_dict()
        mr_pass = pd.Series([lookup.get(f, 0.0) < missing_rate_max for f in features], index=features, name="_mr")
        n_active_conditions += 1
        failed_n = (~mr_pass).sum()
        print("[Vote] 缺失率投票 (< %.2f): %d 通过 / %d 不通过" % (missing_rate_max, mr_pass.sum(), failed_n))

    # ================================================================
    # 2. IV 投票（Step2 计算，总是存在）
    # ================================================================
    iv_pass = pd.Series(True, index=features, name="_iv")
    if iv_df is not None:
        feat_col_iv = 'variable' if 'variable' in iv_df.columns else 'feature'
        iv_lookup = iv_df.groupby(feat_col_iv)['info_value'].max().to_dict()
        iv_pass = pd.Series([iv_lookup.get(f, 0.0) >= iv_min for f in features], index=features, name="_iv")
        n_active_conditions += 1
        failed_n = (~iv_pass).sum()
        print("[Vote] IV 投票 (>= %.4f): %d 通过 / %d 不通过" % (iv_min, iv_pass.sum(), failed_n))

    # ================================================================
    # 3. PSI 投票（Step1产出，可选）
    #    规则：PSI < psi_threshold → 通过
    #    若 psi_df 为 None（开关关闭），默认全部通过
    # ================================================================
    psi_pass = pd.Series(True, index=features, name="_psi")
    if psi_df is not None:
        feat_col_psi = 'feature' if 'feature' in psi_df.columns else 'variable'
        # 兼容两种列名: 'psi' 或 'psi_value'
        val_col_psi = 'psi_value' if 'psi_value' in psi_df.columns else 'psi'
        psi_lookup = psi_df.groupby(feat_col_psi)[val_col_psi].max().to_dict()
        effective_psi_threshold = psi_threshold if psi_threshold is not None else 0.25
        psi_pass = pd.Series([psi_lookup.get(f, 0.0) < effective_psi_threshold for f in features], index=features, name="_psi")
        n_active_conditions += 1
        failed_n = (~psi_pass).sum()
        print("[Vote] PSI 投票 (< %.4f): %d 通过 / %d 不通过" % (effective_psi_threshold, psi_pass.sum(), failed_n))

    # ================================================================
    # 4. 相关性投票（Step1产出，可选）
    # ================================================================
    corr_pass = pd.Series(True, index=features, name="_corr")
    if corr_matrix_df is not None and len(corr_matrix_df.columns) >= 2:
        # 只要相关性矩阵存在且有数据就参与投票
        # 只考虑那些同时在候选特征和相关性矩阵中的特征
        n_active_conditions += 1
        # 找出所有高相关对中需要被剔的特征（IV 低的那个）
        to_drop_corr = set()
        feature_idx = {f: i for i, f in enumerate(corr_matrix_df.columns) if f in features}
        feat_indices = sorted(feature_idx.values())

        for ii in range(len(feat_indices)):
            i = feat_indices[ii]
            f_i = corr_matrix_df.columns[i]
            if f_i in to_drop_corr:
                continue
            for jj in range(ii + 1, len(feat_indices)):
                j = feat_indices[jj]
                f_j = corr_matrix_df.columns[j]
                if abs(corr_matrix_df.iloc[i, j]) >= corr_threshold:
                    iv_i = iv_lookup.get(f_i, 0.0) if iv_df is not None else 0.0
                    iv_j = iv_lookup.get(f_j, 0.0) if iv_df is not None else 0.0
                    if iv_i >= iv_j:
                        to_drop_corr.add(f_j)
                    else:
                        to_drop_corr.add(f_i)

        corr_pass = pd.Series([f not in to_drop_corr for f in features], index=features, name="_corr")
        failed_n = (~corr_pass).sum()
        print("[Vote] 相关性投票 (>= %.2f): %d 通过 / %d 不通过" % (corr_threshold, corr_pass.sum(), failed_n))

    # ================================================================
    # 汇总投票表
    # 只包含实际有数据的条件列；默认通过的条件不计票
    # ================================================================
    vote_df = pd.DataFrame(index=pd.Index(features, name="feature"))

    # MR：有数据才加入投票
    if missing_rate_df is not None:
        vote_df["missing_rate"] = mr_pass

    # IV：总是存在（Step2 自行计算）
    vote_df["iv"] = iv_pass

    # PSI：只有 psi_df 不为 None 时才加入投票
    if psi_df is not None:
        vote_df["psi"] = psi_pass

    # CORR：只有相关性矩阵有效加载时才加入投票
    if corr_matrix_df is not None and len(corr_matrix_df.columns) >= 2:
        vote_df["corr"] = corr_pass

    # 总票数（只算实际参与的条件）
    vote_df["vote_count"] = vote_df.sum(axis=1)
    total_conditions = n_active_conditions

    # 实际动态门槛：vote_pass_count 不能超过实际参与的条件数
    effective_threshold = min(vote_pass_count, total_conditions)

    # ================================================================
    # 5. 人工干预前的人工判定
    # ================================================================
    vote_df["pass_vote"] = vote_df["vote_count"] >= effective_threshold
    before_intervention = vote_df["pass_vote"].sum()
    print("\n[Vote] 总票数门槛: >= %d / %d" % (effective_threshold, total_conditions))
    print("[Vote] 投票通过: %d / %d 特征" % (before_intervention, len(features)))

    # ================================================================
    # 6. 人工干预：var_keep_list / var_drop_list
    # ================================================================
    keep_changed = 0
    drop_changed = 0

    for f in var_keep:
        if f in vote_df.index and not vote_df.loc[f, "pass_vote"]:
            vote_df.loc[f, "pass_vote"] = True
            keep_changed += 1

    for f in var_drop:
        if f in vote_df.index and vote_df.loc[f, "pass_vote"]:
            vote_df.loc[f, "pass_vote"] = False
            drop_changed += 1

    final_count = vote_df["pass_vote"].sum()
    print("[Vote] 人工干预 - 强制保留: %d, 强制删除: %d" % (keep_changed, drop_changed))
    print("[Vote] 最终投票通过: %d / %d 特征" % (final_count, len(features)))

    result = list(vote_df[vote_df["pass_vote"]].index)

    return result, vote_df
