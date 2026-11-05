import pandas as pd
import numpy as np
from tqdm import tqdm

#============== 定义缺失率统计函数 ================
def compute_missing_rate(
    df: pd.DataFrame,
    output_excel_path: str,
    date_col: str = None,
    group_cols=None,
    feature_cols=None
):
    """
    统计特征的缺失率，支持分组和时间窗，并导出Excel。

    参数：
    ----------
    df : pd.DataFrame  数据表
    output_excel_path : str  输出 Excel 文件路径
    date_col : str or None   时间窗列（可选）
    group_cols : str or list[str] or None  分组列（可选）
    feature_cols : str or list[str] or None  要统计的特征列名（默认统计所有非分组/时间列）
    """

    # 参数预处理
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if group_cols is None:
        group_cols = []

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # 构建分组列组合（时间列 + 分组列）
    group_by_cols = []
    if date_col:
        group_by_cols.append(date_col)
    group_by_cols.extend(group_cols)

    # 自动推断特征列
    if feature_cols is None:
        feature_cols = [
            col for col in df.columns
            if col not in group_by_cols
        ]

    result = []

    # 分组统计
    if group_by_cols:
        grouped = df.groupby(group_by_cols)
        grouped = list(grouped)
        total_groups = len(grouped)
        print(f"共分为 {total_groups} 个分组...")

        for keys, group in tqdm(grouped, desc="计算分组缺失率"):
            total = len(group)
            for col in feature_cols:
                missing = group[col].isna().sum()
                record = {
                    **(dict(zip(group_by_cols, keys)) if isinstance(keys, tuple) else {group_by_cols[0]: keys}),
                    'feature': col,
                    'missing_count': missing,
                    'total_count': total,
                    'missing_rate': round(missing / total, 6) if total > 0 else None
                }
                result.append(record)

    else:
        total = len(df)
        total_features = len(feature_cols)
        print(f"对全表统计 {total_features} 个特征缺失率...")

        for col in tqdm(feature_cols, desc="计算缺失率"):
            missing = df[col].isna().sum()
            result.append({
                'feature': col,
                'missing_count': missing,
                'total_count': total,
                'missing_rate': round(missing / total, 6) if total > 0 else None
            })

    # 保存结果
    result_df = pd.DataFrame(result)

    # 确保 feature 放在第一列
    cols = result_df.columns.tolist()
    if 'feature' in cols:
        cols.remove('feature')
        cols = ['feature'] + cols
        result_df = result_df[cols]

    result_df.to_excel(output_excel_path, index=False)
    print(f"\n[OK] 缺失率结果已保存至：{output_excel_path}")

#
# compute_missing_rate(
#     df =log_ali_score_filter,
#     date_col=None,
#     group_cols=None,
#     feature_cols=None,
#     output_excel_path= r'C:\Users\Desktop\3.最终数据\missing_rate.xlsx'
# )
#
#










