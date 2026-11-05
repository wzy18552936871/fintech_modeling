import numpy as np
import pandas as pd


# ==========================================
# 3. 建模特征探查与类型转换
# ==========================================
def explore_and_convert(df, model_features):
    print("=" * 40)
    print("【步骤一】建模特征探查与类型转换")
    print("=" * 40)

    df_model = df[model_features].copy()

    # 3.1 查看 describe 探查潜在异常
    print("\n[emoji] 数值型特征统计 (describe):")
    print(df_model.describe(include='all'))

    # 3.2 识别并转换 Category 类型 (非数值型)
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    print(f"\n[emoji] 发现以下非数值特征，已转换为 category 类型: {cat_cols}")

    for col in cat_cols:
        df_model[col] = df_model[col].astype('category')

        # 统计数量分布
        print(f"\n[emoji] 类别特征 [{col}] 的数量分布:")
        value_counts = df_model[col].value_counts(dropna=False)
        print(value_counts)

    dtype_map = {col: str(df_model[col].dtype) for col in df_model.columns}
    return df_model, dtype_map


# ==========================================
# 4. 按月份与标签统计分布
# ==========================================
def calc_monthly_label_stats(df, month_col, labels):
    print("\n" + "=" * 40)
    print("【步骤二】按月份统计 Label 分布")
    print("=" * 40)

    stats_list = []
    for label in labels:
        # 按月份分组，计算总量、Bad数量、Bad占比
        monthly_stats = df.groupby(month_col).agg(
            total_cnt=(label, 'count'),
            bad_cnt=(label, 'sum')
        ).reset_index()

        monthly_stats['bad_rate'] = (monthly_stats['bad_cnt'] / monthly_stats['total_cnt']).round(4)
        monthly_stats['label_name'] = label

        stats_list.append(monthly_stats)

        print(f"\n[emoji] 标签 [{label}] 的月度分布:")
        print(monthly_stats.to_string(index=False))

    # 合并所有标签的统计结果
    final_stats = pd.concat(stats_list, ignore_index=True)
    return final_stats