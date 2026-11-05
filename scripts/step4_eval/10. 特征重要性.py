import joblib
import pandas as pd
import numpy as np

import pandas as pd
import joblib


def extract_model_importance(model_or_path, save_path=None):
    """
    提取模型特征重要性，并可选导出为 CSV。
    兼容 LightGBM 及通用 sklearn 模型。

    Parameters:
    -----------
    model_or_path : str or model object
        模型文件路径（.pkl）或直接传入的模型对象。
    save_path : str, optional
        CSV 保存路径。如果为 None，则仅返回 DataFrame 不保存。

    Returns:
    --------
    pd.DataFrame : 包含特征名及重要性指标的 DataFrame，按 gain 降序排列。
    """
    # 1. 加载模型（如果传入的是路径）
    if isinstance(model_or_path, str):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path

    # 2. 获取特征名
    try:
        # 优先从 LightGBM 的 booster 获取
        feature_names = model.booster_.feature_name()
    except AttributeError:
        # 回退方案：根据模型内置的特征数量生成默认名称
        n_features = getattr(model, 'n_features_in_', len(model.feature_importances_))
        feature_names = [f'feature_{i}' for i in range(n_features)]

    # 3. 获取特征重要性
    try:
        # 优先获取 LightGBM 的 split 和 gain
        importance_split = model.booster_.feature_importance(importance_type='split')
        importance_gain = model.booster_.feature_importance(importance_type='gain')
    except AttributeError:
        # 回退方案：使用通用 sklearn 的 feature_importances_
        importance = model.feature_importances_
        importance_split = importance
        importance_gain = importance

    # 4. 构建并排序 DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_split': importance_split,
        'importance_gain': importance_gain
    }).sort_values(by='importance_gain', ascending=False).reset_index(drop=True)

    # 5. 可选保存
    if save_path:
        df_importance.to_csv(save_path, index=False)
        print(f"[OK] 特征重要性已成功保存至: {save_path}")

    return df_importance
