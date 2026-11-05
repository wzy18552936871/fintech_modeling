import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df, features, save_path=None, figsize=(10, 8), annot=True):
    """
    计算特征相关性矩阵并绘制热力图。

    Parameters:
    -----------
    df : pd.DataFrame
        包含特征数据的 DataFrame。
    features : list
        需要计算相关性的特征名列表。
    save_path : str, optional
        图片保存路径（如 'corr_heatmap.png'）。如果为 None，则仅展示不保存。
    figsize : tuple, optional
        图表尺寸，默认为 (10, 8)。
    annot : bool, optional
        是否在热力图中显示数值标注，默认为 True。

    Returns:
    --------
    pd.DataFrame : 特征相关系数矩阵。
    """
    # 1. 计算相关系数矩阵
    corr_matrix = df[features].corr()

    # 2. 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        linewidths=0.5
    )
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()

    # 3. 可选保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 热力图已成功保存至: {save_path}")

    plt.show()

    return corr_matrix