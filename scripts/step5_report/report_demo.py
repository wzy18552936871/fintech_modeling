"""
结案报告生成器 - 完整版 (支持模型分 + 标分)

Sheet 结构:
  1. 缺失率              - 入模特征缺失率统计
  2. 特征重要性           - 入模变量 importance (split + gain)
  3. KS                  - 模型分 & 每个标分的 KS (整体/分月/分group)
  4. AUC                 - 模型分 & 每个标分的 AUC (整体/分月/分group)
  5. {分数名}_Lift       - 每个分数的 Lift & WOE (一个Sheet一个分数)
  6. {模型名}_top20趋势图 - Top20入模变量的每月分箱占比&坏率图
  7. Score_KDE           - 所有分数 Good/Bad KDE 对比图

标分: extra_features_eval.csv 里的 feature 列 = 原始数据中直接存在的评分字段
模型分: 自己训练的模型 predict 出来的概率值
标签: y (可能多个)
"""
import numpy as np
import pandas as pd
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import xlsxwriter
import warnings
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)


def draw_graph(data, feature, label, month='month', num_bins=10):
    """变量每月分箱占比和坏率双轴图"""
    df = data[[label, feature, month]].copy().reset_index(drop=True)
    df['_is_null'] = df[feature].isna()
    df['_bin'] = None
    nunique = df[feature].nunique(dropna=True)

    if nunique <= num_bins:
        non_null_vals = sorted(df[df['_is_null'] == False][feature].unique())
        bin_labels = [str(v) for v in non_null_vals]
        for i, val in enumerate(non_null_vals):
            df.loc[df[feature] == val, '_bin'] = i
        null_bin_idx = len(non_null_vals)
        df.loc[df['_is_null'], '_bin'] = null_bin_idx
        bin_labels.append('null')
    else:
        non_null_series = df[df['_is_null'] == False][feature]
        try:
            qcut_bins = pd.qcut(non_null_series, q=num_bins, duplicates='drop', retbins=True)[1]
        except ValueError:
            qcut_bins = np.linspace(non_null_series.min(), non_null_series.max(), num_bins + 1)
        qcut_bins[0] = -np.inf
        qcut_bins[-1] = np.inf
        df.loc[df['_is_null'] == False, '_bin'] = pd.cut(
            df.loc[df['_is_null'] == False, feature],
            bins=qcut_bins, labels=False, include_lowest=True
        )
        bin_labels = [f'({qcut_bins[i]:.2f}, {qcut_bins[i + 1]:.2f}]' for i in range(len(qcut_bins) - 1)]
        bin_labels = [l.replace('-inf', '-\u221e').replace('inf', '\u221e') for l in bin_labels]
        null_bin_idx = len(bin_labels)
        df.loc[df['_is_null'], '_bin'] = null_bin_idx
        bin_labels.append('null')

    agg_df = df.groupby([month, '_bin'], dropna=False).agg(
        total=(label, 'count'), bad=(label, 'sum')
    ).reset_index()
    agg_df['bad_rate'] = (agg_df['bad'] / agg_df['total']).fillna(0)
    month_total = df.groupby(month).size().reset_index(name='month_total')
    agg_df = agg_df.merge(month_total, on=month)
    agg_df['ratio'] = agg_df['total'] / agg_df['month_total']

    months = sorted(df[month].unique())
    n_bins_count = len(bin_labels)
    x = np.arange(n_bins_count, dtype=np.float64)
    n_months = len(months)
    width = 0.8 / n_months if n_months > 0 else 0.8

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    handles_ratio, labels_ratio, handles_bad, labels_bad = [], [], [], []
    for i, m in enumerate(months):
        m_data = agg_df[agg_df[month] == m].copy()
        m_data = m_data.set_index('_bin').reindex(np.arange(n_bins_count), fill_value=0).reset_index()
        bar_x = (x - 0.4 + i * width).astype(np.float64)
        ratio_vals = m_data['ratio'].values.astype(np.float64)
        bad_rate_vals = m_data['bad_rate'].values.astype(np.float64)
        bar = ax1.bar(bar_x, ratio_vals, width=width, label=f'{m} ratio')
        line, = ax2.plot(x, bad_rate_vals, marker='.', label=f'{m} bad rate')
        handles_ratio.append(bar); labels_ratio.append(f'{m} ratio')
        handles_bad.append(line); labels_bad.append(f'{m} bad rate')

    ax1.set_ylabel('Ratio'); ax1.set_xticks(x)
    short_labels = [str(l)[:12] for l in bin_labels]
    ax1.set_xticklabels(short_labels, rotation=45, ha='right')
    ax1.set_ylim(0); ax2.set_ylabel('Bad Rate'); ax2.set_ylim(0)
    plt.title(f"Feature: {feature} | Label: {label}")
    handles = handles_ratio + handles_bad; labels = labels_ratio + labels_bad
    if handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig, agg_df


def ks_calc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def calc_ks_auc(df, label_col, score_col):
    """计算单组的KS和AUC — df必须包含label_col和score_col"""
    y_vals = df[label_col].values
    s_vals = df[score_col].values

    if len(np.unique(y_vals)) < 2:
        return {'KS': np.nan, 'AUC': np.nan, 'Good_sum': 0, 'Bad_sum': 0,
                'BadRate': 0.0, 'Total': int(len(y_vals))}

    good = int((y_vals == 0).sum())
    bad = int((y_vals == 1).sum())
    total = good + bad
    bad_rate = bad / total if total > 0 else 0.0
    auc = roc_auc_score(y_vals, s_vals)
    ks = ks_calc(y_vals, s_vals)

    return {'KS': round(ks, 6), 'AUC': round(auc, 6),
            'Good_sum': good, 'Bad_sum': bad,
            'BadRate': round(bad_rate, 6), 'Total': total}


def calc_lift(df, label_col, score_col, n_bins=10):
    """计算分箱 Lift & WOE"""
    t = df.copy()
    try:
        t['Bin_No'] = pd.qcut(t[score_col], q=n_bins, labels=False, duplicates='drop')
    except Exception:
        t['Bin_No'] = pd.cut(t[score_col], bins=n_bins, labels=False)

    grp = t.groupby('Bin_No').agg(
        Count_0=(label_col, lambda x: (x == 0).sum()),
        Count_1=(label_col, lambda x: (x == 1).sum())
    ).reset_index()

    total_0 = int(grp['Count_0'].sum())
    total_1 = int(grp['Count_1'].sum())
    obr = total_1 / (total_0 + total_1) if (total_0 + total_1) > 0 else 0

    grp['Total'] = grp['Count_0'] + grp['Count_1']
    grp['Target_Rate'] = grp['Count_1'] / grp['Total']
    grp['Lift'] = grp['Target_Rate'] / obr if obr > 0 else 0

    eps = 1e-6
    p0 = grp['Count_0'] / total_0 if total_0 > 0 else 0
    p1 = grp['Count_1'] / total_1 if total_1 > 0 else 0
    grp['Woe_Value'] = np.log((p0 + eps) / (p1 + eps))

    try:
        be = pd.qcut(t[score_col], q=n_bins, retbins=True, duplicates='drop')[1]
    except Exception:
        be = np.linspace(t[score_col].min(), t[score_col].max(), n_bins + 1)
    grp['Bin_Range'] = [f'({be[i]:.4f}, {be[i+1]:.4f}]' for i in grp['Bin_No']]

    return grp[['Bin_No', 'Bin_Range', 'Count_0', 'Count_1', 'Total',
                'Target_Rate', 'Lift', 'Woe_Value']]


def safe_sheet(name, max_len=31):
    return str(name)[:max_len]


# ================================================================
def generate_multi_label_report(
    df_raw,
    label_list,              # ['y'] 或 ['y','p2m0']
    feature_list,            # 入模特征 B001, B002, ...
    month_col,               # 'month' or None
    output_path='report.xlsx',
    model_name='LGBM',
    extra_scores=None,       # 标分列名列表: ['xy_score','nwd_score']
):
    """
    df_raw 必须包含:
      - feature_list (入模特征)
      - label_list   (标签列)
      - score_{label} (模型预测分, 由 main.py predict 后写入)
      - extra_scores   (标分原始字段)
    """

    workbook = xlsxwriter.Workbook(
        output_path, {'constant_memory': True, 'nan_inf_to_errors': True}
    )
    hf = workbook.add_format({
        'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
        'border': 1, 'align': 'center'})
    cf = workbook.add_format({'border': 1, 'align': 'center', 'num_format': '0.000000'})

    if extra_scores is None:
        extra_scores = []

    # ---------------------------------------------------------------
    # 构建 all_scores_info: [(列名, 显示名, 类型), ...]
    #   类型: '模型分' or '标分'
    # ---------------------------------------------------------------
    all_scores = []
    for lbl in label_list:
        sc = f'score_{lbl}'
        if sc in df_raw.columns:
            all_scores.append((sc, f'{model_name}模型分', '模型分'))
            print(f"  [Score] {sc} -> {model_name}模型分")

    for col in extra_scores:
        if col in df_raw.columns:
            all_scores.append((col, col, '标分'))
            print(f"  [Score] {col} -> {col} (标分)")
        else:
            print(f"  [WARN] 标分 {col} 不在数据中，跳过")

    if not all_scores:
        print("[ERROR] 没有分数列"); workbook.close(); return

    # ---------------------------------------------------------------
    # Sheet 1: 缺失率
    # ---------------------------------------------------------------
    print("[Sheet] 缺失率 ...")
    if feature_list:
        af = [f for f in feature_list if f in df_raw.columns]
        mr = df_raw[af].isna().mean() * 100
        mr_df = pd.DataFrame({
            '特征': mr.index, '缺失率(%)': mr.values.round(4),
            '非空样本数': (df_raw[af].notna().sum()).values,
            '总样本': len(df_raw)
        }).sort_values('缺失率(%)', ascending=False).reset_index(drop=True)

        ws = workbook.add_worksheet('缺失率')
        hdr = ['特征', '缺失率(%)', '非空样本数', '总样本']
        ws.write_row(0, 0, hdr, hf)
        for r, (_, row) in enumerate(mr_df.iterrows(), 1):
            ws.write_row(r, 0, [row[hdr[0]], round(row[hdr[1]],4), int(row[hdr[2]]), int(row[hdr[3]])], cf)

    # ---------------------------------------------------------------
    # Sheet 2: 特征重要性
    # ---------------------------------------------------------------
    print("[Sheet] 特征重要性 ...")
    if feature_list:
        feat_names = [f for f in feature_list if f in df_raw.columns]
        try:
            if hasattr(df_raw, '_feat_importance'):
                is_ = np.array(df_raw._feat_importance['split'])
                ig_ = np.array(df_raw._feat_importance['gain'])
            else:
                is_ = np.zeros(len(feat_names))
                ig_ = np.zeros(len(feat_names))
        except Exception:
            is_ = np.zeros(len(feat_names)); ig_ = np.zeros(len(feat_names))

        df_imp = pd.DataFrame({
            '特征名': feat_names,
            'importance_split': is_,
            'importance_gain': ig_,
        }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

        ws = workbook.add_worksheet('特征重要性')
        ihdr = ['特征名', 'importance_split', 'importance_gain']
        ws.write_row(0, 0, ihdr, hf)
        for r, (_, row) in enumerate(df_imp.iterrows(), 1):
            ws.write_row(r, 0, [row[ihdr[0]], int(row[ihdr[1]]), round(float(row[ihdr[2]]),4)], cf)

        top20 = df_imp.head(20)['特征名'].tolist()
    else:
        top20 = []

    # ================================================================
    # 以下逻辑按 label 循环（每个标签独立分析）
    # ================================================================
    for label in label_list:
        lu = label.upper()
        lbl_scores = [(sc, dn, tp) for sc, dn, tp in all_scores]
        if f'score_{label}' not in [s[0] for s in lbl_scores]:
            continue  # 没有该标签的模型分，跳过

        # -----------------------------------------------------------
        # Sheet 3: KS (模型分+标分 × 整体/分月/分group)
        # -----------------------------------------------------------
        print(f"[Sheet] {label}_KS ...")
        ws_ks = workbook.add_worksheet(safe_sheet(f'KS_{lu}'))
        khdr = ['分数', '分数类型', '分组', '样本数', '好账数', '坏账数', '坏账率', 'KS']
        ws_ks.write_row(0, 0, khdr, hf)

        ri = 1
        for sc_col, sc_name, sc_type in lbl_scores:
            # base_cols 必须包含 score + label
            bc = [sc_col, label]
            if month_col and month_col in df_raw.columns:
                bc.append(month_col)
            if 'group' in df_raw.columns:
                bc.append('group')

            base = df_raw[bc].dropna(subset=[sc_col])

            m = calc_ks_auc(base, label, sc_col)
            ws_ks.write(ri, 0, sc_name); ws_ks.write(ri, 1, sc_type)
            ws_ks.write(ri, 2, '整体')
            ws_ks.write(ri, 3, m['Total']); ws_ks.write(ri, 4, m['Good_sum'])
            ws_ks.write(ri, 5, m['Bad_sum'])
            ws_ks.write(ri, 6, round(m['BadRate'],6))
            ws_ks.write(ri, 7, round(m['KS'],6))
            ri += 1

            if month_col and month_col in df_raw.columns:
                mm = df_raw[bc].dropna(subset=[sc_col])
                for mv, mg in mm.groupby(month_col):
                    r = calc_ks_auc(mg, label, sc_col)
                    ws_ks.write(ri, 0, sc_name); ws_ks.write(ri, 1, sc_type)
                    ws_ks.write(ri, 2, str(mv))
                    ws_ks.write(ri, 3, r['Total']); ws_ks.write(ri, 4, r['Good_sum'])
                    ws_ks.write(ri, 5, r['Bad_sum'])
                    ws_ks.write(ri, 6, round(r['BadRate'],6))
                    ws_ks.write(ri, 7, round(r['KS'],6))
                    ri += 1

            if 'group' in df_raw.columns:
                gm = df_raw[bc].dropna(subset=[sc_col])
                for gv, gg in gm.groupby('group'):
                    r = calc_ks_auc(gg, label, sc_col)
                    ws_ks.write(ri, 0, sc_name); ws_ks.write(ri, 1, sc_type)
                    ws_ks.write(ri, 2, f'group_{gv}')
                    ws_ks.write(ri, 3, r['Total']); ws_ks.write(ri, 4, r['Good_sum'])
                    ws_ks.write(ri, 5, r['Bad_sum'])
                    ws_ks.write(ri, 6, round(r['BadRate'],6))
                    ws_ks.write(ri, 7, round(r['KS'],6))
                    ri += 1

        # -----------------------------------------------------------
        # Sheet 4: AUC (同上结构)
        # -----------------------------------------------------------
        print(f"[Sheet] {label}_AUC ...")
        ws_auc = workbook.add_worksheet(safe_sheet(f'AUC_{lu}'))
        ahdr = ['分数', '分数类型', '分组', '样本数', '好账数', '坏账数', '坏账率', 'AUC']
        ws_auc.write_row(0, 0, ahdr, hf)

        ri = 1
        for sc_col, sc_name, sc_type in lbl_scores:
            bc = [sc_col, label]
            if month_col and month_col in df_raw.columns:
                bc.append(month_col)
            if 'group' in df_raw.columns:
                bc.append('group')

            base = df_raw[bc].dropna(subset=[sc_col])

            m = calc_ks_auc(base, label, sc_col)
            ws_auc.write(ri, 0, sc_name); ws_auc.write(ri, 1, sc_type)
            ws_auc.write(ri, 2, '整体')
            ws_auc.write(ri, 3, m['Total']); ws_auc.write(ri, 4, m['Good_sum'])
            ws_auc.write(ri, 5, m['Bad_sum'])
            ws_auc.write(ri, 6, round(m['BadRate'],6))
            ws_auc.write(ri, 7, round(m['AUC'],6))
            ri += 1

            if month_col and month_col in df_raw.columns:
                mm = df_raw[bc].dropna(subset=[sc_col])
                for mv, mg in mm.groupby(month_col):
                    r = calc_ks_auc(mg, label, sc_col)
                    ws_auc.write(ri, 0, sc_name); ws_auc.write(ri, 1, sc_type)
                    ws_auc.write(ri, 2, str(mv))
                    ws_auc.write(ri, 3, r['Total']); ws_auc.write(ri, 4, r['Good_sum'])
                    ws_auc.write(ri, 5, r['Bad_sum'])
                    ws_auc.write(ri, 6, round(r['BadRate'],6))
                    ws_auc.write(ri, 7, round(r['AUC'],6))
                    ri += 1

            if 'group' in df_raw.columns:
                gm = df_raw[bc].dropna(subset=[sc_col])
                for gv, gg in gm.groupby('group'):
                    r = calc_ks_auc(gg, label, sc_col)
                    ws_auc.write(ri, 0, sc_name); ws_auc.write(ri, 1, sc_type)
                    ws_auc.write(ri, 2, f'group_{gv}')
                    ws_auc.write(ri, 3, r['Total']); ws_auc.write(ri, 4, r['Good_sum'])
                    ws_auc.write(ri, 5, r['Bad_sum'])
                    ws_auc.write(ri, 6, round(r['BadRate'],6))
                    ws_auc.write(ri, 7, round(r['AUC'],6))
                    ri += 1

        # -----------------------------------------------------------
        # Sheet 5: 所有分数的 Lift（合并到一个 Sheet，分数间用空行分隔）
        # -----------------------------------------------------------
        print(f"[Sheet] {label}_Lift ...")
        ws_lift = workbook.add_worksheet(safe_sheet(f'Lift_{lu}'))
        lh = ['分数', '分箱', '区间', '好账数', '坏账数', '总数', '坏账率', 'Lift', 'WOE']
        ws_lift.write_row(0, 0, lh, hf)

        ri = 1
        for sc_col, sc_name, sc_type in lbl_scores:
            cl = df_raw[[sc_col, label]].dropna()
            lf = calc_lift(cl, label, sc_col, n_bins=10)

            # 每个分数的数据行
            for _, row in lf.iterrows():
                vals = [sc_name,
                        int(row['Bin_No']), str(row['Bin_Range']),
                        int(row['Count_0']), int(row['Count_1']), int(row['Total']),
                        round(float(row['Target_Rate']),6),
                        round(float(row['Lift']),6),
                        round(float(row['Woe_Value']),4)]
                ws_lift.write_row(ri, 0, vals, cf)
                ri += 1

            # 分数之间留一个空行
            ri += 1

        # -----------------------------------------------------------
        # Sheet 6: Top20趋势图 (仅模型分的入模特征)
        # -----------------------------------------------------------
        if top20:
            print(f"[Sheet] {model_name}_top20趋势图 ...")
            sn = safe_sheet(f"{model_name}_top20趋势图")
            ws_t = workbook.add_worksheet(sn)
            ws_t.set_column('A:A', 22); ws_t.set_column('B:B', 95)

            img_s = 0.35; rspan = int((300 * img_s) / 15) + 2
            for rank, feat in enumerate(top20):
                if feat not in df_raw.columns:
                    continue
                try:
                    fig, _ = draw_graph(df_raw, feat, label, month=month_col, num_bins=10)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0); plt.close(fig)

                    sr = rank * rspan
                    ws_t.write(sr, 0, f"Top {rank+1}: {feat}", hf)
                    ws_t.insert_image(f'B{sr+1}', 'img.png',
                        {'image_data': buf, 'x_scale': img_s, 'y_scale': img_s})
                except Exception as e:
                    print(f"  [WARN] {feat} plot failed: {e}")

        # -----------------------------------------------------------
        # Sheet 7a: Score_KDE_对比 — 所有分数画在同一坐标系
        # Sheet 7b: Score_KDE_子图   — 每个分数独立一个子图
        # -----------------------------------------------------------
        print(f"[Sheet] Score_KDE_{lu} ...")

        model_scores = [s for s in lbl_scores if s[2] == '模型分']
        flag_scores = [s for s in lbl_scores if s[2] == '标分']
        all_plot_scores = model_scores + flag_scores

        kde_colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#e67e22',
                      '#1abc9c','#34495e']

        # ===== Sheet 7a: 所有分数叠加在同一张图上 =====
        fig_all, ax_all = plt.subplots(figsize=(12, 7))
        for idx, (sc_col, sc_name, sc_type) in enumerate(all_plot_scores):
            c = kde_colors[idx % len(kde_colors)]
            valid_data = df_raw[sc_col].dropna()
            if len(valid_data) > 1:
                valid_data.plot(kind='kde', ax=ax_all, color=c, linewidth=2.5,
                                label=f'{sc_name} ({sc_type})')
        ax_all.set_title(f"所有分数整体分布对比 | Label: {label}", fontsize=14)
        ax_all.set_xlabel("Score Value")
        ax_all.set_ylabel("Density")
        ax_all.legend(loc='best', fontsize=9, ncol=2)
        ax_all.grid(alpha=0.3)
        plt.tight_layout()

        buf_all = io.BytesIO()
        fig_all.savefig(buf_all, format='png', dpi=120, bbox_inches='tight')
        buf_all.seek(0); plt.close(fig_all)

        ws_kde_all = workbook.add_worksheet(safe_sheet(f'Score_KDE_对比_{lu}'))
        ws_kde_all.insert_image('B2', 'img.png',
            {'image_data': buf_all, 'x_scale': 0.75, 'y_scale': 0.75})
        ws_kde_all.set_column('B:B', 120)

        # ===== Sheet 7b: 每个分数各占一个子图 =====
        n_total = len(all_plot_scores)
        cols_grid = min(3, n_total)
        rows_grid = (n_total + cols_grid - 1) // cols_grid

        fig_sub, axes = plt.subplots(rows_grid, cols_grid,
                                     figsize=(14, 4 * rows_grid),
                                     squeeze=False)

        for idx, (sc_col, sc_name, sc_type) in enumerate(all_plot_scores):
            r = idx // cols_grid
            c_col = idx % cols_grid
            ax = axes[r, c_col]
            c = kde_colors[idx % len(kde_colors)]

            valid_data = df_raw[sc_col].dropna()
            if len(valid_data) > 1:
                valid_data.plot(kind='kde', ax=ax, color=c, linewidth=2.5,
                                label=sc_name)
            ax.set_title(f"{sc_name} ({sc_type})", fontsize=12)
            ax.set_xlabel("Score Value")
            if c_col == 0:
                ax.set_ylabel("Density")
            ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

        for i in range(n_total, rows_grid * cols_grid):
            r = i // cols_grid
            c_col = i % cols_grid
            axes[r, c_col].axis('off')

        fig_sub.suptitle(f"各分数独立分布 | Label: {label}", fontsize=14)
        plt.tight_layout()

        buf_sub = io.BytesIO()
        fig_sub.savefig(buf_sub, format='png', dpi=120, bbox_inches='tight')
        buf_sub.seek(0); plt.close(fig_sub)

        ws_kde_sub = workbook.add_worksheet(safe_sheet(f'Score_KDE_子图_{lu}'))
        ws_kde_sub.insert_image('B2', 'img.png',
            {'image_data': buf_sub, 'x_scale': 0.65, 'y_scale': 0.65})
        ws_kde_sub.set_column('B:B', 130)

    workbook.close()
    print(f"\n[emoji] 报告生成完毕: {output_path}")

    gc.collect()
