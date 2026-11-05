import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import gc
import xlsxwriter

# ==========================================
# 1. 完全保留您的原始绘图逻辑 (仅移除 plt.show() 改为返回 fig)
# ==========================================
def draw_graph(data, feature, label, num_bins=7, month='month', bins=None):
    df = data[[label, feature, month]].copy().reset_index(drop=True)
    df['_is_null'] = df[feature].isna()
    df['_bin'] = None

    nunique = df[feature].nunique(dropna=True)
    
    if bins is not None:
        bins = np.array(bins)
        bins[0] = -np.inf
        bins[-1] = np.inf
        df.loc[df['_is_null'] == False, '_bin'] = pd.cut(
            df.loc[df['_is_null'] == False, feature],
            bins=bins, labels=False, include_lowest=True
        )
        bin_labels = []
        for i in range(len(bins)-1):
            l, r = bins[i], bins[i+1]
            l_str = '-inf' if l == -np.inf else f'{l:.2f}'
            r_str = 'inf' if r == np.inf else f'{r:.2f}'
            bin_labels.append(f'({l_str}, {r_str}]')
        null_bin_idx = len(bin_labels)
        df.loc[df['_is_null'], '_bin'] = null_bin_idx
        bin_labels.append('null')
        
    elif nunique <= num_bins:
        non_null_vals = sorted(df[df['_is_null'] == False][feature].unique())
        bin_labels = non_null_vals
        for i, val in enumerate(non_null_vals):
            df.loc[df[feature] == val, '_bin'] = i
        null_bin_idx = len(non_null_vals)
        df.loc[df['_is_null'], '_bin'] = null_bin_idx
        bin_labels.append('null')
        
    else:
        non_null_series = df[df['_is_null'] == False][feature]
        qcut_bins = pd.qcut(non_null_series, q=num_bins, duplicates='drop', retbins=True)[1]
        qcut_bins[0] = -np.inf
        qcut_bins[-1] = np.inf
        df.loc[df['_is_null'] == False, '_bin'] = pd.cut(
            df.loc[df['_is_null'] == False, feature],
            bins=qcut_bins, labels=False, include_lowest=True
        )
        bin_labels = []
        for i in range(len(qcut_bins)-1):
            l, r = qcut_bins[i], qcut_bins[i+1]
            l_str = '-inf' if l == -np.inf else f'{l:.2f}'
            r_str = 'inf' if r == np.inf else f'{r:.2f}'
            bin_labels.append(f'({l_str}, {r_str}]')
        null_bin_idx = len(bin_labels)
        df.loc[df['_is_null'], '_bin'] = null_bin_idx
        bin_labels.append('null')

    # 聚合数据
    agg_df = df.groupby([month, '_bin'], dropna=False).agg(
        total=(label, 'count'),
        bad=(label, 'sum')
    ).reset_index()
    agg_df['bad_rate'] = agg_df['bad'] / agg_df['total']
    agg_df['bad_rate'] = agg_df['bad_rate'].fillna(0)
    month_total = df.groupby(month).size().reset_index(name='month_total')
    agg_df = agg_df.merge(month_total, on=month)
    agg_df['ratio'] = agg_df['total'] / agg_df['month_total']

    # 绑图逻辑
    months = sorted(df[month].unique())
    n_bins = len(bin_labels)
    x = np.arange(n_bins)
    n_months = len(months)
    width = 0.8 / n_months

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    handles_ratio, labels_ratio, handles_bad, labels_bad = [], [], [], []
    for i, m in enumerate(months):
        m_data = agg_df[agg_df[month] == m].copy()
        m_data = m_data.set_index('_bin').reindex(np.arange(n_bins), fill_value=0).reset_index()
        bar_x = x - 0.4 + i * width
        bar = ax1.bar(bar_x, m_data['ratio'], width=width)
        line, = ax2.plot(x, m_data['bad_rate'], marker='.')
        handles_ratio.append(bar); labels_ratio.append(f'{m} ratio')
        handles_bad.append(line); labels_bad.append(f'{m} bad rate')

    ax1.set_ylabel('Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.set_ylim(0)
    ax2.set_ylabel('Bad Rate')
    ax2.set_ylim(0)
    plt.title(feature)
    fig.legend(handles_ratio + handles_bad, labels_ratio + labels_bad, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # 【修改点】：返回 figure 和聚合数据，不再 plt.show()
    return fig, agg_df


# ==========================================
# 2. 批量处理与 xlsxwriter 写入调度器
# ==========================================
def draw_graph_batch(data, feature_list, label, output_path, **kwargs):
    if output_path is None:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 【性能优化】：开启 constant_memory 模式，流式写入，防内存爆炸
    workbook = xlsxwriter.Workbook(output_path, {'constant_memory': True})
    
    header_format = workbook.add_format({
        'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
        'border': 1, 'align': 'center'
    })
    cell_format = workbook.add_format({'border': 1, 'align': 'center'})

    for feat in feature_list:
        try:
            # 调用您原始的绘图函数
            fig, agg_df = draw_graph(data, feat, label, **kwargs)

            # 【解决空白问题】：将图片保存到内存流，降低 DPI 到 100
            img_stream = io.BytesIO()
            fig.savefig(img_stream, format='png', dpi=100, bbox_inches='tight')
            img_stream.seek(0)
            plt.close(fig)  # 必须关闭释放内存

            # 写入 Excel
            sheet_name = f"{feat[:31]}"
            worksheet = workbook.add_worksheet(sheet_name)

            # 写入表头和数据
            headers = list(agg_df.columns)
            worksheet.write_row(0, 0, headers, header_format)
            data_rows = agg_df.values.tolist()
            for r_idx, row_data in enumerate(data_rows, start=1):
                worksheet.write_row(r_idx, 0, row_data, cell_format)

            # 插入图片 (x_scale=0.6 完美适配 figsize=(12,6) 且不会白屏)
            worksheet.insert_image(
                'H2', 'chart.png',
                {
                    'image_data': img_stream,
                    'x_scale': 0.6,
                    'y_scale': 0.6,
                    'object_position': 1
                }
            )
            worksheet.set_column('H:H', 80)
            print(f"[OK] 成功处理特征: {feat}")

        except Exception as e:
            print(f"[ERROR] 处理特征 {feat} 失败: {e}")

    workbook.close()
    print(f"\n[emoji] 所有任务完成！文件已保存至: {output_path}")

    gc.collect()


# draw_graph_batch(data= old_feats_df_all[['p3m2_score','p4m2_score','p5m2_score','p6m2_score','app_date','p6m2']],
#            feature_list=['p3m2_score','p4m2_score','p5m2_score','p6m2_score'],
#            label='p6m2',
#            num_bins=10,
#            month='app_date',
#            output_path = './fenyuetezhengqushi.xlsx')