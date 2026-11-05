import gc
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
from tqdm import tqdm
import scorecardpy as sc

#====================== 定义计算IV值函数 ============================
def calculate_iv_with_bins(
    data: pd.DataFrame,
    label_col: str,
    features: list,
    positive: int = 1,
    group_cols: list = None,
    bin_method: str = 'chimerge',
    output_excel: str = None
):
    """
    计算特征IV值并自动分箱。

    默认行为（group_cols=None）：基于全量数据一次性计算，不按 label/月份/客群分组拆分。
    若需分组计算，显式传入 group_cols 参数。

    参数:
        data: 包含特征和标签的DataFrame（全量数据）
        label_col: 标签列名
        features: 需要计算的特征列表
        positive: 正样本标识(默认1)
        group_cols: 分组列名列表（默认为None，即全量一次性计算）
        bin_method: 分箱方法(默认chimerge)
        output_excel: 结果输出路径
    返回:
        IV值汇总表和分箱明细表
    """
    bin_details = []
    iv_all = []

    data = data.copy(deep=True)

    # 处理标签列中的NaN
    if data[label_col].isna().any():
        print(f"[WARN] 标签列 '{label_col}' 中存在 {data[label_col].isna().sum()} 个NaN值，已自动移除")
        data = data.dropna(subset=[label_col]).copy()

    # 转换标签列类型
    try:
        data.loc[:, label_col] = data[label_col].astype(int)
    except Exception as e:
        print(f"[ERROR] 标签列转换失败: {str(e)}")
        return None

    # 如果未提供分组列
    temp_group_flag = False
    if not group_cols:
        data['_tmp_group_'] = '全局'
        group_cols = ['_tmp_group_']
        temp_group_flag = True

    data = data.reset_index(drop=True)
    grouped = data.groupby(group_cols)
    group_keys_list = list(grouped.groups.keys())

    total_tasks = len(group_keys_list) * len(features)
    pbar = tqdm(total=total_tasks, desc='总进度')

    for group_keys in group_keys_list:
        sub_data = grouped.get_group(group_keys).copy(deep=True).reset_index(drop=True)

        if isinstance(group_keys, tuple):
            group_name = '_'.join([str(k) for k in group_keys])
            group_dict = {col: val for col, val in zip(group_cols, group_keys)}
        else:
            group_name = str(group_keys)
            group_dict = {group_cols[0]: group_keys}

        for col in features:
            pbar.set_description(f"处理 {group_name} - {col}")
            pbar.update(1)
            try:
                if sub_data[col].nunique(dropna=True) <= 1:
                    continue

                unique_y = sub_data[label_col].dropna().unique()
                if len(unique_y) != 2:
                    print(f"[ERROR] 分组[{group_name}] - 特征[{col}] 失败: 需要严格二分类标签，实际唯一值: {unique_y}")
                    continue

                # 创建干净的工作数据集
                working_df = sub_data[[col, label_col]].copy(deep=True)
                working_df = working_df.dropna(subset=[col, label_col])
                working_df = working_df.reset_index(drop=True)
                working_df[col] = working_df[col].astype(float)  # 确保是float类型

                # 执行woebin
                try:
                    bin_result = sc.woebin(
                        dt=working_df,
                        y=label_col,
                        positive=str(positive),
                        method=bin_method,
                        no_cores=1
                    )
                except Exception as e:
                    print(f"[ERROR] 分组[{group_name}] - 特征[{col}] woebin调用失败: {str(e)}")
                    continue

                df = bin_result[col].copy()
                df['variable'] = col

                for gcol, gval in group_dict.items():
                    df[gcol] = gval

                keep_cols = [*group_cols, 'variable']
                other_cols = [c for c in df.columns if c not in keep_cols]
                df = df[keep_cols + other_cols]

                bin_details.append(df)

                iv_row = {
                    **group_dict,
                    'variable': col,
                    'info_value': df['bin_iv'].sum(),
                    'bin_num': df.shape[0],
                }
                iv_all.append(iv_row)

            except Exception as e:
                print(f"[ERROR] 分组[{group_name}] - 特征[{col}] 处理异常: {str(e)}")
                continue

    pbar.close()

    try:
        bin_details_df = pd.concat(bin_details, ignore_index=True) if bin_details else pd.DataFrame()
        iv_df_all = pd.DataFrame(iv_all) if iv_all else pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] 结果合并失败: {str(e)}")
        return None

    if not iv_df_all.empty:
        sort_cols = group_cols + ['info_value']
        iv_df_all = iv_df_all.sort_values(by=sort_cols, ascending=[True] * (len(sort_cols)-1) + [False])
        iv_df_all = iv_df_all[[*group_cols, 'variable', 'info_value', 'bin_num']]

    if output_excel:
        try:
            with pd.ExcelWriter(output_excel) as writer:
                iv_df_all.to_excel(writer, sheet_name='IV总览', index=False)
                if not bin_details_df.empty:
                    bin_details_df.to_excel(writer, sheet_name='分箱明细', index=False)
            print(f"\n[OK] 结果已保存至: {output_excel}")
        except Exception as e:
            print(f"[ERROR] Excel写入失败: {str(e)}")
            return None

    if temp_group_flag and '_tmp_group_' in data.columns:
        data.drop(columns=['_tmp_group_'], inplace=True)

    del bin_details, iv_all
    gc.collect()
    return iv_df_all, bin_details_df



# #======== 读取数据并计算 =============
# total_data_finnal = pd.read_csv(r'D:\workspace\jyz202508\1.工作过程\total_data_modelscore.csv')
# total_data_finnal = total_data_finnal[total_data_finnal['entry_channel_name'] == '还呗API']
#
# calculate_iv_with_bins(
#     data = total_data_finnal,
#     label_col = 'mob3_ever30_flag',
#     features = [
#         'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',
#         'f11','f12','f13','f14','f15','f16','f17','f18','f19','f20',
#         'f21','f22','f23','f24','f25','f26','f27','f28','f29','f30',
#         'f31','f32','f33','f34','f35','f36','f37','f38','f39','f40',
#         'f41','f42','f43','f44','f45','f46','f47','f48','f49','f50',
#         'f51','f52','f53','f54','f55','f56','f57','f58','f59','f60'
#     ],
#     positive = 1,
#     group_cols = None,
#     bin_method = 'chimerge',
#     output_excel = r'iv_还呗.xlsx'
#
# )




