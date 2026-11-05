"""Split wide_table into Train/Valid/OOT parquet and save for Step 3."""

import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def get_source_split_config(data_source: Dict) -> Dict:
    """从单个data_source中读取data_split配置，返回带默认值的字典。"""
    ds = data_source.get("data_split", {})
    defaults = {
        "oot_tail_months": 4,
        "val_ratio": 0.3,
        "random_state": 42,
    }
    return {k: ds.get(k, v) for k, v in defaults.items()}


def _filter_valid_months(df, month_col, label):
    """按月份过滤：排除该label在该月只有一个唯一值（0或1或全空）的月份。

    注意：这里传入的是 dropna(label) 后的数据，只用于推断哪些月份有区分度。
    数据集拆分本身（train/valid/OOT的行分配）不使用 label 过滤，
    而是基于全量数据的月份分布来切分，保证特征分布的代表性。

    返回 valid_months 列表。
    """
    months = sorted(df[month_col].dropna().unique().tolist())
    if len(months) == 0:
        return []

    # 检查每个月份中该label是否有有效分布（至少2个不同值）
    valid_months = []
    for m in months:
        subset = df[df[month_col] == m][label].dropna()
        if len(subset.unique()) >= 2:
            valid_months.append(m)
        else:
            print(f"[Data Split] Skipping month {m} for label={label}: "
                  f"only one unique value ({subset.unique().tolist()}), insufficient for AUC/KS.")
    return valid_months


def infer_and_split_months(month_col_values: pd.Series, oot_tail: int = 4) -> Tuple[List, List]:
    """从月份列推断 train/oot 月份列表。

    Returns (train_months, oot_months)
    """
    months = sorted(month_col_values.dropna().unique().tolist())
    if len(months) <= oot_tail:
        return months, []
    return months[:-oot_tail], months[-oot_tail:]


def infer_and_split_months_by_label(df, month_col, label, oot_tail: int = 4) -> Tuple[List, List]:
    """针对特定label推断 train/oot 月份列表，先过滤无效月份。

    无效月份定义：该label在该月只有一个唯一值（全0或全1），无法计算AUC/KS。

    Returns (train_months, oot_months)
    """
    valid_months = _filter_valid_months(df, month_col, label)
    if len(valid_months) <= oot_tail:
        return valid_months, []
    return valid_months[:-oot_tail], valid_months[-oot_tail:]


def split_wide_table_and_save(
    wide_table_path: str,
    month_col: str,
    label_cols: List[str],
    feature_cols: List[str],
    output_dir: str,
    oot_tail_months: int = 4,
    val_ratio: float = 0.3,
    random_state: int = 42,
) -> Dict[str, str]:
    """读取 wide_table → 按月份拆分 Train/Valid/OOT → 保存为 parquet。

    每个 label 各产出一套 train/valid/oot parquet（只含选中特征+label列，不含无关数据）。
    OOT 月份筛选：先排除该label只有一个唯一值的月份，再从剩余月份中取最后N个月。

    Returns dict of {split_name: file_path} for verification.
    """
    from pathlib import Path as _Path

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Safety: filter to only columns that actually exist in the parquet file
    import pyarrow.parquet as pq
    schema_cols = pq.ParquetFile(wide_table_path).schema.names

    # Check if month_col exists; if not, try app_date as fallback
    actual_month_col = month_col
    if month_col not in schema_cols and "app_date" in schema_cols:
        print(f"[Data Split] WARNING: month_col '{month_col}' not found, falling back to 'app_date'")
        # Extract month from app_date (first 6 chars: YYYYMM -> integer)
        actual_month_col = "_month_from_appdate"
    elif month_col not in schema_cols:
        print(f"[Data Split] ERROR: month_col '{month_col}' not found in wide_table. Available cols: {schema_cols[:20]}")

    all_needed_cols = list(feature_cols) + [actual_month_col] + label_cols
    existing_cols = [c for c in all_needed_cols if c in schema_cols]
    missing_cols = [c for c in all_needed_cols if c not in schema_cols]
    if missing_cols:
        print(f"[Data Split] WARNING: {len(missing_cols)} columns not found in wide_table, skipping them: {missing_cols[:10]}")
    all_needed_cols = existing_cols

    print("[Data Split] Reading wide_table: %d columns..." % len(all_needed_cols))
    df = pd.read_parquet(wide_table_path, columns=all_needed_cols)

    # If we need to derive month from app_date
    if actual_month_col == "_month_from_appdate" and "app_date" in df.columns:
        print("[Data Split] Deriving month column from app_date...")
        df["_month_from_appdate"] = pd.to_numeric(df["app_date"].astype(str).str[:6], errors="coerce")
        actual_month_col = "_month_from_appdate"

    print("[Data Split] Loaded shape: %s" % str(df.shape))

    # Update feature_cols to reflect actual available features
    feature_cols = [c for c in feature_cols if c in schema_cols]

    # Use actual_month_col for all subsequent operations
    month_col = actual_month_col

    results = {}

    # 每个 label 分别保存（因为Y不同，且月份有效性可能不同）
    for label in label_cols:
        # --- 用有label的数据推断有效月份（仅用于判断哪些月有区分度）---
        df_label_valid = df.dropna(subset=[label]).copy()
        if df_label_valid.empty:
            print("[Data Split] WARNING: %s has no non-null values, skipping." % label)
            continue

        # 针对该label过滤无效月份后再拆分
        train_months, oot_months = infer_and_split_months_by_label(
            df_label_valid, month_col, label, oot_tail=oot_tail_months
        )
        print("[Data Split] %s: valid train months=%d | OOT months=%d" % (
            label, len(train_months), len(oot_months)))

        if not train_months:
            print("[Data Split] ERROR: No valid train months for %s, aborting." % label)
            continue

        # --- 数据集拆分：先按月份筛选，再去掉label为NaN的行，仅保留有效标签数据 ---
        df_train_set = df[df[month_col].isin(train_months)].copy().dropna(subset=[label])

        del df_label_valid  # 月份推断完就释放
        gc.collect()

        # OOT：同样去除label为NaN的行
        if oot_months:
            df_oot = df[df[month_col].isin(oot_months)].copy().dropna(subset=[label])
        else:
            df_oot = pd.DataFrame()

        n_train_total = len(df_train_set)

        print("[Data Split] %s: train_set有效标签=%d rows (已去除label为NaN的数据)" % (
            label, n_train_total))

        # 直接对整个train_set分层拆分 Train/Valid（已经没有空标签行了）
        if len(df_train_set) == 0:
            print("[Data Split] ERROR: No rows with valid label for %s after month filter." % label)
            continue

        n_valid = int(len(df_train_set) * val_ratio)

        from sklearn.model_selection import train_test_split
        df_train, df_valid = train_test_split(
            df_train_set,
            test_size=n_valid,
            random_state=random_state,
            stratify=df_train_set[label],
        )

        n_train_final = len(df_train)
        n_valid_final = len(df_valid)
        print("[Data Split] %s: train=%d, valid=%d, oot=%d" % (
            label, n_train_final, n_valid_final, len(df_oot)))

        # float64 → float32 降内存
        for part_name, part_df in [("train", df_train), ("valid", df_valid), ("oot", df_oot)]:
            if part_df.empty:
                continue
            for col in feature_cols:
                if part_df[col].dtype == np.float64:
                    part_df[col] = part_df[col].astype(np.float32)
            gc.collect()

        # 保存（只保留该label需要的列）
        save_cols = list(feature_cols) + [month_col, label]

        for part_name, part_df in [("train", df_train), ("valid", df_valid), ("oot", df_oot)]:
            if part_df.empty:
                continue
            out_path = output_dir / ("dataset_%s_%s.parquet" % (label, part_name))
            part_df[save_cols].to_parquet(str(out_path), index=False)
            results["%s/%s" % (label, part_name)] = str(out_path)

        del df_train_set, df_train, df_valid
        if oot_months:
            del df_oot
        gc.collect()

    del df
    gc.collect()

    print("[Data Split] Saved %d parquet files to %s" % (len(results), output_dir))
    return results


def load_split_dataset(
    label_key: str,
    split: str,       # 'train' | 'valid' | 'oot'
    dataset_dir: str,
) -> Optional[pd.DataFrame]:
    """加载已拆好的数据集 parquet。

    Returns DataFrame or None if file doesn't exist.
    """
    from pathlib import Path as _Path

    out_path = _Path(dataset_dir) / ("dataset_%s_%s.parquet" % (label_key, split))
    if not out_path.exists():
        return None
    return pd.read_parquet(str(out_path))


def check_datasets_complete(
    label_key: str,
    dataset_dir: str,
) -> bool:
    """检查某个label的train/valid数据集是否存在且完整。"""
    from pathlib import Path as _Path

    d = _Path(dataset_dir)
    train_exists = (d / ("dataset_%s_train.parquet" % label_key)).exists()
    valid_exists = (d / ("dataset_%s_valid.parquet" % label_key)).exists()
    return train_exists and valid_exists
