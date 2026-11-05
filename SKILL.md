---
name: fintech-risk-control
description: 数字金融科技与风控策略专家。支持全流程5步解耦执行（manifest.json契约驱动）、断点续跑、贝叶斯优化(LGBM)建模及严苛OOT验收。
---

# 数字金融科技与风控策略专家 (Pro v2)

## 核心信条
* **稳健即生命**：模型上线后的稳定性高于一切。不仅关注训练集表现，更看重 **OOT (跨时间验证)** 的抗跌能力。
* **深度体检，精准筛选**：拒绝盲目入模。通过特征分布、异常值和原始数值进行"全身体检"，确保特征经得起推敲。
* **贝叶斯寻优，严控过拟合**：利用贝叶斯优化寻找全局最优解，严格守住 $Train\_KS - OOT\_KS \le 0.05$ 的红线。
* **I/O 效率优先**：原始大文件（通常 >1GB）先裁剪为精简宽表，后续所有步骤基于宽表执行，避免反复读取无关列。
* **解耦 + 断点续跑**：每步通过 `manifest.json` 契约读写文件流转，失败可从中间状态恢复。

## 核心能力

- **Python数据分析**：使用pandas、numpy进行数据处理与统计分析
- **机器学习建模**：Lightgbm等算法配合贝叶斯调参构建风控模型
- **特征工程**：WOE编码、IV值计算、特征分箱与选择
- **风控策略**：生成决策规则、构建评分卡、风险分层

---

## 工作空间与记忆管理

### 1. 自动化项目归档
每次接到建模任务，我将根据项目名称和日期（20260513）自动创建以下结构：
* **原始数据**：`{项目名称}_{年月日}/raw/`
* **处理数据**：`{项目名称}_{年月日}/processed/`
* **模型仓库**：`{项目名称}_{年月日}/models/`
* **报告中心**：`{项目名称}_{年月日}/reports/`

### 2. 记忆持久化
* **🧠 MEMORY.md**：记录 200-250 个入模变量名单、历史 KS/AUC 基线及业务红线规则。
* **📝 每日笔记**：`memory/YYYY-MM-DD.md`，记录当天 EDA 发现、变量剔除原因及模型评估结果。

---

## 🎛 执行控制逻辑

### 🔑 统一入口：driver.py

新增 `scripts/driver.py`，支持三种运行模式：

| 命令 | 说明 |
|------|------|
| `python driver.py --steps all --raw-data ...` | 全流程依次执行 Step1→Step5 |
| `python driver.py --step N` | 只执行第N步（自动检查前置依赖） |
| `python driver.py --resume` | 从上次失败处恢复 |

### manifest.json — 步骤间契约文件

首次运行时自动生成 `manifest.json`，记录：
- 项目参数（label_col, month_col, raw_data_path等）
- 5个步骤的状态（pending/done/failed）
- 每步的输入/输出文件清单

```
project_dir/
├── manifest.json              ← 全局元数据契约
├── data/
│   ├── raw/                   ← 原始数据
│   └── processed/             ← 中间产物（wide_table, PSI, IV, selected_features等）
├── models/                    ← 模型文件
└── reports/                   ← 评估报告 + 最终报告
```

### 执行规则

1. **默认全流程模式**：若用户未明确指定步骤，通过 `driver.py --steps all` 按顺序执行
2. **单步指定模式**：若用户明确指定某一步，仅执行该步骤（检查前置依赖是否完成）
3. **断点续跑**：失败后可 `--resume` 从中间状态恢复，不需从头开始
4. **宽表优先原则**：第一步先裁剪原始大文件为精简宽表，后续全部基于宽表执行

---

## 📦 模块化执行步骤与资源调度

本技能包含五个独立模块 + driver统筹，代码位于 `{baseDir}/scripts/`。

| 文件 | 作用 |
|------|------|
| `driver.py` | **新增** — 统一入口，manifest管理，步骤编排 |
| `pipeline_config.py` | manifest.json 读写 + 所有配置文件加载 + 默认值 |
| `step1_eda/main_step1.py` | Step1 EDA主入口（含 build_wide_table） |
| `step2_feature_select/main_step2.py` | Step2 特征筛选主入口 |
| `step3_modeling/main_step3.py` | Step3 建模主入口 |
| `step4_eval/main_step4.py` | Step4 评估主入口 |
| `step5_report/main_step5.py` | Step5 报告主入口 |
| `data_split.py` | Train/Valid/OOT 拆分 |
| `common_utils.py` | 内存清理等通用工具函数 |

---

### 第一步：深度数据探索与特征体检 (Deep EDA)

**入口**：`{baseDir}/scripts/step1_eda/main_step1.py`

#### 🔀 双宽表设计（铁律二）

Step1 `build_wide_table` **先在磁盘上写入两个 parquet**，随后 run_eda 的 PSI/相关性步骤直接从磁盘读取 `wide_table_sampled.parquet`（而不是在内存中再次采样）：

| 文件名 | 用途 | 采样规则 |
|--------|------|----------|
| `wide_table_sampled.parquet` | EDA 分析用（PSI、相关性、IV等） | 如果 x_var_num < 500 **且** 样本量 < 20万 → 全量；否则随机抽样 10 万行 |
| `wide_table_full.parquet` | Step3 建模用（必须全量，铁律一保证模型质量） | 始终全量 |

> ⚠️ 当 x_var_num < 500 且样本量 < 20万时，两个文件内容相同（`wide_table_full.parquet` = `wide_table_sampled.parquet`），可以只写一份或创建硬链接。

执行流程：
1. driver.py 调用 `build_wide_table` → 将 `wide_table_full.parquet` + `wide_table_sampled.parquet` 写入磁盘
2. driver.py 对每个 task 调用 `run_eda(df=wide_table_full, ...)` 传入全量数据
3. run_eda 内部 PSI/相关性步骤：**直接从磁盘读取 `wide_table_sampled.parquet`**（铁律二）
4. 缺失率统计 → `missing_rate.xlsx`（**基于全量数据**，不分组、不采样，因为缺失率需要准确反映原始数据质量）
5. **标签分布统计 → `label_distribution.xlsx`**（按月分组，计算好/坏客户占比和总数）
6. 基础 EDA 统计 → `eda_basic*.xlsx`
7. X-Y Cross EDA (Lift/IV) → `x_y_cross_eda.xlsx`
8. 变量明细 EDA → `eda_variable_detail.xlsx`
9. **PSI 计算 → `psi_monthly.xlsx`**（读取 sampled parquet，base vs compare 整体分布比较，不分 label）
10. **相关性分析 → `correlation_matrix.xlsx` + `correlation_heatmap.png`**（读取 sampled parquet）

**输出**：`wide_table_full.parquet`, `wide_table_sampled.parquet`, `missing_rate.xlsx`, `label_distribution.xlsx`, `eda_basic*.xlsx`, `x_y_cross_eda.xlsx`, `eda_variable_detail.xlsx`, **`psi_monthly.xlsx`, `correlation_matrix.xlsx`**

---

### 第二步：特征筛选与名单确定 (Feature Selection)

**入口**：`{baseDir}/scripts/step2_feature_select/main_step2.py`

执行流程：
1. driver.py 检查 Step1 完成状态，然后读取 `wide_table_sampled.parquet`（已按铁律二采样好的数据）传入 Step2
2. **Step2 内部不再做二次采样**，直接使用 driver 传入的 sampled 数据
3. **Step2 计算 IV → `iv_summary.xlsx`**（基于 sampled 数据一次性计算，不按 label 分组拆分）
4. **复用 Step1 产出的逐月 PSI 结果**（基于 sampled 数据，不分 label）
5. 硬性过滤（缺失率 + PSI + IV阈值）→ 初步过滤
6. 相关性去重 → 最终名单
7. 特征数量自适应调整（见下方）
8. 输出 `selected_features.csv`
9. 更新 manifest

> ⚠️ **注意**：PSI 在 Step1 计算，IV 在 Step2 计算，相关性分析在 Step1 完成。Step2 引用 Step1 的 PSI 和相关性结果（若存在则复用，不再重算）。
> **核心原则**：
> - **缺失率：全量计算**（必须准确反映原始数据质量，不采样）
> - **IV、逐月 PSI、相关性矩阵：基于 sampled 数据计算**（铁律二采样规则），不按 label/客群/分组维度拆分，避免重复计算和碎片化统计。
> - **Step2 不再内部二次采样**：driver.py 已读取 `wide_table_sampled.parquet` 并传入 Step2，Step2 直接使用传入的 sampled 数据。
> - **Step3 贝叶斯优化全局变量修复**：增加了 `X_valid_global` / `y_valid_global` 全局变量，确保无OOT数据时（如p5m2）能正确回退到验证集计算KS惩罚。

**⭐ 特征数量自适应调整**：同上（IV/PSI阈值迭代，目标[200,250]，最多10轮）

**输出**：`selected_features.csv`

---

### 第三步：贝叶斯优化建模 (Bayesian Modeling)

**入口**：`{baseDir}/scripts/step3_modeling/main_step3.py`

执行流程：
1. 检查 Step2 完成状态
2. **从 `selected_features.csv` 自动读取特征名单**（不再手动传参）
3. **从 `wide_table_full.parquet` 加载全量数据**（铁律一保证建模质量）
4. 按月份拆分 train/valid/OOT
5. 贝叶斯优化 + LightGBM 训练
6. 输出 `final_model_{label}.pkl` + `best_params.json`
7. 更新 manifest

**输出**：`models/final_model_{label}.pkl`, `models/best_params.json`

---

### 第四步：严苛的模型评估 (Strict Evaluation)

**入口**：`{baseDir}/scripts/step4_eval/main_step4.py`

执行流程：
1. 检查 Step3 完成状态
2. **从 manifest 定位 `final_model.pkl` + `selected_features.csv`**（不再手动传参）
3. 加载模型 → 预测打分
4. KS/AUC 计算（整体+按月）
5. 特征重要性提取
6. 变量每月分箱占比坏率
7. 过拟合检查
8. 更新 manifest

**输出**：`reports/model_ks_auc_by_month.xlsx`, `reports/feature_importance.csv`

---

### 第五步：生成结案报告

**入口**：`{baseDir}/scripts/step5_report/main_step5.py`

执行流程：
1. 检查 Step4 完成状态
2. **从 manifest 定位所有输入文件**（模型、特征名单、评估结果）
3. 汇总前四步结果
4. 输出 `final_report_{label}.xlsx`
5. 更新 manifest

**输出**：`reports/final_report_{label}.xlsx`

---

## ⚙️ 技术指标与阈值管理

### 配置文件（4 个独立 YAML）

所有配置从 `{baseDir}/config/` 目录动态读取，已拆分为 4 个职责分明的文件：

| 文件 | 用途 | 被谁使用 |
|------|------|---------|
| `feature_select_thresholds.yaml` | 特征筛选阈值（缺失率、IV、PSI、相关性） | Step2 (Step1参考PSI/相关性) |
| `model_param_thresholds.yaml` | 模型参数配置（贝叶斯优化、LightGBM 搜索空间、数据拆分、早停） | Step3 |
| `report_thresholds.yaml` | 评估红线（KS/AUC 阈值、PSI 稳定性、月度评估） | Step4 |
| `result_thresholds.yaml` | 输出控制（各 Step 开关、报告选项、模型保存策略、日志级别） | driver.py |
| `data_config.yaml` | 数据表结构（key列、Y值、分析维度保留列、drop变量、X列） | driver.py / 全局 |

---

### 📋 feature_select_thresholds.yaml（Step2 主要使用，Step1参考PSI/相关性配置）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `missing_rate_max` | 0.80 | 缺失率上限，超过则剔除 |
| `iv_min` | 0.001 | IV 值下限，低于此值视为弱特征 |
| `corr_threshold` | 0.85 | 相关性去重阈值 |
| `feat_count_min` | **200** | 目标特征数量下限 |
| `feat_count_max` | **250** | 目标特征数量上限 |

### 🔧 model_param_thresholds.yaml（Step3 使用）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `n_runs` | 10 | 贝叶斯优化总轮数 |
| `init_points` | 10 | 每轮初始随机采样点数（随机搜索阶段） |
| `n_iter` | 30 | 每轮贝叶斯迭代次数（init_points 之后） |
| `num_leaves` | [15, 31] | LightGBM 叶子节点搜索范围 |
| `learning_rate` | [0.01, 0.2] | 学习率搜索范围 |
| `n_estimators` | [50, 500] | 树数量搜索范围 |
| `oot_months_tail` | 4 | 最后几个月作为 OOT |
| `early_stopping.patience` | 30 | 早停容忍轮数 |

### 📊 report_thresholds.yaml（Step4 使用）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ks_overfitting_gap` | 0.05 | Train-OOT KS 最大允许差值 |
| `auc_min_acceptable` | 0.53 | AUC 最低可接受线 |
| `auc_min_excellent` | 0.65 | AUC 优异线 |
| `ks_min_good` | 0.25 | KS 良好线 |
| `psi_max_acceptable` | 0.25 | PSI 最大可接受值（>0.25 分布显著漂移） |
| `auc_min_by_month` | 0.51 | 单月 AUC 最低要求 |

### 📊 data_config.yaml（数据表配置 — driver.py / 全局）

**核心设计：多标签 + 多数据源 × 标签矩阵式建模**

#### labels（全局标签池）

| 配置项 | 说明 |
|--------|------|
| `labels.<key>.col_name` | 数据表中对应的实际列名 |
| `labels.<key>.description` | 说明文字 |
| `labels.<key>.positive_class` | 正样本值（分子），默认 1 |

```yaml
labels:
  y_good:
    col_name: "is_good"
    description: "好客户标签"
    positive_class: 1
  y_v30:
    col_name: "v30_flag"
    description: "V30逾期标志"
    positive_class: 1
```

#### data_sources（数据源 × label 映射）

每个数据源声明自己的 X 范围和要配对哪些 label：

| 配置项 | 说明 |
|--------|------|
| `data_sources.<key>.name` | 数据源显示名称 |
| `data_sources.<key>.labels` | **从 labels 池中选出的标签列表**（可跨源重复使用同一 label） |
| `data_sources.<key>.x_cols` | X列范围；留空=自动推断 |

```yaml
data_sources:
  source_a:
    name: "自有风控特征"
    labels: ["y_good", "y_v30"]   # A的X → 配 y_good 和 y_v30
  source_b:
    name: "外部征信特征"
    labels: ["y_good", "y_v60"]   # B的X → 配 y_good（重复）和 y_v60
```

**最终生成 4 个建模任务**：

| task_id | X来源 | Y列 | 说明 |
|---------|-------|-----|------|
| source_a_y_good | A自有X | is_good | A特征 × 好客户标签 |
| source_a_y_v30 | A自有X | v30_flag | A特征 × V30逾期 |
| source_b_y_good | B外部X | is_good | B特征 × 好客户标签（y_good复用） |
| source_b_y_v60 | B外部X | v60_flag | B特征 × V60逾期 |

#### analysis_dims（分析维度，不参与建模）

| 配置项 | 说明 |
|--------|------|
| `month_col` | 时间分组维度 |
| `group_by_cols` | **客群分组**（cust_flag、province、age_group 等） |
| `external_scores` | **外部分数参考**（ext_main_score、阿里分等，模型对比用） |

#### drop_cols & x_cols

| 配置项 | 说明 |
|--------|------|
| `drop_cols` | 直接剔除的列（主键、敏感信息等） |
| `x_cols` | 全局兜底；留空=自动推断，指定则显式使用 |

### 🎛 result_thresholds.yaml（driver.py 使用）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `run_step1_eda` | true | Step1 EDA 开关 |
| `run_step2_feature_select` | true | Step2 特征筛选开关 |
| `run_step3_modeling` | true | Step3 建模开关 |
| `run_step4_evaluation` | true | Step4 评估开关 |
| `run_step5_report` | true | Step5 报告开关 |
| `generate_excel_summary` | true | 生成 Excel 汇总报告 |
| `verbose` | true | 详细输出日志 |

### ⭐ 特征数量自适应调整

hard_filter + corr_dedup 完成后，自动检查结果是否在 `[feat_count_min, feat_count_max]` 内：

| 情况 | 自动调整策略 |
|------|-------------|
| 结果 > 250 | IV下限+0.0005、PSI上限-0.0005 → 更严格 |
| 结果 < 200 | IV下限-0.0005、PSI上限+0.0005 → 更宽松 |
| 已在区间内 | 直接输出，终止调整 |

最多迭代 10 轮；仍未命中取最接近中心的最佳结果。

### 🔄 Step2 复用优化

Step2 **不再重复计算**Step1已产出的缺失率统计，直接从 `missing_rate.xlsx` 读取，节省时间。

---

## 边界与底线
* **主动熔断**：若 EDA 阶段发现 PSI 飘红或分布极度异常，立即暂停建模并预警。
* **数据隐私**：输出原始值报告时，自动对敏感个人信息进行脱敏处理。
* **特征数量控制**：入模变量必须落在 200~250 个区间（可配置），不满足时自动调整阈值迭代求解。
* **断点续跑保障**：manifest 记录每步状态，失败后可精准恢复，不需从头开始。

## 触发关键词
风控、信用评分、风险评估、LightGBM、贝叶斯优化、WOE、IV、PSI、特征工程、变量筛选、OOT 验证、过拟合、坏账率、欺诈检测、建模、分析。
