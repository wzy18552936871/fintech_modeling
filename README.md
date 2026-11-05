# FinTech Risk Control Modeling Pipeline
# 金融科技风控自动化建模框架

<p align="center">
  <strong>A Modular, Manifest-Driven Framework for End-to-End Credit Risk Modeling</strong>
  <br/>
  <strong>一个模块化、配置驱动的端到端金融风控建模框架</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Model-LightGBM-green" />
  <img src="https://img.shields.io/badge/Optimization-Bayesian-orange" />
  <img src="https://img.shields.io/badge/Workflow-Automated-purple" />
  <img src="https://img.shields.io/badge/Validation-OOT-red" />
</p>

---

# English

## 1. Project Overview

**FinTech Risk Control Modeling Pipeline** is a modular, Python-based machine learning framework designed for credit risk modeling and financial risk analytics.

The project organizes the end-to-end modeling workflow into five independently executable stages:

1. Exploratory Data Analysis (EDA)
2. Feature Selection
3. Model Training
4. Model Evaluation
5. Automated Reporting

The framework integrates statistical feature analysis, LightGBM classification, Bayesian hyperparameter optimization, time-based validation, and automated report generation.

A centralized `driver.py` orchestrates the entire workflow, while `manifest.json` tracks execution status, configurations, and intermediate artifacts.

The goal is to improve modeling efficiency, reproducibility, maintainability, and workflow automation.

> **Note:** This repository contains the modeling framework and source code. Users must provide their own authorized datasets and configure the pipeline according to their use cases. No real customer data is included.

## 2. Key Features

| Module | Features |
|---|---|
| Automated EDA | Data preprocessing, missing-value analysis, label distribution, variable profiling |
| Statistical Analysis | IV, PSI, Lift, correlation analysis |
| Feature Selection | Missing-rate filtering, PSI filtering, IV-based selection, correlation deduplication |
| Model Training | LightGBM classification |
| Hyperparameter Optimization | Bayesian Optimization |
| Model Validation | Train / Validation / Out-of-Time (OOT) |
| Model Evaluation | KS, AUC, monthly performance analysis |
| Model Diagnostics | Feature importance and stability analysis |
| Automated Reporting | Excel-based evaluation reports |
| Workflow Management | Manifest-driven execution and checkpoint recovery |
| Configuration Management | YAML-based parameter and threshold management |

## 3. Pipeline Architecture

The framework consists of five major stages:

```text
                     Raw Dataset
                          |
                          v
              +------------------------+
              |  1. Exploratory Data   |
              |       Analysis         |
              |                        |
              |  - Data Preprocessing  |
              |  - Missing Rate        |
              |  - PSI Analysis        |
              |  - Correlation         |
              |  - Variable Profiling  |
              +------------------------+
                          |
                          v
              +------------------------+
              |  2. Feature Selection  |
              |                        |
              |  - IV Calculation      |
              |  - PSI Filtering       |
              |  - Missing Rate Filter |
              |  - Correlation Dedup   |
              +------------------------+
                          |
                          v
              +------------------------+
              |  3. Model Training     |
              |                        |
              |  - LightGBM            |
              |  - Bayesian Optimization|
              |  - Train / Validation  |
              |  - OOT Dataset         |
              +------------------------+
                          |
                          v
              +------------------------+
              |  4. Model Evaluation   |
              |                        |
              |  - KS / AUC             |
              |  - Monthly Evaluation  |
              |  - Feature Importance  |
              |  - Stability Analysis  |
              +------------------------+
                          |
                          v
              +------------------------+
              |  5. Automated Reporting|
              |                        |
              |  - Result Aggregation  |
              |  - Excel Report        |
              +------------------------+
```

### Workflow Orchestration

The `driver.py` module acts as the centralized entry point.

The `manifest.json` file manages:

- Project configuration
- Pipeline execution status
- Stage dependencies
- Intermediate artifacts
- Execution checkpoints

This architecture supports full-pipeline execution, individual-stage execution, and recovery after interruptions.

## 4. Technical Highlights

### 4.1 Automated Exploratory Data Analysis

The EDA module provides automated data-quality checks and statistical analysis.

Key functions include:

- Data preprocessing
- Missing-value statistics
- Target-label distribution analysis
- Variable descriptive statistics
- Feature-target cross analysis
- IV and Lift analysis
- Monthly PSI calculation
- Correlation matrix generation
- Correlation visualization

The pipeline supports both full and sampled datasets to balance analytical completeness and computational efficiency.

### 4.2 Statistical Feature Selection

The feature selection module combines multiple statistical criteria to identify useful and stable features.

Methods include:

- Information Value (IV)
- Population Stability Index (PSI)
- Missing-rate filtering
- Correlation-based deduplication
- Configurable feature-selection thresholds

Selected features are exported as CSV artifacts for subsequent model training.

### 4.3 LightGBM Modeling

The modeling module uses LightGBM for binary classification tasks.

Key capabilities:

- Gradient boosting classification
- Bayesian hyperparameter optimization
- Configurable model parameters
- Training and validation evaluation
- Model artifact persistence

Bayesian optimization helps explore model hyperparameters more systematically than manually defined parameter combinations.

### 4.4 Out-of-Time Validation

The framework supports time-based dataset splitting:

- **Train:** Model training
- **Validation:** Hyperparameter selection and model validation
- **OOT:** Out-of-time evaluation

OOT validation is used to examine whether model performance generalizes to later time periods.

Evaluation metrics include:

- KS
- AUC
- Monthly KS/AUC
- Feature importance

### 4.5 Automated Reporting

The reporting module consolidates model evaluation results and generates structured Excel reports using predefined templates.

This reduces repetitive manual reporting work and improves consistency across modeling experiments.

### 4.6 Manifest-Driven Workflow Management

The framework uses `manifest.json` to track pipeline execution.

Supported workflow operations include:

- Full pipeline execution
- Individual stage execution
- Selected-stage execution
- Execution-state tracking
- Checkpoint recovery

This design makes long-running modeling tasks easier to manage and reproduce.

## 5. Project Structure

```text
fintech-risk-control/
│
├── config/
│   ├── data_config.yaml
│   ├── feature_select_thresholds.yaml
│   ├── model_param_thresholds.yaml
│   ├── report_thresholds.yaml
│   └── result_thresholds.yaml
│
├── references/
│   └── risk_rules.md
│
├── scripts/
│   ├── driver.py
│   ├── pipeline_config.py
│   ├── data_split.py
│   ├── ensure_env.py
│   ├── lgb_fit_compat.py
│   ├── outputs.py
│   │
│   ├── step1_eda/
│   │   ├── main_step1.py
│   │   ├── eda_basic.py
│   │   ├── eda_variable_detail.py
│   │   ├── missing_rate.py
│   │   ├── psi_calc.py
│   │   └── correlation.py
│   │
│   ├── step2_feature_select/
│   │   ├── main_step2.py
│   │   ├── iv_calc.py
│   │   ├── vote_filter.py
│   │   └── corr_dedup.py
│   │
│   ├── step3_modeling/
│   │   ├── main_step3.py
│   │   └── bayesian_optimize.py
│   │
│   ├── step4_eval/
│   │   ├── main_step4.py
│   │   └── feature_importance.py
│   │
│   └── step5_report/
│
├── templates/
│   └── report_template.xlsx
│
└── SKILL.md
```

## 6. Getting Started

### Prerequisites

- Python 3.9+
- pandas
- NumPy
- scikit-learn
- LightGBM
- PyYAML
- PyArrow
- joblib
- openpyxl
- Bayesian optimization dependencies

> Dependency versions are not currently pinned in a `requirements.txt`. Please verify package compatibility with your Python environment before running the pipeline.

### Step 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>

cd fintech-risk-control
```

### Step 2. Configure the Pipeline

Review the YAML files under `config/`.

| Configuration | Description |
|---|---|
| `data_config.yaml` | Dataset and data analysis settings |
| `feature_select_thresholds.yaml` | Feature selection thresholds |
| `model_param_thresholds.yaml` | Model parameters and optimization settings |
| `result_thresholds.yaml` | Evaluation criteria |
| `report_thresholds.yaml` | Reporting configuration |

Update the configuration according to your dataset.

Important settings include:

- Target label column
- Date or month column
- Feature columns
- Modeling task definitions
- Feature selection thresholds
- Model hyperparameters
- Evaluation criteria

### Step 3. Run the Complete Pipeline

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --raw-data ./data/your_dataset.parquet \
  --label-col target \
  --month-col app_date \
  --steps all
```

Replace the example dataset path and column names with your actual configuration.

### Step 4. Run Individual Stages

Run a specific stage:

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --step 3
```

Run selected stages:

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --steps step1_eda,step3_modeling
```

Required upstream artifacts must already exist before executing an individual stage.

### Step 5. Resume an Interrupted Workflow

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --resume
```

The pipeline reads the manifest and determines which stages have already been completed.

## 7. Expected Outputs

A typical project workspace looks like this:

```text
my_risk_project/
│
├── manifest.json
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── wide_table_full.parquet
│       ├── wide_table_sampled.parquet
│       ├── selected_features*.csv
│       └── EDA and feature analysis files
│
├── models/
│   ├── final_model_*.pkl
│   └── best_params.json
│
└── reports/
    └── Evaluation and consolidated reports
```

Exact output filenames depend on the configured modeling tasks and labels.

## 8. Configuration and Customization

The framework is designed to support configurable modeling workflows.

Users can customize:

- Data preprocessing settings
- Feature selection thresholds
- IV and PSI criteria
- LightGBM hyperparameters
- Bayesian optimization settings
- Train / Validation / OOT split configuration
- Model evaluation thresholds
- Report generation settings

The configuration files allow different modeling experiments to share a consistent workflow while maintaining independent parameter settings.

## 9. Responsible Use and Data Security

- Only use datasets that you are authorized to process.
- Do not upload customer-level records or personally identifiable information.
- Keep sensitive datasets and credentials outside the public repository.
- Verify target definitions and time-based split boundaries.
- Evaluate OOT performance before interpreting model generalization.
- Check potential data leakage during feature engineering.
- Select thresholds based on actual data characteristics and business requirements.

This project is a modeling and analytical framework. It does not guarantee regulatory compliance or production deployment readiness.

## 10. Current Limitations

- No real customer dataset is included.
- A pinned `requirements.txt` is not currently provided.
- Model performance depends on data quality, label construction, and validation design.
- Production deployment and live decision-system integration are outside the scope of this repository.

## 11. License

No LICENSE file is currently included.

Please add an appropriate open-source license before allowing external users to reuse, modify, or redistribute this project.

## 12. Feedback and Contributions

Issues, suggestions, and discussions are welcome.

Feel free to open a GitHub Issue to report bugs or suggest improvements.

---

# 中文版

## 一、项目简介

**FinTech Risk Control Modeling Pipeline（金融科技风控自动化建模框架）** 是一个基于 Python 开发的模块化机器学习建模项目，主要面向金融科技及信用风险建模场景。

项目将完整的风控建模流程拆分为五个可独立执行的阶段：

1. 数据探索与分析（EDA）
2. 特征筛选（Feature Selection）
3. 模型训练（Model Training）
4. 模型评估（Model Evaluation）
5. 自动化报告生成（Automated Reporting）

框架整合了统计特征分析、LightGBM 分类建模、贝叶斯超参数优化、时间外验证以及自动化报告生成等功能。

项目以 `driver.py` 作为统一执行入口，通过 `manifest.json` 管理项目配置、任务执行状态和各阶段中间产物。

设计目标是提高风控建模工作的自动化程度、可复现性、可维护性和执行效率。

> **说明：** 本仓库主要提供风控建模流程及相关代码框架。使用者需要自行准备具有合法使用权限的数据集，并根据实际业务场景调整配置。仓库不包含真实客户数据。

## 二、核心功能

| 模块 | 功能 |
|---|---|
| 自动化数据探索 | 数据预处理、缺失率、标签分布、变量分析 |
| 统计分析 | IV、PSI、Lift、相关性分析 |
| 特征筛选 | 缺失率过滤、PSI 过滤、IV 筛选、相关性去重 |
| 模型训练 | LightGBM 分类模型 |
| 超参数优化 | Bayesian Optimization |
| 模型验证 | Train / Validation / OOT |
| 模型评估 | KS、AUC、月度表现分析 |
| 模型诊断 | 特征重要性、稳定性分析 |
| 自动化报告 | Excel 模型评估报告 |
| 流程管理 | Manifest 任务管理、断点续跑 |
| 参数管理 | YAML 配置文件及阈值管理 |

## 三、整体建模架构

项目主要由五个核心阶段构成：

```text
                     原始数据集
                          |
                          v
              +------------------------+
              |  1. 数据探索与分析     |
              |                        |
              |  - 数据预处理          |
              |  - 缺失率统计          |
              |  - PSI 分析            |
              |  - 相关性分析          |
              |  - 变量分析            |
              +------------------------+
                          |
                          v
              +------------------------+
              |  2. 特征筛选           |
              |                        |
              |  - IV 计算             |
              |  - PSI 筛选            |
              |  - 缺失率过滤          |
              |  - 相关性去重          |
              +------------------------+
                          |
                          v
              +------------------------+
              |  3. 模型训练           |
              |                        |
              |  - LightGBM             |
              |  - 贝叶斯超参数优化    |
              |  - Train / Validation  |
              |  - OOT 数据集          |
              +------------------------+
                          |
                          v
              +------------------------+
              |  4. 模型评估           |
              |                        |
              |  - KS / AUC             |
              |  - 月度表现分析        |
              |  - 特征重要性          |
              |  - 稳定性分析          |
              +------------------------+
                          |
                          v
              +------------------------+
              |  5. 自动化报告         |
              |                        |
              |  - 评估结果汇总        |
              |  - Excel 报告生成      |
              +------------------------+
```

### 流程调度与任务管理

项目通过 `driver.py` 统一管理各阶段的执行。

`manifest.json` 主要用于记录：

- 项目配置信息
- 建模流程执行状态
- 各阶段之间的依赖关系
- 中间数据及模型产物
- 任务执行进度

支持完整流程执行、指定阶段执行以及任务中断后的恢复。

## 四、核心技术亮点

### 4.1 自动化数据探索（EDA）

EDA 模块负责对原始数据进行预处理和自动化统计分析。

主要功能包括：

- 数据预处理
- 字段缺失率统计
- 目标标签分布分析
- 变量描述性统计
- 特征与标签交叉分析
- IV 和 Lift 分析
- 月度 PSI 分析
- 特征相关性矩阵
- 相关性可视化

同时支持全量宽表和抽样宽表，兼顾模型训练的数据完整性与探索性分析的计算效率。

### 4.2 多指标特征筛选

特征筛选模块结合多种统计指标和配置规则，对模型输入特征进行过滤。

主要方法包括：

- 信息值（Information Value，IV）
- 群体稳定性指标（Population Stability Index，PSI）
- 字段缺失率过滤
- 基于相关性的特征去重
- 可配置的特征筛选阈值

最终筛选结果会保存为 CSV 文件，供后续模型训练阶段使用。

### 4.3 LightGBM 模型训练

模型训练模块基于 LightGBM 实现分类模型训练。

主要功能包括：

- 梯度提升树分类建模
- 贝叶斯超参数优化
- 可配置模型参数
- 训练集与验证集评估
- 模型文件持久化

通过贝叶斯优化对模型超参数进行系统性搜索，减少完全依赖人工调整参数的工作量。

### 4.4 OOT 时间外验证

框架支持按照时间维度进行数据集划分：

- **Train：** 模型训练集
- **Validation：** 模型验证集
- **OOT：** 时间外测试集

通过 OOT 数据评估模型在后续时间区间中的泛化能力。

主要评估指标包括：

- KS
- AUC
- 月度 KS/AUC
- 特征重要性

### 4.5 自动化报告生成

报告模块整合模型评估阶段产生的结果，并通过预设 Excel 模板生成结构化报告。

减少重复性的手工整理工作，提高不同建模实验之间的报告一致性。

### 4.6 Manifest 驱动的自动化流程管理

项目通过 `manifest.json` 记录各阶段的执行状态和任务信息。

支持：

- 完整建模流程执行
- 单独阶段执行
- 指定阶段执行
- 执行状态记录
- 中断后断点续跑

通过统一调度入口，降低多阶段建模任务的管理复杂度。

## 五、项目目录结构

```text
fintech-risk-control/
│
├── config/
│   ├── data_config.yaml
│   ├── feature_select_thresholds.yaml
│   ├── model_param_thresholds.yaml
│   ├── report_thresholds.yaml
│   └── result_thresholds.yaml
│
├── references/
│   └── risk_rules.md
│
├── scripts/
│   ├── driver.py
│   ├── pipeline_config.py
│   ├── data_split.py
│   ├── ensure_env.py
│   ├── lgb_fit_compat.py
│   ├── outputs.py
│   │
│   ├── step1_eda/
│   │   ├── main_step1.py
│   │   ├── eda_basic.py
│   │   ├── eda_variable_detail.py
│   │   ├── missing_rate.py
│   │   ├── psi_calc.py
│   │   └── correlation.py
│   │
│   ├── step2_feature_select/
│   │   ├── main_step2.py
│   │   ├── iv_calc.py
│   │   ├── vote_filter.py
│   │   └── corr_dedup.py
│   │
│   ├── step3_modeling/
│   │   ├── main_step3.py
│   │   └── bayesian_optimize.py
│   │
│   ├── step4_eval/
│   │   ├── main_step4.py
│   │   └── feature_importance.py
│   │
│   └── step5_report/
│
├── templates/
│   └── report_template.xlsx
│
└── SKILL.md
```

## 六、环境准备与快速开始

### 环境要求

- Python 3.9+
- pandas
- NumPy
- scikit-learn
- LightGBM
- PyYAML
- PyArrow
- joblib
- openpyxl
- 贝叶斯优化相关依赖

> 当前仓库尚未提供固定版本的 `requirements.txt`，正式运行前请检查脚本中的实际依赖，并根据 Python 环境安装兼容版本。

### 第一步：克隆项目

```bash
git clone <YOUR_REPOSITORY_URL>

cd fintech-risk-control
```

### 第二步：修改配置文件

进入 `config/` 目录，根据实际数据集调整配置。

| 配置文件 | 主要用途 |
|---|---|
| `data_config.yaml` | 数据集及数据分析配置 |
| `feature_select_thresholds.yaml` | 特征筛选阈值 |
| `model_param_thresholds.yaml` | 模型参数与优化设置 |
| `result_thresholds.yaml` | 模型评估及结果标准 |
| `report_thresholds.yaml` | 报告生成配置 |

重点检查以下内容：

- 目标标签字段
- 日期或月份字段
- 特征字段
- 建模任务及标签定义
- 特征筛选阈值
- LightGBM 模型参数
- 模型评估标准

### 第三步：运行完整建模流程

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --raw-data ./data/your_dataset.parquet \
  --label-col target \
  --month-col app_date \
  --steps all
```

请根据实际数据替换示例数据路径及字段名称。

### 第四步：运行指定阶段

运行单独阶段：

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --step 3
```

运行指定阶段组合：

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --steps step1_eda,step3_modeling
```

注意：单独运行某个阶段时，需要确保对应的上游中间产物已经存在。

### 第五步：断点续跑

```bash
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --resume
```

程序会读取 Manifest 中记录的执行状态，判断已经完成的阶段，并从可继续执行的位置恢复工作流。

## 七、预期输出结果

项目运行后，会在指定工作目录中生成 Manifest、中间数据、模型文件和评估报告。

典型目录结构：

```text
my_risk_project/
│
├── manifest.json
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── wide_table_full.parquet
│       ├── wide_table_sampled.parquet
│       ├── selected_features*.csv
│       └── EDA 与特征分析文件
│
├── models/
│   ├── final_model_*.pkl
│   └── best_params.json
│
└── reports/
    └── 模型评估及汇总报告
```

具体文件名会根据实际配置的建模任务及标签有所不同。

## 八、配置与扩展

框架采用配置驱动的设计，支持根据不同的数据集和建模任务调整：

- 数据预处理参数
- 特征筛选阈值
- IV 和 PSI 标准
- LightGBM 超参数
- 贝叶斯优化参数
- Train / Validation / OOT 时间切分
- 模型评估标准
- 报告生成参数

不同建模实验可以使用统一的执行流程，同时保留各自独立的参数配置。

## 九、数据安全与使用规范

- 仅使用具有合法使用权限的数据集。
- 不要上传客户级明细或个人身份信息。
- 不要将敏感数据、账号凭据提交到公开仓库。
- 在模型评估前确认标签定义及时间切分边界。
- 检查特征工程过程中是否存在数据泄漏。
- 通过 OOT 表现检验模型跨时间泛化能力。
- 特征筛选及模型评估阈值应根据实际数据和业务要求确定。

本项目属于分析与建模框架，不保证自动满足监管合规要求，也不代表模型已经完成生产环境部署。

## 十、当前项目限制

- 不提供真实客户样本数据。
- 当前未提供固定版本的 `requirements.txt`。
- 模型表现取决于数据质量、标签构造和验证集设计。
- 生产环境部署及实时决策系统集成不属于当前项目的实现范围。

## 十一、开源许可

当前仓库尚未包含 LICENSE 文件。

如果希望允许其他开发者使用、修改或再分发项目代码，请先添加合适的开源许可证。

## 十二、反馈与交流

欢迎通过 GitHub Issues 反馈问题、提出改进建议或参与技术交流。

---

<p align="center">
  <strong>FinTech Risk Control Modeling Pipeline</strong>
  <br/>
  Built with Python and Machine Learning.
  <br/>
  <strong>让风控建模流程更加模块化、自动化与可复现。</strong>
</p>
