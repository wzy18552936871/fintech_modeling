FinTech Risk Control Modeling Pipeline
<p align="center">
  <strong>A modular, manifest-driven framework for end-to-end credit risk modeling.</strong>
</p>A Python-based risk modeling pipeline that organizes the modeling workflow into five independently executable stages: exploratory data analysis, feature selection, LightGBM modeling, out-of-time evaluation, and automated reporting. It is designed to make experiments reproducible, model validation more rigorous, and long-running workflows easier to resume.
Note: This repository provides the modeling workflow and code structure. You must supply your own authorized dataset and project-specific configuration. No real customer data is included.

Key Features
- Automated EDA: Data preprocessing, missing-value analysis, label distribution analysis, variable profiling, correlation analysis, and monthly Population Stability Index (PSI) analysis.
- Feature selection: IV calculation, missing-rate and PSI filtering, correlation-based deduplication, and configurable threshold-based selection.
- Model training: LightGBM classification with Bayesian hyperparameter optimization and configurable training iterations.
- Time-based validation: Train/validation/OOT data splitting and evaluation across overall and monthly KS/AUC metrics.
- Model diagnostics: Feature importance analysis and stability-oriented evaluation outputs.
- Automated reporting: Consolidates model evaluation results and generates structured Excel-based reports.
- Manifest-driven orchestration: Tracks project configuration, step status, and input/output artifacts through manifest.json.
- Flexible execution: Run the complete pipeline, selected stages, or resume a failed workflow from its last recorded state.
- Configurable workflow: Data, feature-selection, model, result, and reporting thresholds are maintained in YAML configuration files.
Pipeline Overview
                 Raw Dataset
                      |
                      v
          +------------------------+
          |  1. Exploratory Data   |
          |       Analysis         |
          | EDA / Missing / PSI    |
          | Correlation / Profiling|
          +------------------------+
                      |
                      v
          +------------------------+
          |  2. Feature Selection  |
          | IV / PSI / Missing Rate|
          | Correlation Dedup      |
          +------------------------+
                      |
                      v
          +------------------------+
          |  3. Model Training     |
          | LightGBM               |
          | Bayesian Optimization |
          | Train / Valid / OOT    |
          +------------------------+
                      |
                      v
          +------------------------+
          |  4. Model Evaluation   |
          | KS / AUC / Feature     |
          | Importance / Stability|
          +------------------------+
                      |
                      v
          +------------------------+
          |  5. Automated Reporting|
          | Consolidated Reports   |
          +------------------------+
Each stage reads the artifacts produced by previous stages. The manifest.json file records the workflow state and helps the driver validate dependencies before executing a stage.
Project Structure
fintech-risk-control/
├── config/
│   ├── data_config.yaml
│   ├── feature_select_thresholds.yaml
│   ├── model_param_thresholds.yaml
│   ├── report_thresholds.yaml
│   └── result_thresholds.yaml
├── references/
│   └── risk_rules.md
├── scripts/
│   ├── driver.py                  # Unified pipeline entry point
│   ├── pipeline_config.py         # Configuration and manifest utilities
│   ├── data_split.py              # Time-based dataset splitting
│   ├── outputs.py                 # Output configuration utilities
│   ├── step1_eda/                 # Exploratory data analysis
│   ├── step2_feature_select/      # Feature selection
│   ├── step3_modeling/            # LightGBM training and optimization
│   ├── step4_eval/                # Model evaluation
│   └── step5_report/              # Report generation
├── templates/
│   └── report_template.xlsx
└── SKILL.md
Workflow Details
1. Exploratory Data Analysis
The EDA stage prepares a modeling dataset and generates data-quality and exploratory outputs, including:
- Missing-rate statistics
- Label distribution summaries
- Variable-level descriptive analysis
- Feature/label cross analysis, including IV and Lift-related analysis
- Monthly PSI analysis
- Correlation matrix and correlation visualization
The workflow supports separate full and sampled wide-table artifacts. The full dataset is retained for model training, while sampling can be used for computationally intensive exploratory calculations.
2. Feature Selection
The feature-selection stage combines configurable filters and statistical analysis, including:
- Information Value (IV)
- Population Stability Index (PSI)
- Missing-rate thresholds
- Correlation-based feature deduplication
- Configurable threshold and feature-count rules
The selected feature list is saved as a CSV artifact and consumed by the modeling stage.
3. Model Training
The modeling stage trains LightGBM classification models and supports Bayesian hyperparameter optimization. Data is split by time into training, validation, and out-of-time (OOT) datasets where the configured data permits.

Model artifacts and optimization parameters are saved for subsequent evaluation.
4. Model Evaluation
The evaluation stage generates model predictions and evaluates discrimination and stability. Metrics include overall and monthly KS/AUC, together with feature-importance outputs. OOT performance is used to assess generalization to later time periods.
5. Automated Reporting
The final stage consolidates evaluation outputs and produces structured reports using the report-generation modules and Excel template.
Getting Started
Prerequisites
- Python 3.9+ is recommended; the exact compatible version depends on the dependency versions used in your environment.
- A dataset in a format supported by the data-loading code (for example, Parquet).
- Sufficient memory and disk space for your dataset and intermediate artifacts.
The project does not currently include a pinned requirements.txt. Review the imports in the scripts and install compatible versions of the required packages before running the pipeline. The codebase uses packages from the Python data-science and machine-learning ecosystem, including pandas, NumPy, scikit-learn, LightGBM, joblib, PyYAML, PyArrow, openpyxl, and the Bayesian-optimization package used by the modeling module.
1. Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd fintech-risk-control
2. Configure your project
Review the YAML files in config/ and adapt the data, feature-selection, model, result, and reporting settings to your use case.

Make sure the project configuration identifies the correct:
- Target/label column
- Date or month column used for time-based splitting
- Feature columns, if you want to specify them explicitly
- Modeling tasks and label definitions
The checked-in config/data_config.yaml is a template. Update it for your dataset before execution.
3. Run the complete pipeline
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --raw-data ./data/your_dataset.parquet \
  --label-col target \
  --month-col app_date \
  --steps all
Replace the example paths and column names with values from your own dataset. The first run initializes the project workspace and its manifest.json file.
Execution Options
Run the complete workflow
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --raw-data ./data/your_dataset.parquet \
  --label-col target \
  --month-col app_date \
  --steps all
Run a single stage
# Run stage 3: model training
python scripts/driver.py --project-dir ./projects/my_risk_project --step 3
A single stage requires its upstream artifacts to be available. The driver checks the recorded workflow state and dependencies.
Run selected stages
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --steps step1_eda,step3_modeling
Use this mode only when the required intermediate artifacts for the selected stages already exist.
Resume after a failure
python scripts/driver.py --project-dir ./projects/my_risk_project --resume
The driver uses the project manifest to determine which stages have completed and where execution should continue.
Specify feature columns
python scripts/driver.py \
  --project-dir ./projects/my_risk_project \
  --raw-data ./data/your_dataset.parquet \
  --label-col target \
  --month-col app_date \
  --feature-cols feature_a feature_b feature_c \
  --steps all
Outputs
The pipeline creates a project workspace containing a manifest, intermediate data, model artifacts, and reports. A typical layout is:
my_risk_project/
├── manifest.json
├── data/
│   ├── raw/
│   └── processed/
│       ├── wide_table_full.parquet
│       ├── wide_table_sampled.parquet
│       ├── selected_features*.csv
│       └── EDA and feature-analysis files
├── models/
│   ├── final_model_*.pkl
│   └── best_params.json
└── reports/
    └── evaluation and consolidated reports
Exact filenames depend on the configured modeling tasks and labels.
Configuration
The config/ directory contains YAML templates for the main parts of the workflow:
File	Purpose
data_config.yaml	Dataset, label, and analysis configuration
feature_select_thresholds.yaml	Feature filtering and selection thresholds
model_param_thresholds.yaml	Model parameter and optimization settings
result_thresholds.yaml	Evaluation and result criteria
report_thresholds.yaml	Report generation settings


Review these files before running the pipeline. Thresholds should be selected based on the dataset, business context, and validation requirements rather than copied blindly.
Reproducibility and Responsible Use
- Use only data that you are authorized to process.
- Do not commit personal information, customer-level records, credentials, or proprietary datasets.
- Keep raw data and generated model artifacts outside the public repository unless they are explicitly safe to publish.
- Validate time columns, label definitions, positive-class conventions, and split boundaries before interpreting model metrics.
- Treat KS/AUC as model discrimination metrics; use PSI and data-quality analyses to investigate population and feature distribution stability.
- Review OOT performance and potential data leakage before considering a model suitable for business use.
- This repository is an analytical modeling framework, not a guarantee of regulatory compliance or production readiness.
Limitations
- No sample customer dataset is distributed with this repository.
- Dependency versions are not pinned in a requirements file.
- Results depend on data quality, label construction, time-split design, and configuration choices.
- The pipeline does not, by itself, establish that a model has been deployed to a production decision system.
License
No license file is currently included. Add a license before permitting reuse, modification, or redistribution by others.
