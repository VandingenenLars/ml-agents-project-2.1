## Data Pipeline Overview

This data pipeline transforms metrics collected during Unity ML-Agents training into machine learning ready feature datasets.  

The data goes through four main stages:  
**Collection -> Processing -> Feature Extraction -> Visualization**

### Stage 1 - Data Collection

**Scripts:**
- `utils/system_metrics_collector.py` - Responsible for monitoring system metrics
- `utils/training_metrics_collector.py` - Responsible for monitoring training metrics
- `training/data_collector.py` - Collect all necessary data using the helper classes and create raw JSON data files.

**Inputs:**
- Live Unity ML-Agents training session
- Hardware performance data (CPU, GPU, RAM)

**Outputs:**  
Timestamped directory under `data/raw`, containing:  

| File                    | Description                                                                                                        |  
|-------------------------|--------------------------------------------------------------------------------------------------------------------|  
| `system_metrics.json`   | Timestamped entries of CPU, RAM and GPU utilization sampled every few seconds                                      |
| `training_metrics.json` | Training progress logs from ML-Agents (steps, reward, loss, etc.) sampled based on Tensorboard's logging frequency |
| `config.json`           | Training configuration parameters (algorithm, learning rate, network size, etc.)                                   |

Each folder represents one training run, e.g.:  
`data/raw/run_2025-11-10_12-00-00`

### Stage 2 - Data Processing  

**Scripts**
- `utils/data_processor.py` - Convert multiple raw JSON logs into a single, clean .csv file suitable for feature extraction.

**Tasks performed:**
- Merge `system_metrics.json` and `training_metrics.json` for each run.
- Compute derived metrics such as:
  - Average and peak RAM usage
  - Average CPU and GPU utilization
  - Total training time to target reward
  - Iterations to target reward
- Validate data types and ensure consistent units according to `schema.json`
- Store results in a single `csv` file

**Schema:**
`schema.json`         | Field definitions (types, expected ranges, units, examples and descriptions).
**Inputs:**
- `system_metrics.json`
- `training_metrics.json`
- `config.json`

**Outputs:**

| File                  | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| `processed_data.csv`  | Clean dataset where each row represents one training session                  |

### Stage 3 - Feature Engineering

**Scripts**
- `utils/feature_extractor.py` - Encode, scale and extract all necessary features from `processed_data.csv` and output a feature matrix.

**Tasks performed:**
- Load `processed_data.csv`
- Separate features (X) from targets (y)
- Remove non-training identifiers (e.g `run_id`) while preserving them for analysis.
- Encode categorical variables using one-hot encoding:
  - `game_type`
  - `algorithm`
- Standardize numerical features
- Performed stratified train/test split to preserve balance for `reached_threshold`
- Export datasets to `delivarables/data/features`

**Inputs:**
| File                  | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| `processed_data.csv`  | Aggregated and validated training dataset

**Outputs:**
Stored under `deiverables/data/features/`
| File                  | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| `X_train.csv` | Training feature matrix
| `X_test.csv` | Test feature matrix
| `y_train.csv` | Training target variables
| `y_test.csv` | Test target variables
| `run_ids_train.csv` | Run identifiers for training samples
| `run_ids_test.csv` | Run identifiers for test samples

These outputs are ready to be used by our supervised machine learning models






