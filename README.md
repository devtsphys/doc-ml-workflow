# # Azure ML Time Series Forecasting with Random Forest - Complete Reference

## 1. Environment Setup & Authentication

### Install Required Packages

```bash
pip install azureml-sdk pandas numpy scikit-learn matplotlib seaborn
pip install azureml-train-automl-client azureml-widgets
```

### Authentication & Workspace Connection

```python
from azureml.core import Workspace, Environment, Experiment, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.sklearn import SKLearn
from azureml.core.runconfig import RunConfiguration

# Authentication
auth = InteractiveLoginAuthentication(tenant_id="your-tenant-id")

# Connect to workspace
ws = Workspace.from_config(auth=auth)
# OR create new workspace
ws = Workspace.create(
    name="ml-workspace",
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    location="eastus"
)
```

## 2. Data Preparation & Feature Engineering

### Load and Prepare Time Series Data

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data
df = pd.read_csv('timeseries_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

# Basic time series features
def create_time_features(df, target_col):
    """Create comprehensive time-based features"""
    df_features = df.copy()
    
    # Temporal features
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['day'] = df_features.index.day
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['quarter'] = df_features.index.quarter
    df_features['is_weekend'] = df_features.index.dayofweek.isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)
    
    return df_features

# Lag features for time series
def create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30]):
    """Create lagged features"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    windows = [3, 7, 14, 30]
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
    
    return df

# Apply feature engineering
df_processed = create_time_features(df, 'target_value')
df_processed = create_lag_features(df_processed, 'target_value')
df_processed = df_processed.dropna()
```

### Upload Data to Azure ML

```python
from azureml.core import Datastore, Dataset

# Get default datastore
datastore = ws.get_default_datastore()

# Upload data
datastore.upload_files(
    files=['./processed_data.csv'],
    target_path='timeseries-data/',
    overwrite=True
)

# Create dataset
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'timeseries-data/processed_data.csv')
)
dataset = dataset.register(
    workspace=ws,
    name='timeseries-dataset',
    description='Time series data with engineered features'
)
```

## 3. Model Development

### Random Forest Time Series Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class TimeSeriesRandomForest:
    def __init__(self, **rf_params):
        self.model = RandomForestRegressor(**rf_params)
        self.feature_importance_ = None
        
    def prepare_data(self, df, target_col, test_size=0.2):
        """Prepare data for time series modeling"""
        # Sort by date
        df_sorted = df.sort_index()
        
        # Split features and target
        feature_cols = [col for col in df_sorted.columns if col != target_col]
        X = df_sorted[feature_cols]
        y = df_sorted[target_col]
        
        # Time-based split
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, cv_folds=5):
        """Train with time series cross-validation"""
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=tscv, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.feature_importance_ = self.model.feature_importances_
        
        return grid_search.best_params_
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred

# Usage example
ts_model = TimeSeriesRandomForest(random_state=42)
X_train, X_test, y_train, y_test = ts_model.prepare_data(df_processed, 'target_value')
best_params = ts_model.train(X_train, y_train)
metrics, predictions = ts_model.evaluate(X_test, y_test)
```

## 4. Azure ML Experiment & Training

### Create Compute Cluster

```python
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "cpu-cluster"
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS3_V2",
        min_nodes=0,
        max_nodes=4
    )
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
```

### Training Script (train.py)

```python
# train.py
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from azureml.core import Run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    
    args = parser.parse_args()
    run = Run.get_context()
    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target_value', 'date']]
    X = df[feature_cols]
    y = df['target_value']
    
    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Log metrics
    run.log('mae', mae)
    run.log('rmse', rmse)
    
    # Save model
    joblib.dump(model, 'outputs/model.pkl')
    
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')

if __name__ == '__main__':
    main()
```

### Submit Training Job

```python
from azureml.core import ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create environment
env = Environment(name="sklearn-env")
conda_dep = CondaDependencies()
conda_dep.add_conda_package("scikit-learn")
conda_dep.add_conda_package("pandas")
conda_dep.add_conda_package("numpy")
env.python.conda_dependencies = conda_dep

# Create experiment
experiment = Experiment(workspace=ws, name='timeseries-forecasting')

# Configure script run
script_config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    arguments=[
        '--data-path', dataset.as_mount(),
        '--n-estimators', 200,
        '--max-depth', 15,
        '--min-samples-split', 5
    ],
    compute_target=compute_target,
    environment=env
)

# Submit run
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
```

## 5. Model Registration & Deployment

### Register Model

```python
from azureml.core.model import Model

# Register model
model = run.register_model(
    model_name='timeseries-rf-model',
    model_path='outputs/model.pkl',
    description='Random Forest model for time series forecasting'
)
```

### Scoring Script (score.py)

```python
# score.py
import json
import joblib
import pandas as pd
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('timeseries-rf-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return predictions
        return predictions.tolist()
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### Deploy as Web Service

```python
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    description="Time series forecasting endpoint"
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name="timeseries-rf-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```

## 6. Model Monitoring & Evaluation

### Advanced Evaluation Metrics

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error

def comprehensive_evaluation(y_true, y_pred, dates=None):
    """Comprehensive time series model evaluation"""
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R²': r2_score(y_true, y_pred)
    }
    
    # Residual analysis
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actual vs Predicted
    axes[0,0].scatter(y_true, y_pred, alpha=0.6)
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual')
    axes[0,0].set_ylabel('Predicted')
    axes[0,0].set_title('Actual vs Predicted')
    
    # Residuals plot
    if dates is not None:
        axes[0,1].plot(dates, residuals)
    else:
        axes[0,1].plot(residuals)
    axes[0,1].set_title('Residuals over Time')
    axes[0,1].set_ylabel('Residuals')
    
    # Residuals distribution
    axes[1,0].hist(residuals, bins=30, alpha=0.7)
    axes[1,0].set_title('Residuals Distribution')
    axes[1,0].set_xlabel('Residuals')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    return metrics
```

### Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names, top_n=15):
    """Analyze and visualize feature importance"""
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title('Top Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return importance_df
```

## 7. Production Pipeline

### Azure ML Pipeline

```python
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Pipeline data
processed_data = PipelineData("processed_data", datastore=datastore)
model_output = PipelineData("model_output", datastore=datastore)

# Data preprocessing step
prep_step = PythonScriptStep(
    script_name="preprocess.py",
    source_directory="./pipeline_steps",
    outputs=[processed_data],
    compute_target=compute_target,
    runconfig=run_config
)

# Training step
train_step = PythonScriptStep(
    script_name="train.py",
    source_directory="./pipeline_steps",
    inputs=[processed_data],
    outputs=[model_output],
    compute_target=compute_target,
    runconfig=run_config
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[prep_step, train_step])
pipeline_run = experiment.submit(pipeline)
```

### Batch Inference Pipeline

```python
from azureml.pipeline.steps import ParallelRunStep
from azureml.pipeline.core import ParallelRunConfig

# Batch scoring configuration
batch_config = ParallelRunConfig(
    source_directory='./batch_scoring',
    entry_script='batch_score.py',
    mini_batch_size='1000',
    error_threshold=-1,
    output_action='append_row',
    environment=env,
    compute_target=compute_target,
    node_count=2
)

# Batch scoring step
batch_step = ParallelRunStep(
    name='batch-forecasting',
    parallel_run_config=batch_config,
    inputs=[input_dataset.as_named_input('input_data')],
    output=output_data,
    allow_reuse=False
)
```

## 8. Key Functions & Utilities

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(model, X, y, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores), np.std(cv_scores)
```

### Data Drift Detection

```python
def detect_data_drift(reference_data, current_data, threshold=0.1):
    """Simple data drift detection"""
    from scipy.stats import ks_2samp
    
    drift_results = {}
    for column in reference_data.columns:
        if column in current_data.columns:
            statistic, p_value = ks_2samp(
                reference_data[column].dropna(),
                current_data[column].dropna()
            )
            drift_results[column] = {
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
    
    return drift_results
```

## 9. Best Practices & Tips

### Time Series Specific Considerations

- **No Data Leakage**: Ensure future information doesn’t leak into past predictions
- **Temporal Ordering**: Always respect chronological order in splits
- **Seasonality**: Consider seasonal decomposition and seasonal features
- **Stationarity**: Check for trends and seasonality, apply differencing if needed
- **Cross-validation**: Use TimeSeriesSplit instead of standard k-fold

### Random Forest for Time Series

- **Lag Features**: Include multiple lag periods (1, 7, 30 days)
- **Rolling Statistics**: Moving averages, standard deviations
- **Temporal Features**: Day of week, month, quarter, holidays
- **Feature Selection**: Use feature importance to remove irrelevant features
- **Hyperparameter Tuning**: Focus on n_estimators, max_depth, min_samples_split

### Azure ML Optimization

- **Compute Management**: Use auto-scaling for cost efficiency
- **Data Versioning**: Version datasets for reproducibility
- **Environment Management**: Use registered environments for consistency
- **Model Versioning**: Tag and version models systematically
- **Monitoring**: Set up data drift and model performance monitoring

### Performance Tips

- **Parallel Processing**: Leverage Azure ML’s parallel capabilities
- **Feature Store**: Consider Azure ML feature store for reusable features
- **Automated ML**: Compare with Azure AutoML for time series
- **Resource Optimization**: Right-size compute based on data volume
- **Caching**: Use dataset caching for repeated experiments