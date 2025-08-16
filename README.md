# Azure ML Time Series Forecasting with Random Forest - Complete Reference

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


# Azure Databricks ML Time Series Forecasting Reference Card

## Complete Workflow with Random Forest Regressor

-----

## 1. Environment Setup & Imports

### Essential Libraries

```python
# Core data manipulation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

# Databricks specific
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
```

### Spark Session Initialization

```python
spark = SparkSession.builder.appName("TimeSeriesForecasting").getOrCreate()
```

-----

## 2. Data Loading & Initial Exploration

### Load Data from Various Sources

```python
# From Delta Lake
df = spark.table("catalog.schema.timeseries_table")

# From CSV in DBFS
df = spark.read.csv("/databricks-datasets/timeseries/data.csv", 
                    header=True, inferSchema=True)

# From Azure Data Lake
df = spark.read.format("delta").load("abfss://container@storage.dfs.core.windows.net/path/")

# Convert to Pandas for ML workflow
pdf = df.toPandas()
```

### Data Exploration Functions

```python
def explore_timeseries(df, date_col, target_col):
    """Comprehensive time series exploration"""
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Basic statistics
    print("\nTarget variable statistics:")
    print(df[target_col].describe())
    
    # Plot time series
    plt.figure(figsize=(15, 6))
    plt.plot(df[date_col], df[target_col])
    plt.title(f'{target_col} Over Time')
    plt.xticks(rotation=45)
    plt.show()
    
    return df.info()

# Usage
explore_timeseries(pdf, 'date', 'target_value')
```

-----

## 3. Data Preprocessing & Cleaning

### Date Processing

```python
def preprocess_dates(df, date_col):
    """Convert and extract date features"""
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Extract date components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    return df

pdf = preprocess_dates(pdf, 'date')
```

### Missing Value Handling

```python
def handle_missing_values(df, method='interpolate'):
    """Handle missing values in time series"""
    if method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'forward_fill':
        return df.fillna(method='ffill')
    elif method == 'backward_fill':
        return df.fillna(method='bfill')
    elif method == 'mean':
        return df.fillna(df.mean())
    
# Apply missing value handling
pdf = handle_missing_values(pdf)
```

### Outlier Detection & Treatment

```python
def detect_outliers(df, col, method='iqr', factor=1.5):
    """Detect outliers using IQR or Z-score"""
    if method == 'iqr':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return (df[col] < lower) | (df[col] > upper)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df[col]))
        return z_scores > factor

# Detect and handle outliers
outliers = detect_outliers(pdf, 'target_value', method='iqr')
pdf_clean = pdf[~outliers].copy()
```

-----

## 4. Feature Engineering for Time Series

### Lag Features

```python
def create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30]):
    """Create lagged features"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

pdf = create_lag_features(pdf, 'target_value', lags=[1, 2, 3, 7, 14, 30])
```

### Rolling Window Features

```python
def create_rolling_features(df, target_col, windows=[7, 14, 30]):
    """Create rolling statistics features"""
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window=window).max()
    return df

pdf = create_rolling_features(pdf, 'target_value', windows=[7, 14, 30])
```

### Seasonal Features

```python
def create_seasonal_features(df, date_col):
    """Create seasonal and cyclical features"""
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Cyclical encoding for better ML performance
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

pdf = create_seasonal_features(pdf, 'date')
```

### Technical Indicators

```python
def create_technical_features(df, target_col, short_window=5, long_window=20):
    """Create technical analysis features"""
    # Moving averages
    df[f'SMA_{short_window}'] = df[target_col].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df[target_col].rolling(window=long_window).mean()
    
    # Exponential moving average
    df[f'EMA_{short_window}'] = df[target_col].ewm(span=short_window).mean()
    
    # Rate of change
    df[f'ROC_{short_window}'] = df[target_col].pct_change(periods=short_window)
    
    # Relative Strength Index (simplified)
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

pdf = create_technical_features(pdf, 'target_value')
```

-----

## 5. Feature Selection & Engineering

### Correlation Analysis

```python
def analyze_feature_correlation(df, target_col, threshold=0.1):
    """Analyze feature correlation with target"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    high_corr_features = correlations[correlations > threshold].index.tolist()
    high_corr_features.remove(target_col)  # Remove target itself
    
    plt.figure(figsize=(10, 8))
    correlations[1:21].plot(kind='barh')  # Top 20 excluding target
    plt.title(f'Feature Correlation with {target_col}')
    plt.show()
    
    return high_corr_features

selected_features = analyze_feature_correlation(pdf, 'target_value', threshold=0.05)
```

### Advanced Feature Selection

```python
def select_best_features(X, y, k=20, method='f_regression'):
    """Select best features using statistical tests"""
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    # Feature scores
    scores = pd.DataFrame({
        'feature': X.columns[selected_indices],
        'score': selector.scores_[selected_indices]
    }).sort_values('score', ascending=False)
    
    return X_selected, selected_features, scores

# Prepare features for selection
feature_cols = [col for col in pdf.columns if col not in ['date', 'target_value']]
X = pdf[feature_cols].dropna()
y = pdf.loc[X.index, 'target_value']

X_selected, best_features, feature_scores = select_best_features(X, y, k=15)
```

-----

## 6. Data Splitting for Time Series

### Time-Based Split

```python
def time_series_split(df, date_col, train_ratio=0.7, val_ratio=0.15):
    """Split time series data chronologically"""
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = df_sorted.iloc[:train_end]
    val_data = df_sorted.iloc[train_end:val_end]
    test_data = df_sorted.iloc[val_end:]
    
    print(f"Train: {len(train_data)} samples ({df_sorted.iloc[0][date_col]} to {train_data.iloc[-1][date_col]})")
    print(f"Validation: {len(val_data)} samples ({val_data.iloc[0][date_col]} to {val_data.iloc[-1][date_col]})")
    print(f"Test: {len(test_data)} samples ({test_data.iloc[0][date_col]} to {df_sorted.iloc[-1][date_col]})")
    
    return train_data, val_data, test_data

# Split the data
train_df, val_df, test_df = time_series_split(pdf.dropna(), 'date')
```

### Prepare Features and Targets

```python
def prepare_features_targets(train_df, val_df, test_df, feature_cols, target_col):
    """Prepare feature matrices and target vectors"""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = prepare_features_targets(
    train_df, val_df, test_df, best_features, 'target_value'
)
```

-----

## 7. Model Development & Training

### Random Forest Configuration

```python
def create_rf_model(n_estimators=100, max_depth=None, min_samples_split=2, 
                   min_samples_leaf=1, random_state=42):
    """Create Random Forest Regressor with specified parameters"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        oob_score=True  # Out-of-bag score for validation
    )
    return model
```

### Hyperparameter Tuning

```python
def hyperparameter_tuning(X_train, y_train, cv_folds=3):
    """Perform hyperparameter tuning using TimeSeriesSplit"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_}")
    
    return grid_search.best_estimator_, grid_search.best_params_

# Perform hyperparameter tuning
best_model, best_params = hyperparameter_tuning(X_train, y_train, cv_folds=3)
```

### Model Training

```python
def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """Train the model and return training history"""
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    
    # Validation metrics if provided
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
    
    return model

# Train the model
trained_model = train_model(best_model, X_train, y_train, X_val, y_val)
```

-----

## 8. Model Evaluation & Metrics

### Comprehensive Evaluation Function

```python
def evaluate_model(model, X_test, y_test, model_name="Random Forest"):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return y_pred, metrics

# Evaluate the model
predictions, eval_metrics = evaluate_model(trained_model, X_test, y_test)
```

### Visualization Functions

```python
def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    # Time series plot
    plt.subplot(2, 2, 2)
    plt.plot(y_true.values, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title('Time Series Comparison')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # Residuals histogram
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()

plot_predictions(y_test, predictions)
```

### Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names, top_k=15):
    """Analyze and plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_k)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_k} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

importance_df = analyze_feature_importance(trained_model, best_features, top_k=15)
```

-----

## 9. Model Persistence & Deployment

### Model Saving

```python
import joblib
from datetime import datetime

def save_model(model, model_path, metadata=None):
    """Save model with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{model_path}/rf_model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    
    # Save metadata
    if metadata:
        metadata_filename = f"{model_path}/metadata_{timestamp}.json"
        import json
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"Model saved to: {model_filename}")
    return model_filename

# Save the model
model_metadata = {
    'model_type': 'RandomForestRegressor',
    'features': best_features,
    'performance_metrics': eval_metrics,
    'training_date': datetime.now(),
    'hyperparameters': best_params
}

model_path = save_model(trained_model, "/dbfs/models/timeseries", model_metadata)
```

### Model Loading and Prediction Function

```python
def load_model_and_predict(model_path, X_new):
    """Load model and make predictions"""
    model = joblib.load(model_path)
    predictions = model.predict(X_new)
    return predictions

# Example usage
# loaded_predictions = load_model_and_predict(model_path, X_test)
```

-----

## 10. Production Deployment Patterns

### Batch Prediction Pipeline

```python
def batch_prediction_pipeline(model, input_table, output_table, feature_cols):
    """Production batch prediction pipeline"""
    # Load new data
    new_data = spark.table(input_table)
    new_pdf = new_data.toPandas()
    
    # Feature engineering (apply same transformations)
    new_pdf = preprocess_dates(new_pdf, 'date')
    new_pdf = create_lag_features(new_pdf, 'target_value')
    new_pdf = create_rolling_features(new_pdf, 'target_value')
    new_pdf = create_seasonal_features(new_pdf, 'date')
    
    # Make predictions
    X_new = new_pdf[feature_cols].fillna(method='ffill')
    predictions = model.predict(X_new)
    
    # Save results
    results_df = new_pdf[['date']].copy()
    results_df['predicted_value'] = predictions
    results_df['prediction_date'] = datetime.now()
    
    # Convert back to Spark DataFrame and save
    results_spark = spark.createDataFrame(results_df)
    results_spark.write.mode('overwrite').saveAsTable(output_table)
    
    print(f"Predictions saved to {output_table}")
    return results_df
```

### Real-time Prediction Function

```python
def real_time_prediction(model, feature_vector):
    """Real-time prediction for streaming data"""
    prediction = model.predict([feature_vector])[0]
    confidence = np.std([tree.predict([feature_vector])[0] for tree in model.estimators_])
    
    return {
        'prediction': prediction,
        'confidence_std': confidence,
        'timestamp': datetime.now()
    }
```

-----

## 11. Monitoring & Maintenance

### Model Performance Monitoring

```python
def monitor_model_performance(y_true, y_pred, baseline_metrics):
    """Monitor model performance against baseline"""
    current_metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }
    
    # Calculate performance degradation
    degradation = {}
    for metric, value in current_metrics.items():
        if metric in baseline_metrics:
            if metric == 'R²':  # Higher is better
                degradation[metric] = (baseline_metrics[metric] - value) / baseline_metrics[metric] * 100
            else:  # Lower is better
                degradation[metric] = (value - baseline_metrics[metric]) / baseline_metrics[metric] * 100
    
    print("Performance Monitoring Results:")
    print("-" * 40)
    for metric in current_metrics:
        print(f"{metric}: Current={current_metrics[metric]:.4f}, "
              f"Baseline={baseline_metrics.get(metric, 'N/A')}, "
              f"Degradation={degradation.get(metric, 0):.2f}%")
    
    # Alert if degradation > threshold
    alert_threshold = 10  # 10% degradation
    for metric, deg in degradation.items():
        if deg > alert_threshold:
            print(f"⚠️  ALERT: {metric} degraded by {deg:.2f}% (threshold: {alert_threshold}%)")
    
    return current_metrics, degradation
```

### Data Drift Detection

```python
def detect_data_drift(X_baseline, X_current, threshold=0.05):
    """Simple data drift detection using statistical tests"""
    drift_detected = {}
    
    for col in X_baseline.columns:
        if X_baseline[col].dtype in ['int64', 'float64']:
            # Use Kolmogorov-Smirnov test for continuous variables
            stat, p_value = stats.ks_2samp(X_baseline[col], X_current[col])
            drift_detected[col] = p_value < threshold
            
            if drift_detected[col]:
                print(f"⚠️  Data drift detected in {col} (p-value: {p_value:.6f})")
    
    return drift_detected
```

-----

## 12. Advanced Techniques & Extensions

### Ensemble Methods

```python
def create_ensemble_model(X_train, y_train):
    """Create ensemble of different models"""
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
    
    # Individual models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr_model = LinearRegression()
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model)
    ])
    
    ensemble.fit(X_train, y_train)
    return ensemble
```

### Feature Store Integration

```python
def setup_feature_store(feature_df, feature_store_name):
    """Setup feature store for reusable features"""
    from databricks.feature_store import FeatureStoreClient
    
    fs = FeatureStoreClient()
    
    # Create feature table
    fs.create_table(
        name=feature_store_name,
        primary_keys=['date'],
        df=spark.createDataFrame(feature_df),
        description="Time series features for forecasting"
    )
    
    return fs
```

### Automated Retraining Pipeline

```python
def automated_retraining_pipeline(model, X_train, y_train, performance_threshold=0.1):
    """Automated model retraining when performance degrades"""
    # Check current performance
    current_score = model.score(X_train, y_train)
    
    # If performance below threshold, retrain
    if current_score < performance_threshold:
        print("Performance below threshold. Starting retraining...")
        
        # Retrain with hyperparameter tuning
        new_model, _ = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate new model
        new_score = new_model.score(X_train, y_train)
        
        if new_score > current_score:
            print(f"Retraining successful. Score improved: {current_score:.4f} → {new_score:.4f}")
            return new_model
        else:
            print("Retraining did not improve performance. Keeping original model.")
            return model
    
    return model
```

-----

## 13. Key Functions Summary

### Essential Functions Checklist

```python
# Data Processing
✓ preprocess_dates()          # Date feature extraction
✓ handle_missing_values()     # Missing value treatment
✓ detect_outliers()          # Outlier detection

# Feature Engineering  
✓ create_lag_features()      # Lagged variables
✓ create_rolling_features()  # Rolling statistics
✓ create_seasonal_features() # Seasonal patterns
✓ create_technical_features() # Technical indicators

# Model Development
✓ hyperparameter_tuning()    # Grid search with CV
✓ train_model()             # Model training
✓ evaluate_model()          # Comprehensive evaluation

# Production
✓ save_model()              # Model persistence
✓ batch_prediction_pipeline() # Batch predictions
✓ monitor_model_performance() # Performance monitoring
```

### Best Practices Summary

- **Always use time-based splits** for time series data
- **Create comprehensive lag and rolling features**
- **Use TimeSeriesSplit for cross-validation**
- **Monitor for data drift and model degradation**
- **Implement proper feature stores for production**
- **Set up automated retraining pipelines**
- **Use cyclical encoding for temporal features**
- **Handle missing values appropriately for time series**

-----

## 14. Common Pitfalls & Solutions

|Pitfall                        |Solution                           |
|-------------------------------|-----------------------------------|
|Data leakage from future       |Use only past data for features    |
|Random train/test split        |Use time-based splitting           |
|Missing seasonal patterns      |Add cyclical date features         |
|Overfitting to recent data     |Use longer validation periods      |
|Ignoring data drift            |Implement monitoring systems       |
|Static feature engineering     |Automate feature creation          |
|Poor handling of missing values|Use time-series appropriate methods|
|No model versioning            |Implement MLOps practices          |



