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


# Azure Machine Learning NLP Workflow - Complete Reference Card

## Table of Contents

1. [Environment Setup](#environment-setup)
1. [Data Ingestion & Storage](#data-ingestion--storage)
1. [Data Preprocessing](#data-preprocessing)
1. [Feature Engineering](#feature-engineering)
1. [Model Training](#model-training)
1. [Model Evaluation](#model-evaluation)
1. [Model Deployment](#model-deployment)
1. [Monitoring & MLOps](#monitoring--mlops)
1. [Advanced Techniques](#advanced-techniques)
1. [Common Code Patterns](#common-code-patterns)

-----

## Environment Setup

### Azure ML SDK Installation

```bash
pip install azureml-sdk[notebooks,automl]
pip install azureml-core
pip install transformers torch datasets
pip install nltk spacy textblob
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Workspace Connection

```python
from azureml.core import Workspace, Dataset, Experiment
from azureml.core.authentication import InteractiveLoginAuthentication

# Connect to workspace
ws = Workspace.from_config()  # Uses config.json
# Or explicit connection
ws = Workspace(subscription_id='your-sub-id', 
               resource_group='your-rg',
               workspace_name='your-workspace')

print(f"Workspace: {ws.name}, Location: {ws.location}")
```

### Compute Setup

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "nlp-cluster"
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC6s_v3',  # GPU instance
        min_nodes=0,
        max_nodes=4,
        idle_seconds_before_scaledown=1800
    )
    compute_target = ComputeTarget.create(ws, compute_name, config)
    compute_target.wait_for_completion(show_output=True)
```

-----

## Data Ingestion & Storage

### Dataset Registration

```python
from azureml.core import Dataset
from azureml.data.datapath import DataPath

# From local files
dataset = Dataset.File.from_files('data/text_files/')
dataset = dataset.register(workspace=ws, name='text-dataset', 
                          description='NLP training data')

# From Azure Blob Storage
datastore = ws.get_default_datastore()
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'nlp-data/reviews.csv'))
dataset = dataset.register(ws, 'reviews-dataset')

# From URLs
dataset = Dataset.File.from_files(['https://example.com/data.txt'])
```

### Data Loading Patterns

```python
import pandas as pd

# Load tabular dataset
dataset = Dataset.get_by_name(ws, 'reviews-dataset')
df = dataset.to_pandas_dataframe()

# Load file dataset
file_dataset = Dataset.get_by_name(ws, 'text-dataset')
download_path = file_dataset.download(target_path='.', overwrite=True)

# Streaming large datasets
for batch in dataset.to_pandas_dataframe().iterrows():
    # Process in batches
    pass
```

-----

## Data Preprocessing

### Text Cleaning Pipeline

```python
import re
import nltk
import spacy
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def clean_text(self, text):
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def tokenize_and_process(self, text):
        doc = nlp(text)
        tokens = []
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                if self.lemmatize:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
        
        return tokens
    
    def preprocess(self, texts):
        processed = []
        for text in texts:
            clean_text = self.clean_text(text)
            tokens = self.tokenize_and_process(clean_text)
            processed.append(' '.join(tokens))
        return processed

# Usage
preprocessor = TextPreprocessor()
df['processed_text'] = preprocessor.preprocess(df['raw_text'].tolist())
```

### Advanced Text Processing

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=2,
    max_df=0.95
)
X_tfidf = tfidf.fit_transform(df['processed_text'])

# BERT Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

tokenized_data = tokenize_texts(df['processed_text'].tolist())
```

-----

## Feature Engineering

### Traditional NLP Features

```python
import numpy as np
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class FeatureEngineer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def extract_basic_features(self, texts):
        features = []
        for text in texts:
            feat_dict = {
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'avg_word_length': np.mean([len(word) for word in text.split()]),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_count': sum(1 for c in text if c.isupper()),
                'readability': textstat.flesch_reading_ease(text)
            }
            features.append(feat_dict)
        return pd.DataFrame(features)
    
    def extract_sentiment_features(self, texts):
        sentiments = []
        for text in texts:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiments.append(scores)
        return pd.DataFrame(sentiments)
    
    def extract_pos_features(self, texts):
        pos_features = []
        for text in texts:
            doc = nlp(text)
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            pos_features.append(pos_counts)
        return pd.DataFrame(pos_features).fillna(0)

# Usage
fe = FeatureEngineer()
basic_features = fe.extract_basic_features(df['processed_text'])
sentiment_features = fe.extract_sentiment_features(df['processed_text'])
pos_features = fe.extract_pos_features(df['processed_text'])

# Combine all features
feature_df = pd.concat([basic_features, sentiment_features, pos_features], axis=1)
```

### Embedding Generation

```python
from sentence_transformers import SentenceTransformer
import torch

# Sentence-BERT Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['processed_text'].tolist())

# Word2Vec/FastText
from gensim.models import Word2Vec, FastText

# Train custom Word2Vec
tokenized_texts = [text.split() for text in df['processed_text']]
w2v_model = Word2Vec(sentences=tokenized_texts, 
                     vector_size=100, 
                     window=5, 
                     min_count=2, 
                     workers=4)

def get_text_vector(text, model, vector_size=100):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

text_vectors = np.array([get_text_vector(text, w2v_model) 
                        for text in df['processed_text']])
```

-----

## Model Training

### Traditional ML Models

```python
from azureml.train.sklearn import SKLearn
from azureml.core import ScriptRunConfig, Environment
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Prepare training script
training_script = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import argparse
from azureml.core import Run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--model-output', type=str, help='Path to save model')
    args = parser.parse_args()
    
    # Get run context
    run = Run.get_context()
    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Feature extraction
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Log metrics
    score = model.score(X, y)
    run.log('accuracy', score)
    
    # Save model and vectorizer
    joblib.dump(model, f'{args.model_output}/model.pkl')
    joblib.dump(tfidf, f'{args.model_output}/vectorizer.pkl')

if __name__ == '__main__':
    main()
"""

# Create training configuration
env = Environment.from_conda_specification(
    name='nlp-env',
    file_path='conda_env.yml'
)

config = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=['--data-path', dataset.as_mount(),
               '--model-output', './outputs/'],
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(ws, 'nlp-experiment')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
```

### Deep Learning with Transformers

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                         TrainingArguments, Trainer)
from datasets import Dataset
import torch

class TransformerTrainer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
    
    def prepare_dataset(self, texts, labels, max_length=512):
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        return dataset
    
    def train(self, train_dataset, eval_dataset=None, output_dir='./results'):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='steps' if eval_dataset else 'no',
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True if eval_dataset else False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        return trainer

# Usage
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

trainer = TransformerTrainer()
train_dataset = trainer.prepare_dataset(X_train, y_train)
eval_dataset = trainer.prepare_dataset(X_test, y_test)
trained_model = trainer.train(train_dataset, eval_dataset)
```

### AutoML for NLP

```python
from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    training_data=dataset,
    label_column_name='label',
    n_cross_validations=5,
    enable_early_stopping=True,
    experiment_timeout_minutes=60,
    max_concurrent_iterations=4,
    compute_target=compute_target,
    enable_deep_learning=True,  # Enables BERT-based models
    enable_feature_engineering=True
)

experiment = Experiment(ws, 'automl-nlp-experiment')
run = experiment.submit(automl_config, show_output=True)
```

-----

## Model Evaluation

### Comprehensive Evaluation Suite

```python
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.predictions = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            self.probabilities = model.predict_proba(X_test)
    
    def print_classification_report(self):
        print(classification_report(self.y_test, self.predictions, 
                                  target_names=self.class_names))
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def calculate_metrics(self):
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'f1_macro': f1_score(self.y_test, self.predictions, average='macro'),
            'f1_weighted': f1_score(self.y_test, self.predictions, average='weighted')
        }
        
        if hasattr(self, 'probabilities') and len(set(self.y_test)) == 2:
            metrics['auc_roc'] = roc_auc_score(self.y_test, self.probabilities[:, 1])
        
        return metrics
    
    def log_metrics_to_azure(self, run):
        metrics = self.calculate_metrics()
        for metric_name, value in metrics.items():
            run.log(metric_name, value)

# Usage
evaluator = ModelEvaluator(trained_model, X_test, y_test, ['Negative', 'Positive'])
evaluator.print_classification_report()
evaluator.plot_confusion_matrix()
metrics = evaluator.calculate_metrics()
print(metrics)
```

### Advanced Evaluation Techniques

```python
from sklearn.model_selection import cross_val_score
from lime.lime_text import LimeTextExplainer
import shap

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# LIME Explanations
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

def predict_proba_wrapper(texts):
    # Transform texts using your preprocessing pipeline
    transformed = tfidf.transform(texts)
    return model.predict_proba(transformed)

# Explain a single prediction
idx = 0
exp = explainer.explain_instance(X_test[idx], predict_proba_wrapper, 
                                num_features=10)
exp.show_in_notebook(text=True)

# SHAP Values (for tree-based models)
if hasattr(model, 'predict_proba'):
    explainer_shap = shap.Explainer(model)
    shap_values = explainer_shap(X_test_sample)
    shap.summary_plot(shap_values, X_test_sample)
```

-----

## Model Deployment

### Real-time Inference Endpoint

```python
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment

# Register model
model = Model.register(
    workspace=ws,
    model_path='./model.pkl',
    model_name='nlp-classifier',
    description='Text classification model'
)

# Create scoring script
scoring_script = """
import joblib
import pandas as pd
from azureml.core.model import Model

def init():
    global model, vectorizer
    model_path = Model.get_model_path('nlp-classifier')
    model = joblib.load(model_path + '/model.pkl')
    vectorizer = joblib.load(model_path + '/vectorizer.pkl')

def run(raw_data):
    try:
        data = pd.read_json(raw_data)
        texts = data['text'].tolist()
        
        # Preprocess and vectorize
        X = vectorizer.transform(texts)
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        return result
    except Exception as e:
        return {'error': str(e)}
"""

# Deploy to ACI
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    description='NLP Classification Service'
)

service = Model.deploy(
    workspace=ws,
    name='nlp-service',
    models=[model],
    inference_config=InferenceConfig(
        entry_script='score.py',
        environment=env
    ),
    deployment_config=aci_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"Service URL: {service.scoring_uri}")
```

### Batch Inference Pipeline

```python
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

# Batch inference script
batch_script = """
import argparse
import pandas as pd
import joblib
from azureml.core import Run, Model
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()
    
    # Load model
    run = Run.get_context()
    ws = run.experiment.workspace
    model = Model(ws, 'nlp-classifier')
    model_path = model.download(target_dir='.')
    
    classifier = joblib.load(os.path.join(model_path, 'model.pkl'))
    vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
    
    # Process data
    df = pd.read_csv(args.input_data)
    X = vectorizer.transform(df['text'])
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    
    # Save results
    df['prediction'] = predictions
    df['probability'] = probabilities.max(axis=1)
    df.to_csv(os.path.join(args.output_path, 'predictions.csv'), index=False)

if __name__ == '__main__':
    main()
"""

# Create pipeline step
output_dir = PipelineData('predictions', datastore=ws.get_default_datastore())

batch_step = PythonScriptStep(
    name='batch-inference',
    script_name='batch_inference.py',
    arguments=['--input-data', dataset.as_mount(),
               '--output-path', output_dir],
    outputs=[output_dir],
    compute_target=compute_target,
    runconfig=RunConfiguration()
)

# Create and run pipeline
pipeline = Pipeline(workspace=ws, steps=[batch_step])
pipeline_run = experiment.submit(pipeline)
```

-----

## Monitoring & MLOps

### Model Monitoring Setup

```python
from azureml.monitoring import ModelDataCollector

# Enable data collection in scoring script
def init():
    global model, inputs_dc, predictions_dc
    
    # Load model
    model_path = Model.get_model_path('nlp-classifier')
    model = joblib.load(model_path)
    
    # Initialize data collectors
    inputs_dc = ModelDataCollector(
        model_name='nlp-classifier',
        designation='inputs',
        feature_names=['text']
    )
    predictions_dc = ModelDataCollector(
        model_name='nlp-classifier',
        designation='predictions',
        feature_names=['prediction', 'probability']
    )

def run(raw_data):
    data = pd.read_json(raw_data)
    texts = data['text'].tolist()
    
    # Collect input data
    inputs_dc.collect(texts)
    
    # Make predictions
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    
    # Collect prediction data
    predictions_dc.collect([predictions, probabilities.max(axis=1)])
    
    return {'predictions': predictions.tolist()}
```

### MLOps Pipeline

```python
from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

# Training pipeline
def create_training_pipeline():
    # Data preparation step
    prep_step = PythonScriptStep(
        name='data-prep',
        script_name='data_prep.py',
        compute_target=compute_target,
        runconfig=RunConfiguration()
    )
    
    # Training step
    train_step = PythonScriptStep(
        name='model-training',
        script_name='train.py',
        compute_target=compute_target,
        runconfig=RunConfiguration()
    )
    
    # Evaluation step
    eval_step = PythonScriptStep(
        name='model-evaluation',
        script_name='evaluate.py',
        compute_target=compute_target,
        runconfig=RunConfiguration()
    )
    
    # Register model step
    register_step = PythonScriptStep(
        name='register-model',
        script_name='register_model.py',
        compute_target=compute_target,
        runconfig=RunConfiguration()
    )
    
    pipeline_steps = [prep_step, train_step, eval_step, register_step]
    training_pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    
    return training_pipeline

# Publish pipeline
pipeline = create_training_pipeline()
published_pipeline = pipeline.publish(
    name='nlp-training-pipeline',
    description='End-to-end NLP training pipeline'
)

# Schedule pipeline
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule

recurrence = ScheduleRecurrence(frequency='Week', interval=1)
schedule = Schedule.create(
    workspace=ws,
    name='weekly-retraining',
    pipeline_id=published_pipeline.id,
    experiment_name='scheduled-training',
    recurrence=recurrence
)
```

-----

## Advanced Techniques

### Multi-label Classification

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Prepare multi-label data
def prepare_multilabel_data(df, label_columns):
    y = df[label_columns].values
    return y

# Train multi-label model
multilabel_classifier = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100)
)
multilabel_classifier.fit(X_train, y_train_multilabel)

# Evaluation for multi-label
from sklearn.metrics import hamming_loss, jaccard_score

predictions_multilabel = multilabel_classifier.predict(X_test)
hamming = hamming_loss(y_test_multilabel, predictions_multilabel)
jaccard = jaccard_score(y_test_multilabel, predictions_multilabel, average='micro')
```

### Named Entity Recognition (NER)

```python
import spacy
from spacy.training import Example
from spacy.tokens import DocBin

# Custom NER training
def train_ner_model(training_data, model_name='en_core_web_sm'):
    nlp = spacy.load(model_name)
    ner = nlp.get_pipe('ner')
    
    # Add labels
    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    # Training loop
    nlp.update([Example.from_dict(nlp.make_doc(text), annotations) 
                for text, annotations in training_data])
    
    return nlp

# Usage with Azure ML
training_data = [
    ("Apple Inc. is located in Cupertino", 
     {"entities": [(0, 9, "ORG"), (24, 33, "GPE")]}),
    # More training examples...
]

custom_nlp = train_ner_model(training_data)
```

### Topic Modeling

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.sklearn

# LDA Topic Modeling
def perform_topic_modeling(texts, n_topics=5):
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        lowercase=True
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10
    )
    
    lda.fit(doc_term_matrix)
    
    return lda, vectorizer, doc_term_matrix

# Visualize topics
lda_model, vectorizer, doc_term_matrix = perform_topic_modeling(df['processed_text'])
vis = pyLDAvis.sklearn.prepare(lda_model, doc_term_matrix, vectorizer)
pyLDAvis.display(vis)
```

### Transfer Learning with Custom Transformers

```python
from transformers import (AutoModel, AutoConfig, AutoTokenizer,
                         TrainingArguments, Trainer)
import torch.nn as nn

class CustomTransformerModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

# Fine-tune custom model
model = CustomTransformerModel('bert-base-uncased', num_classes=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

-----

## Common Code Patterns

### Data Pipeline Template

```python
class NLPPipeline:
    def __init__(self, workspace):
        self.ws = workspace
        self.preprocessor = None
        self.vectorizer = None
        self.model = None
    
    def load_data(self, dataset_name):
        dataset = Dataset.get_by_name(self.ws, dataset_name)
        return dataset.to_pandas_dataframe()
    
    def preprocess_data(self, df, text_column):
        self.
```
