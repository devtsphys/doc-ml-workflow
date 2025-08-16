# MLflow Complete Reference Card

## Overview

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Core Components

### 1. MLflow Tracking

Track experiments, parameters, metrics, and artifacts.

#### Basic Tracking Setup

```python
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts

# Set tracking URI (optional)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")
```

#### Run Context Management

```python
# Method 1: Context manager (recommended)
with mlflow.start_run():
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.786)
    
# Method 2: Manual start/end
run = mlflow.start_run()
mlflow.log_param("alpha", 0.1)
mlflow.end_run()

# Method 3: Active run
mlflow.start_run()
# ... logging code ...
mlflow.end_run()
```

#### Logging Functions

```python
# Parameters (hyperparameters, configs)
mlflow.log_param("learning_rate", 0.01)
mlflow.log_params({"batch_size": 32, "epochs": 100})

# Metrics (model performance)
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metrics({"precision": 0.92, "recall": 0.88})

# Step-wise metrics (for tracking over epochs)
for epoch in range(100):
    mlflow.log_metric("loss", loss_value, step=epoch)

# Artifacts (files, models, plots)
mlflow.log_artifact("model.pkl")
mlflow.log_artifacts("output_dir")

# Text
mlflow.log_text("Some important note", "notes.txt")

# Dictionary as JSON
mlflow.log_dict({"key": "value"}, "config.json")
```

#### Tags and Notes

```python
# Set tags
mlflow.set_tag("model_type", "random_forest")
mlflow.set_tags({"version": "1.0", "team": "data_science"})

# Add notes
mlflow.set_tag("mlflow.note.content", "This is a baseline model")
```

### 2. MLflow Models

Standardized model packaging and deployment.

#### Model Flavors

```python
# Scikit-learn
import mlflow.sklearn
mlflow.sklearn.log_model(model, "model")
loaded_model = mlflow.sklearn.load_model("runs:/{}/model".format(run_id))

# PyTorch
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow/Keras
import mlflow.tensorflow
import mlflow.keras
mlflow.keras.log_model(model, "model")

# XGBoost
import mlflow.xgboost
mlflow.xgboost.log_model(model, "model")

# LightGBM
import mlflow.lightgbm
mlflow.lightgbm.log_model(model, "model")

# Statsmodels
import mlflow.statsmodels
mlflow.statsmodels.log_model(model, "model")

# Spark ML
import mlflow.spark
mlflow.spark.log_model(model, "model")

# Custom Python function
import mlflow.pyfunc
mlflow.pyfunc.log_model("model", python_model=custom_model)
```

#### Custom Model Example

```python
import mlflow.pyfunc

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log custom model
with mlflow.start_run():
    wrapped_model = ModelWrapper(trained_model)
    mlflow.pyfunc.log_model(
        "custom_model", 
        python_model=wrapped_model,
        registered_model_name="MyCustomModel"
    )
```

#### Model Signature

```python
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec

# Infer signature automatically
signature = infer_signature(X_train, y_pred)

# Manual signature definition
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
    ColSpec("string", "category")
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 3. MLflow Projects

Reproducible ML code packaging.

#### MLproject File

```yaml
name: My Project
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.1}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"
    
  validate:
    parameters:
      model_uri: string
    command: "python validate.py {model_uri}"
```

#### Running Projects

```python
# Run local project
mlflow.run(".", parameters={"alpha": 0.5})

# Run from GitHub
mlflow.run(
    "https://github.com/user/repo.git",
    parameters={"alpha": 0.5}
)

# Run specific entry point
mlflow.run(".", entry_point="validate", parameters={"model_uri": "runs:/abc/model"})
```

### 4. MLflow Model Registry

Central model store for lifecycle management.

#### Registering Models

```python
# Register during logging
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="ProductionModel"
)

# Register existing run
model_uri = "runs:/{}/model".format(run_id)
mlflow.register_model(model_uri, "ProductionModel")

# Register with version description
result = mlflow.register_model(
    model_uri,
    "ProductionModel",
    description="Model trained on latest dataset"
)
```

#### Model Versions and Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List registered models
models = client.list_registered_models()

# Get model versions
versions = client.get_registered_model("ProductionModel").latest_versions

# Transition model stage
client.transition_model_version_stage(
    name="ProductionModel",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# Update model version
client.update_model_version(
    name="ProductionModel",
    version=1,
    description="Updated description"
)

# Delete model version
client.delete_model_version("ProductionModel", version=1)
```

#### Loading Models from Registry

```python
# Load latest version
model = mlflow.pyfunc.load_model("models:/ProductionModel/latest")

# Load specific version
model = mlflow.pyfunc.load_model("models:/ProductionModel/1")

# Load by stage
model = mlflow.pyfunc.load_model("models:/ProductionModel/Production")
```

## Advanced Features

### Autologging

```python
# Enable autologging for specific libraries
mlflow.sklearn.autolog()
mlflow.keras.autolog()
mlflow.pytorch.autolog()
mlflow.xgboost.autolog()

# Enable for all supported libraries
mlflow.autolog()

# Disable autologging
mlflow.sklearn.autolog(disable=True)

# Configure autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False  # Don't log models automatically
)
```

### Parent and Child Runs

```python
# Parent run
with mlflow.start_run() as parent_run:
    mlflow.log_param("experiment_type", "hyperparameter_tuning")
    
    # Child runs for different configurations
    for alpha in [0.1, 0.5, 1.0]:
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_param("alpha", alpha)
            # Train and log model...
```

### Search and Query Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Get run data
run = client.get_run(run_id)
print(f"Status: {run.info.status}")
print(f"Start time: {run.info.start_time}")
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")

# Get metric history
history = client.get_metric_history(run_id, "loss")
```

### Experiment Management

```python
# Create experiment
experiment_id = mlflow.create_experiment(
    "New Experiment",
    artifact_location="s3://bucket/artifacts",
    tags={"team": "ml", "project": "recommendation"}
)

# Get experiment
experiment = mlflow.get_experiment(experiment_id)

# List experiments
experiments = mlflow.list_experiments()

# Set experiment by name
mlflow.set_experiment("My Experiment")

# Delete experiment
mlflow.delete_experiment(experiment_id)
```

## Deployment Options

### Local Model Serving

```bash
# Serve model locally
mlflow models serve -m models:/ProductionModel/1 -p 5001

# Serve with conda environment
mlflow models serve -m models:/ProductionModel/1 --env-manager conda

# Build Docker image
mlflow models build-docker -m models:/ProductionModel/1 -n my-model

# Generate Dockerfile
mlflow models generate-dockerfile -m models:/ProductionModel/1 -d ./dockerfile_dir
```

### Cloud Deployments

```python
# Deploy to AWS SageMaker
mlflow.deployments.deploy(
    "sagemaker",
    model_uri="models:/ProductionModel/1",
    config={
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "region_name": "us-west-2"
    }
)

# Deploy to Azure ML
mlflow.deployments.deploy(
    "azureml",
    model_uri="models:/ProductionModel/1",
    config={
        "compute_type": "ACI",
        "location": "eastus2"
    }
)
```

### Batch Inference

```python
# Apply model to Spark DataFrame
predictions = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri="models:/ProductionModel/1"
)

df_with_predictions = df.withColumn("predictions", predictions(*feature_cols))
```

## Configuration and Setup

### Tracking Server Setup

```bash
# Start local tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# With remote artifact store
mlflow server \
    --backend-store-uri postgresql://user:pass@host:port/db \
    --default-artifact-root s3://bucket/artifacts \
    --host 0.0.0.0
```

### Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_EXPERIMENT_NAME=default
```

### Configuration File

```python
# mlflow_config.py
import mlflow
import os

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))
    
    # Configure autologging
    mlflow.autolog()
```

## Best Practices and Patterns

### Experiment Organization

```python
# Use descriptive experiment names
mlflow.set_experiment(f"recommendation_model_{datetime.now().strftime('%Y%m%d')}")

# Use consistent tagging strategy
tags = {
    "model_type": "collaborative_filtering",
    "data_version": "v2.1",
    "feature_set": "user_item_interactions",
    "environment": "production"
}
mlflow.set_tags(tags)

# Log important context
mlflow.log_param("git_commit", get_git_commit())
mlflow.log_param("dataset_size", len(train_data))
mlflow.log_param("feature_count", X_train.shape[1])
```

### Model Validation Pattern

```python
def train_and_validate_model(params):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = train_model(params)
        
        # Validate
        train_score = evaluate_model(model, X_train, y_train)
        val_score = evaluate_model(model, X_val, y_val)
        
        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "overfitting": train_score - val_score
        })
        
        # Log model if performance is good
        if val_score > 0.8:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="BestModel" if val_score > 0.9 else None
            )
        
        return model, val_score
```

### Pipeline Integration

```python
class MLflowPipeline:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        
    def run_experiment(self, config):
        with mlflow.start_run() as run:
            # Data preparation
            X_train, X_val, y_train, y_val = self.prepare_data(config)
            mlflow.log_params(config["data"])
            
            # Feature engineering
            features = self.engineer_features(X_train, config["features"])
            mlflow.log_param("feature_count", len(features))
            
            # Model training
            model = self.train_model(X_train, y_train, config["model"])
            mlflow.log_params(config["model"])
            
            # Evaluation
            metrics = self.evaluate_model(model, X_val, y_val)
            mlflow.log_metrics(metrics)
            
            # Model logging
            mlflow.sklearn.log_model(model, "model")
            
            return run.info.run_id
```

### Error Handling and Cleanup

```python
def safe_mlflow_run(training_function, **kwargs):
    run_id = None
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            result = training_function(**kwargs)
            mlflow.log_metrics({"success": 1})
            return result
            
    except Exception as e:
        if run_id:
            # Log error information
            mlflow.log_param("error_message", str(e))
            mlflow.log_metrics({"success": 0})
        
        # End run gracefully
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
```

## Common Use Cases and Examples

### Hyperparameter Tuning with Optuna

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        # Train and evaluate
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        mlflow.log_metric('accuracy', score)
        
        return score

# Run optimization
study = optuna.create_study(direction='maximize')
mlflc = MLflowCallback(tracking_uri="http://localhost:5000", metric_name="accuracy")

with mlflow.start_run():
    study.optimize(objective, n_trials=100, callbacks=[mlflc])
    
    # Log best parameters
    mlflow.log_params(study.best_params)
    mlflow.log_metric('best_accuracy', study.best_value)
```

### A/B Testing Model Comparison

```python
def compare_models():
    models = {
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC()
    }
    
    results = {}
    
    with mlflow.start_run(run_name="model_comparison"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Log results
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "train_accuracy": train_score,
                    "test_accuracy": test_score
                })
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                results[name] = test_score
        
        # Log comparison results
        best_model = max(results.items(), key=lambda x: x[1])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_accuracy", best_model[1])
    
    return results
```

### Cross-Validation Tracking

```python
from sklearn.model_selection import cross_val_score

def cross_validate_with_mlflow(model, X, y, cv=5):
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Log individual fold scores
        for fold, score in enumerate(scores):
            mlflow.log_metric(f"fold_{fold}_accuracy", score)
        
        # Log summary statistics
        mlflow.log_metrics({
            "mean_cv_accuracy": scores.mean(),
            "std_cv_accuracy": scores.std(),
            "min_cv_accuracy": scores.min(),
            "max_cv_accuracy": scores.max()
        })
        
        # Log the model
        model.fit(X, y)  # Fit on full dataset
        mlflow.sklearn.log_model(model, "model")
        
        return scores
```

## CLI Commands Reference

```bash
# Tracking Server
mlflow server --help
mlflow ui  # Start UI on localhost:5000

# Runs
mlflow runs list --experiment-id 1
mlflow runs describe --run-id <run_id>
mlflow runs restore --run-id <run_id>

# Experiments
mlflow experiments list
mlflow experiments create --experiment-name "New Experiment"
mlflow experiments delete --experiment-id 1

# Models
mlflow models serve --help
mlflow models predict --help
mlflow models build-docker --help

# Projects
mlflow run --help
mlflow run . -P alpha=0.1

# Registry
mlflow models list-registered-models
mlflow models get-model-version --name "ModelName" --version 1
```

## Troubleshooting Common Issues

### Connection Issues

```python
# Test connection
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.get_experiment_by_name("Default")
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Storage Issues

```python
# Check artifact location
run = mlflow.get_run(run_id)
print(f"Artifact URI: {run.info.artifact_uri}")

# Download artifacts
mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model",
    dst_path="./downloads"
)
```

### Performance Optimization

```python
# Batch logging for better performance
with mlflow.start_run():
    # Instead of multiple log_metric calls
    metrics = {}
    for epoch in range(100):
        metrics[f"epoch_{epoch}_loss"] = calculate_loss(epoch)
    
    mlflow.log_metrics(metrics)  # Single batch call
```

# MLflow Complete Reference Card

## Overview

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Installation Guide

### Basic Installation

#### pip (recommended)

```bash
# Latest stable version
pip install mlflow

# With extra dependencies
pip install mlflow[extras]

# Specific version
pip install mlflow==2.8.1

# Development version
pip install git+https://github.com/mlflow/mlflow.git
```

#### conda

```bash
# From conda-forge
conda install -c conda-forge mlflow

# With additional packages
conda install -c conda-forge mlflow boto3 psycopg2
```

#### Docker

```bash
# Official MLflow image
docker pull ghcr.io/mlflow/mlflow

# Run MLflow server in container
docker run -p 5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
```

### Platform-Specific Installation

#### Windows

```powershell
# Using pip
pip install mlflow

# Using conda (recommended for Windows)
conda install -c conda-forge mlflow

# Add to PATH if needed
$env:PATH += ";C:\Users\YourUser\AppData\Local\Programs\Python\Python39\Scripts"
```

#### macOS

```bash
# Using Homebrew (if available)
brew install mlflow

# Using pip
pip3 install mlflow

# Using conda
conda install -c conda-forge mlflow

# For M1/M2 Macs, ensure compatibility
pip install --upgrade pip
pip install mlflow
```

#### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Install MLflow
pip3 install mlflow

# For system-wide installation
sudo pip3 install mlflow
```

#### CentOS/RHEL/Fedora

```bash
# Install dependencies
sudo yum install python3-pip python3-devel  # CentOS/RHEL
sudo dnf install python3-pip python3-devel  # Fedora

# Install MLflow
pip3 install mlflow
```

### Database Dependencies

```bash
# PostgreSQL
pip install psycopg2-binary

# MySQL
pip install PyMySQL

# SQL Server
pip install pyodbc

# All database backends
pip install mlflow[extras]
```

### Cloud Storage Dependencies

```bash
# AWS S3
pip install boto3

# Azure Blob Storage
pip install azure-storage-blob

# Google Cloud Storage
pip install google-cloud-storage

# All cloud backends
pip install mlflow[extras]
```

## Platform Configuration and Setup

### Windows Configuration

#### Environment Setup

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_DEFAULT_ARTIFACT_ROOT = "file:///C:/mlflow/artifacts"

# Command Prompt
set MLFLOW_TRACKING_URI=http://localhost:5000
set MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///C:/mlflow/artifacts

# Permanent environment variables
[Environment]::SetEnvironmentVariable("MLFLOW_TRACKING_URI", "http://localhost:5000", "User")
```

#### Windows Service Setup

```powershell
# Create batch file: mlflow_server.bat
@echo off
cd /d "C:\path\to\your\project"
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Install as Windows service using NSSM
nssm install MLflowService "C:\path\to\mlflow_server.bat"
nssm start MLflowService
```

#### Windows Dockerfile

```dockerfile
FROM python:3.9-windowsservercore-1809
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

### macOS Configuration

#### Environment Setup

```bash
# Bash/Zsh
echo 'export MLFLOW_TRACKING_URI=http://localhost:5000' >> ~/.zshrc
echo 'export MLFLOW_DEFAULT_ARTIFACT_ROOT=~/mlflow/artifacts' >> ~/.zshrc
source ~/.zshrc

# Fish shell
echo 'set -x MLFLOW_TRACKING_URI http://localhost:5000' >> ~/.config/fish/config.fish
```

#### macOS Service Setup (launchd)

```xml
<!-- ~/Library/LaunchAgents/com.mlflow.server.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlflow.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/mlflow</string>
        <string>server</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>5000</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.mlflow.server.plist
launchctl start com.mlflow.server
```

### Linux Configuration

#### Environment Setup

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_DEFAULT_ARTIFACT_ROOT=/home/$USER/mlflow/artifacts

# Create artifacts directory
mkdir -p ~/mlflow/artifacts

# Reload configuration
source ~/.bashrc
```

#### SystemD Service Setup

```ini
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/opt/mlflow
Environment=PYTHONPATH=/opt/mlflow
ExecStart=/usr/local/bin/mlflow server \
    --backend-store-uri sqlite:///opt/mlflow/mlflow.db \
    --default-artifact-root /opt/mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Setup service
sudo useradd -r -s /bin/false mlflow
sudo mkdir -p /opt/mlflow/artifacts
sudo chown -R mlflow:mlflow /opt/mlflow

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

#### Ubuntu/Debian Package Installation

```bash
# Create DEB package (optional)
sudo apt-get install build-essential devscripts debhelper
# ... package creation steps ...

# Direct installation from source
git clone https://github.com/mlflow/mlflow.git
cd mlflow
pip install -e .
```

## Docker Deployment Configurations

### Basic Docker Setup

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install mlflow psycopg2-binary boto3

# Create non-root user
RUN useradd -m -u 1000 mlflow
USER mlflow
WORKDIR /home/mlflow

# Expose port
EXPOSE 5000

# Start server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  minio:
    image: minio/minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"

  mlflow:
    build: .
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
      MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://mlflow/artifacts
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on:
      - postgres
      - minio
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root s3://mlflow/artifacts
      --host 0.0.0.0
      --port 5000

volumes:
  postgres_data:
  minio_data:
```

### Kubernetes Deployment

```yaml
# mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow-server
        image: mlflow/mlflow:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: "postgresql://user:pass@postgres:5432/mlflow"
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: "s3://mlflow-artifacts"
        command: ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
spec:
  selector:
    app: mlflow-server
  ports:
  - port: 5000
    targetPort: 5000
  type: LoadBalancer
```

## Cloud Platform Configurations

### AWS Setup

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# MLflow with S3 backend
mlflow server \
    --backend-store-uri postgresql://user:pass@rds-endpoint:5432/mlflow \
    --default-artifact-root s3://your-mlflow-bucket/artifacts \
    --host 0.0.0.0
```

#### AWS ECS Task Definition

```json
{
  "family": "mlflow-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/mlflowTaskRole",
  "containerDefinitions": [
    {
      "name": "mlflow-server",
      "image": "mlflow/mlflow:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MLFLOW_BACKEND_STORE_URI",
          "value": "postgresql://user:pass@rds-endpoint:5432/mlflow"
        },
        {
          "name": "MLFLOW_DEFAULT_ARTIFACT_ROOT",
          "value": "s3://your-bucket/artifacts"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mlflow-server",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Azure Setup

```bash
# Install Azure CLI
pip install azure-cli

# Login to Azure
az login

# Environment variables
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export AZURE_STORAGE_ACCOUNT="your_account"
export AZURE_STORAGE_ACCESS_KEY="your_key"

# MLflow with Azure backend
mlflow server \
    --backend-store-uri postgresql://user:pass@server:5432/mlflow \
    --default-artifact-root azure://container/path \
    --host 0.0.0.0
```

### Google Cloud Setup

```bash
# Install Google Cloud SDK
pip install google-cloud-storage

# Authenticate
gcloud auth application-default login

# Environment variables
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# MLflow with GCS backend
mlflow server \
    --backend-store-uri postgresql://user:pass@cloud-sql-proxy:5432/mlflow \
    --default-artifact-root gs://your-bucket/artifacts \
    --host 0.0.0.0
```

## Development Environment Setup

### Virtual Environment Setup

```bash
# Python venv
python -m venv mlflow-env
source mlflow-env/bin/activate  # Linux/macOS
mlflow-env\Scripts\activate     # Windows

# Install in virtual environment
pip install mlflow jupyter notebook

# Conda environment
conda create -n mlflow python=3.9
conda activate mlflow
conda install -c conda-forge mlflow

# Poetry
poetry init
poetry add mlflow
poetry shell
```

### Jupyter Integration

```python
# Jupyter notebook setup
%load_ext autoreload
%autoreload 2

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("jupyter_experiments")

# Start MLflow run in notebook
mlflow.start_run()
# ... your ML code ...
mlflow.end_run()
```

### IDE Configuration

#### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./mlflow-env/bin/python",
    "python.envFile": "${workspaceFolder}/.env",
    "python.terminal.activateEnvironment": true
}
```

#### PyCharm

```python
# Add to run configuration environment variables
MLFLOW_TRACKING_URI=http://localhost:5000
PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
```

### Configuration Files

```python
# config.py
import os
from pathlib import Path

class MLflowConfig:
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
    ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", str(Path.home() / "mlflow" / "artifacts"))
    
    # Database configuration
    DATABASE_URI = os.getenv("MLFLOW_DATABASE_URI", "sqlite:///mlflow.db")
    
    # S3 configuration
    S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
```

```yaml
# .env file
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=my_experiment
MLFLOW_ARTIFACT_ROOT=./artifacts
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

This expanded section provides comprehensive installation and configuration instructions for all major platforms and deployment scenarios.

## Core Components

### 1. MLflow Tracking

Track experiments, parameters, metrics, and artifacts.

#### Basic Tracking Setup

```python
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts

# Set tracking URI (optional)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")
```

#### Run Context Management

```python
# Method 1: Context manager (recommended)
with mlflow.start_run():
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.786)
    
# Method 2: Manual start/end
run = mlflow.start_run()
mlflow.log_param("alpha", 0.1)
mlflow.end_run()

# Method 3: Active run
mlflow.start_run()
# ... logging code ...
mlflow.end_run()
```

#### Logging Functions

```python
# Parameters (hyperparameters, configs)
mlflow.log_param("learning_rate", 0.01)
mlflow.log_params({"batch_size": 32, "epochs": 100})

# Metrics (model performance)
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metrics({"precision": 0.92, "recall": 0.88})

# Step-wise metrics (for tracking over epochs)
for epoch in range(100):
    mlflow.log_metric("loss", loss_value, step=epoch)

# Artifacts (files, models, plots)
mlflow.log_artifact("model.pkl")
mlflow.log_artifacts("output_dir")

# Text
mlflow.log_text("Some important note", "notes.txt")

# Dictionary as JSON
mlflow.log_dict({"key": "value"}, "config.json")
```

#### Tags and Notes

```python
# Set tags
mlflow.set_tag("model_type", "random_forest")
mlflow.set_tags({"version": "1.0", "team": "data_science"})

# Add notes
mlflow.set_tag("mlflow.note.content", "This is a baseline model")
```

### 2. MLflow Models

Standardized model packaging and deployment.

#### Model Flavors

```python
# Scikit-learn
import mlflow.sklearn
mlflow.sklearn.log_model(model, "model")
loaded_model = mlflow.sklearn.load_model("runs:/{}/model".format(run_id))

# PyTorch
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow/Keras
import mlflow.tensorflow
import mlflow.keras
mlflow.keras.log_model(model, "model")

# XGBoost
import mlflow.xgboost
mlflow.xgboost.log_model(model, "model")

# LightGBM
import mlflow.lightgbm
mlflow.lightgbm.log_model(model, "model")

# Statsmodels
import mlflow.statsmodels
mlflow.statsmodels.log_model(model, "model")

# Spark ML
import mlflow.spark
mlflow.spark.log_model(model, "model")

# Custom Python function
import mlflow.pyfunc
mlflow.pyfunc.log_model("model", python_model=custom_model)
```

#### Custom Model Example

```python
import mlflow.pyfunc

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log custom model
with mlflow.start_run():
    wrapped_model = ModelWrapper(trained_model)
    mlflow.pyfunc.log_model(
        "custom_model", 
        python_model=wrapped_model,
        registered_model_name="MyCustomModel"
    )
```

#### Model Signature

```python
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec

# Infer signature automatically
signature = infer_signature(X_train, y_pred)

# Manual signature definition
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
    ColSpec("string", "category")
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 3. MLflow Projects

Reproducible ML code packaging.

#### MLproject File

```yaml
name: My Project
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.1}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"
    
  validate:
    parameters:
      model_uri: string
    command: "python validate.py {model_uri}"
```

#### Running Projects

```python
# Run local project
mlflow.run(".", parameters={"alpha": 0.5})

# Run from GitHub
mlflow.run(
    "https://github.com/user/repo.git",
    parameters={"alpha": 0.5}
)

# Run specific entry point
mlflow.run(".", entry_point="validate", parameters={"model_uri": "runs:/abc/model"})
```

### 4. MLflow Model Registry

Central model store for lifecycle management.

#### Registering Models

```python
# Register during logging
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="ProductionModel"
)

# Register existing run
model_uri = "runs:/{}/model".format(run_id)
mlflow.register_model(model_uri, "ProductionModel")

# Register with version description
result = mlflow.register_model(
    model_uri,
    "ProductionModel",
    description="Model trained on latest dataset"
)
```

#### Model Versions and Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List registered models
models = client.list_registered_models()

# Get model versions
versions = client.get_registered_model("ProductionModel").latest_versions

# Transition model stage
client.transition_model_version_stage(
    name="ProductionModel",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# Update model version
client.update_model_version(
    name="ProductionModel",
    version=1,
    description="Updated description"
)

# Delete model version
client.delete_model_version("ProductionModel", version=1)
```

#### Loading Models from Registry

```python
# Load latest version
model = mlflow.pyfunc.load_model("models:/ProductionModel/latest")

# Load specific version
model = mlflow.pyfunc.load_model("models:/ProductionModel/1")

# Load by stage
model = mlflow.pyfunc.load_model("models:/ProductionModel/Production")
```

## Advanced Features

### Autologging

```python
# Enable autologging for specific libraries
mlflow.sklearn.autolog()
mlflow.keras.autolog()
mlflow.pytorch.autolog()
mlflow.xgboost.autolog()

# Enable for all supported libraries
mlflow.autolog()

# Disable autologging
mlflow.sklearn.autolog(disable=True)

# Configure autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False  # Don't log models automatically
)
```

### Parent and Child Runs

```python
# Parent run
with mlflow.start_run() as parent_run:
    mlflow.log_param("experiment_type", "hyperparameter_tuning")
    
    # Child runs for different configurations
    for alpha in [0.1, 0.5, 1.0]:
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_param("alpha", alpha)
            # Train and log model...
```

### Search and Query Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Get run data
run = client.get_run(run_id)
print(f"Status: {run.info.status}")
print(f"Start time: {run.info.start_time}")
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")

# Get metric history
history = client.get_metric_history(run_id, "loss")
```

### Experiment Management

```python
# Create experiment
experiment_id = mlflow.create_experiment(
    "New Experiment",
    artifact_location="s3://bucket/artifacts",
    tags={"team": "ml", "project": "recommendation"}
)

# Get experiment
experiment = mlflow.get_experiment(experiment_id)

# List experiments
experiments = mlflow.list_experiments()

# Set experiment by name
mlflow.set_experiment("My Experiment")

# Delete experiment
mlflow.delete_experiment(experiment_id)
```

## Deployment Options

### Local Model Serving

```bash
# Serve model locally
mlflow models serve -m models:/ProductionModel/1 -p 5001

# Serve with conda environment
mlflow models serve -m models:/ProductionModel/1 --env-manager conda

# Build Docker image
mlflow models build-docker -m models:/ProductionModel/1 -n my-model

# Generate Dockerfile
mlflow models generate-dockerfile -m models:/ProductionModel/1 -d ./dockerfile_dir
```

### Cloud Deployments

```python
# Deploy to AWS SageMaker
mlflow.deployments.deploy(
    "sagemaker",
    model_uri="models:/ProductionModel/1",
    config={
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "region_name": "us-west-2"
    }
)

# Deploy to Azure ML
mlflow.deployments.deploy(
    "azureml",
    model_uri="models:/ProductionModel/1",
    config={
        "compute_type": "ACI",
        "location": "eastus2"
    }
)
```

### Batch Inference

```python
# Apply model to Spark DataFrame
predictions = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri="models:/ProductionModel/1"
)

df_with_predictions = df.withColumn("predictions", predictions(*feature_cols))
```

## Configuration and Setup

### Tracking Server Setup

```bash
# Start local tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# With remote artifact store
mlflow server \
    --backend-store-uri postgresql://user:pass@host:port/db \
    --default-artifact-root s3://bucket/artifacts \
    --host 0.0.0.0
```

### Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_EXPERIMENT_NAME=default
```

### Configuration File

```python
# mlflow_config.py
import mlflow
import os

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))
    
    # Configure autologging
    mlflow.autolog()
```

## Best Practices and Patterns

### Experiment Organization

```python
# Use descriptive experiment names
mlflow.set_experiment(f"recommendation_model_{datetime.now().strftime('%Y%m%d')}")

# Use consistent tagging strategy
tags = {
    "model_type": "collaborative_filtering",
    "data_version": "v2.1",
    "feature_set": "user_item_interactions",
    "environment": "production"
}
mlflow.set_tags(tags)

# Log important context
mlflow.log_param("git_commit", get_git_commit())
mlflow.log_param("dataset_size", len(train_data))
mlflow.log_param("feature_count", X_train.shape[1])
```

### Model Validation Pattern

```python
def train_and_validate_model(params):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = train_model(params)
        
        # Validate
        train_score = evaluate_model(model, X_train, y_train)
        val_score = evaluate_model(model, X_val, y_val)
        
        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "overfitting": train_score - val_score
        })
        
        # Log model if performance is good
        if val_score > 0.8:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="BestModel" if val_score > 0.9 else None
            )
        
        return model, val_score
```

### Pipeline Integration

```python
class MLflowPipeline:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        
    def run_experiment(self, config):
        with mlflow.start_run() as run:
            # Data preparation
            X_train, X_val, y_train, y_val = self.prepare_data(config)
            mlflow.log_params(config["data"])
            
            # Feature engineering
            features = self.engineer_features(X_train, config["features"])
            mlflow.log_param("feature_count", len(features))
            
            # Model training
            model = self.train_model(X_train, y_train, config["model"])
            mlflow.log_params(config["model"])
            
            # Evaluation
            metrics = self.evaluate_model(model, X_val, y_val)
            mlflow.log_metrics(metrics)
            
            # Model logging
            mlflow.sklearn.log_model(model, "model")
            
            return run.info.run_id
```

### Error Handling and Cleanup

```python
def safe_mlflow_run(training_function, **kwargs):
    run_id = None
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            result = training_function(**kwargs)
            mlflow.log_metrics({"success": 1})
            return result
            
    except Exception as e:
        if run_id:
            # Log error information
            mlflow.log_param("error_message", str(e))
            mlflow.log_metrics({"success": 0})
        
        # End run gracefully
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
```

## Common Use Cases and Examples

### Hyperparameter Tuning with Optuna

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        # Train and evaluate
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        mlflow.log_metric('accuracy', score)
        
        return score

# Run optimization
study = optuna.create_study(direction='maximize')
mlflc = MLflowCallback(tracking_uri="http://localhost:5000", metric_name="accuracy")

with mlflow.start_run():
    study.optimize(objective, n_trials=100, callbacks=[mlflc])
    
    # Log best parameters
    mlflow.log_params(study.best_params)
    mlflow.log_metric('best_accuracy', study.best_value)
```

### A/B Testing Model Comparison

```python
def compare_models():
    models = {
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC()
    }
    
    results = {}
    
    with mlflow.start_run(run_name="model_comparison"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Log results
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "train_accuracy": train_score,
                    "test_accuracy": test_score
                })
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                results[name] = test_score
        
        # Log comparison results
        best_model = max(results.items(), key=lambda x: x[1])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_accuracy", best_model[1])
    
    return results
```

### Cross-Validation Tracking

```python
from sklearn.model_selection import cross_val_score

def cross_validate_with_mlflow(model, X, y, cv=5):
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Log individual fold scores
        for fold, score in enumerate(scores):
            mlflow.log_metric(f"fold_{fold}_accuracy", score)
        
        # Log summary statistics
        mlflow.log_metrics({
            "mean_cv_accuracy": scores.mean(),
            "std_cv_accuracy": scores.std(),
            "min_cv_accuracy": scores.min(),
            "max_cv_accuracy": scores.max()
        })
        
        # Log the model
        model.fit(X, y)  # Fit on full dataset
        mlflow.sklearn.log_model(model, "model")
        
        return scores
```

## CLI Commands Reference

```bash
# Tracking Server
mlflow server --help
mlflow ui  # Start UI on localhost:5000

# Runs
mlflow runs list --experiment-id 1
mlflow runs describe --run-id <run_id>
mlflow runs restore --run-id <run_id>

# Experiments
mlflow experiments list
mlflow experiments create --experiment-name "New Experiment"
mlflow experiments delete --experiment-id 1

# Models
mlflow models serve --help
mlflow models predict --help
mlflow models build-docker --help

# Projects
mlflow run --help
mlflow run . -P alpha=0.1

# Registry
mlflow models list-registered-models
mlflow models get-model-version --name "ModelName" --version 1
```

## Troubleshooting Common Issues

### Connection Issues

```python
# Test connection
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.get_experiment_by_name("Default")
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Storage Issues

```python
# Check artifact location
run = mlflow.get_run(run_id)
print(f"Artifact URI: {run.info.artifact_uri}")

# Download artifacts
mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model",
    dst_path="./downloads"
)
```

### Performance Optimization

```python
# Batch logging for better performance
with mlflow.start_run():
    # Instead of multiple log_metric calls
    metrics = {}
    for epoch in range(100):
        metrics[f"epoch_{epoch}_loss"] = calculate_loss(epoch)
    
    mlflow.log_metrics(metrics)  # Single batch call
```

