# DA5402 - Assignment 8  
## Nikshay Jain | MM21B044  

This project implements a complete pipeline for handwriting recognition using the IAM dataset. The system includes CRNN model training with MLflow tracking, REST API for prediction, and an MLproject setup for seamless deployment.

The base implementation is adapted from the official Keras example:  
🔗 https://keras.io/examples/vision/handwriting_recognition/

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
   - [Task 1: Model Training with MLflow](#task-1-model-training-with-mlflow)
   - [Task 2: REST API Server](#task-2-rest-api-server)
   - [Task 3: MLProject Inference](#task-3-mlproject-inference)
5. [Results](#results)
6. [Closing Note](#closing-note)

---

## Project Overview

The system uses a **Convolutional Recurrent Neural Network (CRNN)** to transcribe handwritten text images into strings. The training pipeline includes:
- Character-level encoding with CTC loss.
- MLflow logging for all model metrics, artifacts, and configurations.
- Logging of edit distance and validation plots.
- REST API for real-time prediction.
- MLproject definition for portability.

---

## Directory Structure

```
DA5402-Assign-8/
├── conda.yaml                       # Global conda environment file
├── Readme.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
│
├── src/                             # Source code directory
│   ├── task_1.py                    # Script to train the model with MLflow tracking
│   ├── task_2.py                    # Flask-based API server for inference
│   ├── conda.yaml                   # Env file used within src for MLproject
│   └── MLProject                    # MLproject file for running tasks from src
│
├── data/                            # Data directory for raw/extracted datasets
│   ├── IAM_Words.zip                # Original dataset archive
│   ├── IAM_Words_extracted/        # Extracted IAM dataset (images & metadata)
│   └── processed/                   # Preprocessed TF-ready data
│
├── logs/                            # Training and evaluation logs
│   └── training_log_YYYY-MM-DD_HH-MM-SS.log
│
├── plots/                           # Visualizations
│   ├── train_loss_plot.png
│   ├── val_loss_plot.png
│   └── sample_predictions.png
│
├── mlruns/                          # MLflow logs and artifacts
│   └── <experiment_id>/...
│
├── models/                          # Saved model artifacts
│   └── handwriting_model/           # TensorFlow SavedModel format
│
├── test_examples/                   # Sample images for API and inference testing
│   ├── test_example_0.png
│   └── test_example_1.png
│
├── inference.py                     # Script to test inference using MLproject
└── MLproject                        # MLflow project definition file
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Nikshay-Jain/DA5402-Assign-8.git
cd DA5402-Assign-8
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # On Linux/Mac
.\.venv\Scripts\activate        # On Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

## Usage

### Task 1: Model Training with MLflow

This task handles model training, MLflow logging, edit distance metrics, and saving visualizations.
If the execution fails, you might need to manually unzip some folders in the data directory.

```bash
# Start MLflow UI
mlflow ui
# Visit http://localhost:5000

# Run the training script
python src/task_1.py
```

- Logs training/validation loss & mean edit distance.
- Saves the best model under `models/handwriting_model/`.
- Saves plots to `plots/`.
- Logs metadata in `mlruns/`.

---

### Task 2: REST API Server

The script `task_2.py` spins up a Flask server for real-time handwriting prediction.

```bash
# Start the Flask API server
python src/task_2.py
```

#### Example Usage via curl:
```bash
# Upload image file
curl -X POST -F "image=@test_examples/test_example_0.png" http://localhost:5000/predict
```

---

### Task 3: MLproject Inference

The project is also defined as a portable MLproject, allowing inference to be run via MLflow CLI.

```bash
# Predict a given image
mlflow run . -P image_path=test_examples/test_example_0.png

# Or default
mlflow run .
```

---

## Results

### Sample Metrics (from MLflow logs)
| Model | Mean Edit Distance |
|-------|--------------------|
| Run 1 | 0.67               |
| Run 2 | 0.50               |
| Run 3 | 0.48               |

### Saved Artifacts
- Plots (Loss Curves, Predictions): `plots/`
- Logs: `logs/training_log_*.log`
- Model: `models/handwriting_model/`
- MLflow: `mlruns/`

---

## Closing Note

To debug or explore training progress:
- Refer to `logs/` for step-wise debug logs.
- Use `mlflow ui` for complete experiment dashboards.
- Edit and rerun `task_1.py` with modified split parameters for more insights.

For deployment or testing, the REST API or `inference.py` under MLproject covers all scenarios.

---