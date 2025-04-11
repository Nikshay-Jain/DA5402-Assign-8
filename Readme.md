# DA5402 Assign 8
## Nikshay Jain | MM21B044

This project implements a handwriting recognition system using TensorFlow/Keras and MLflow for experiment tracking and model deployment. The implementation is based on the Keras example: https://keras.io/examples/vision/handwriting_recognition/

## Requirements
- Python 3.8+
- TensorFlow 2.9+
- MLflow 2.4+
- Flask (for API)
- NumPy, Matplotlib, Pillow

## Project Structure
```
DA5402-Assign-8/
├── Readme.md                      # Documentation for setup, usage, and structure
├── src/                           # Source code directory
│   ├── task_1.py                  # Script to train the CRNN model with MLflow experiment tracking
│   ├── task_2.py                  # REST API server using FastAPI/Flask to serve the trained model
│   ├── conda.yaml                 # Environment file specific to scripts in src/
│   └── MLProject                  # Optional MLProject file for local runs from src/
|
├── data/                          # Directory for storing datasets (e.g., IAM Handwriting Dataset)
│   ├── IAM_Words.zip              # Original dataset archive
│   ├── IAM_Words_extracted/       # Extracted IAM dataset (images, labels)
│   └── processed/                 # Preprocessed/cleaned data (e.g., TFRecords or .npy)
|
├── logs/                          # Logging output from training and evaluation
│   └── training_log_YYYY-MM-DD_HH-MM-SS.log  # Timestamped log files
|
├── plots/                         # Generated visualizations (e.g., loss curves, predictions)
│   ├── train_loss_plot.png
│   ├── val_loss_plot.png
│   └── sample_predictions.png
|
├── mlruns/                        # MLflow-generated folder for experiment tracking
│   └── [MLflow run metadata and artifacts]
|
├── models/                        # Saved trained model artifacts
│   └── handwriting_model/         # Keras/TensorFlow saved model directory
|
├── test_examples/                 # Sample images for testing model inference
│   ├── test_example_0.png
│   └── test_example_1.png
|
├── inference.py                   # Script to run inference using a saved model (loads images, predicts text)
├── requirements.txt               # Contains dependencies
└── .gitignore                     # Patterns to exclude files from version control (e.g., `mlruns/`, `logs/`, `*.zip`)
```

## Task 1: MLflow Experiment Tracking

The `handwriting_recognition_mlflow.py` script handles the model training with MLflow tracking.

**Features:**
- Logging of parameters, metrics, and artifacts
- Tracking of training and validation losses
- Tracking of Average Edit Distance per epoch
- Model versioning and registration in MLflow Registry
- Visualization of loss and edit distance metrics
- Training with different train-val-test splits

### Running Task 1
```bash
# Install required packages
pip install -r requirements.txt

# Start MLflow UI
mlflow ui

# In a separate terminal, run the training script
python src\task_1.py
```

The script will run 3 training sessions with different data splits and log the results to MLflow.

## Task 2: MLflow API for Handwriting Recognition

The `task_2.py` script implements a Flask-based REST API that serves the handwriting recognition model.

### Running Task 2
```bash
# Start the API server
python src\task_2.py
```

You can test the API using curl or Postman:

```bash
# Using curl with a file
curl -X POST -F "image=@test_examples/test_example_0.png" http://localhost:5000/predict

# Using curl with base64 encoded image
curl -X POST -F "image=$(base64 -w 0 test_examples/test_example_0.png)" http://localhost:5000/predict
```

## Task 3: MLproject for Inference

The `MLproject` file and `inference.py` script implement an MLflow project for performing inference on handwritten images.

### Running Task 3
```bash
# Run the MLproject with a specified image
mlflow run . -P image_path=test_examples/test_example_0.png

# Or use the default image
mlflow run .
```

## Visualizing Results

To view the experiment tracking results and plots:

1. Start the MLflow UI: `mlflow ui`
2. Open your web browser to `http://localhost:5000`