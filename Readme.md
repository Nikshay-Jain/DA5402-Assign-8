# Handwriting Recognition with MLflow

This project implements a handwriting recognition system using TensorFlow/Keras and MLflow for experiment tracking and model deployment. The implementation is based on the Keras example: https://keras.io/examples/vision/handwriting_recognition/

## Requirements

- Python 3.8+
- TensorFlow 2.9+
- MLflow 2.4+
- Flask (for API)
- NumPy, Matplotlib, Pillow

## Project Structure

```
.
├── conda.yaml                     # Conda environment file for MLproject
├── handwriting_recognition_mlflow.py  # Main script for model training with MLflow tracking
├── handwriting_api.py             # API server script for serving the model
├── inference.py                   # Inference script for MLproject
├── MLproject                      # MLproject file
└── README.md                      # This file
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
pip install mlflow tensorflow numpy matplotlib pillow

# Start MLflow UI
mlflow ui

# In a separate terminal, run the training script
python handwriting_recognition_mlflow.py
```

The script will run 3 training sessions with different data splits and log the results to MLflow.

## Task 2: MLflow API for Handwriting Recognition

The `handwriting_api.py` script implements a Flask-based REST API that serves the handwriting recognition model.

### Running Task 2
```bash
# Start the API server
python handwriting_api.py
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