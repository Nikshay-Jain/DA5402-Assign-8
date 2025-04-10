# src/train.py

import mlflow, mlflow.keras, logging, os
from datetime import datetime

from tensorflow import keras
from utils import *

# Create logs directory if it doesn't exist
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
log_filename = f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, log_filename)),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train_model(split_seed=42):
    with mlflow.start_run():
        try:
            logger.info("Preparing data...")
            (train_data, val_data, test_data), char_to_num, num_to_char = get_data(seed=split_seed)

            logger.info("Building model...")
            model = build_model()

            model.compile(optimizer=keras.optimizers.Adam(), loss=keras.backend.ctc_batch_cost)

            logger.info("Training model...")
            history = model.fit(train_data, validation_data=val_data, epochs=20)

            logger.info("Logging model to MLflow...")
            mlflow.keras.log_model(model, "model")
            mlflow.log_param("seed", split_seed)

            logger.info("Logging training metrics to MLflow...")
            for epoch, loss in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss", loss, step=epoch)
            for epoch, val_loss in enumerate(history.history['val_loss']):
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            logger.info("Calculating and logging edit distance...")
            avg_edit_distance = calculate_edit_distance(model, test_data, num_to_char)
            mlflow.log_metric("avg_edit_distance", avg_edit_distance)

            logger.info("Saving and logging plots...")
            plot_metrics(history)
            mlflow.log_artifact("metrics.png")

            logger.info("Training completed successfully.")

        except Exception as e:
            logger.error("Training failed: %s", str(e), exc_info=True) #include stack trace
            raise

if __name__ == "__main__":
    for seed in [42, 1337, 2025]:
        train_model(split_seed=seed)