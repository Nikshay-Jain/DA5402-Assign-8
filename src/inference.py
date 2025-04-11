import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import mlflow.tensorflow
import logging
from PIL import Image
import os, datetime

# Ensure logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Create a log file name with a timestamp
log_filename = os.path.join(
    logs_dir, f"training_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
img_width = 128
img_height = 32
char_vocabulary = list("abcdefghijklmnopqrstuvwxyz0123456789")
num_to_char = layers.StringLookup(
    vocabulary=char_vocabulary, mask_token=None, invert=True
)


def decode_prediction(pred):
    """Decode the prediction."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Get back the text
    res = tf.gather(results[0], tf.where(tf.not_equal(results[0], -1)))
    res = tf.squeeze(res)
    res = num_to_char(res)
    res = tf.strings.reduce_join(res).numpy().decode("utf-8")
    return res


def preprocess_image(image_path):
    """Preprocess the image for model input."""
    # Load image
    image = Image.open(image_path)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to array
    img_array = np.array(image)
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Reshape for the model
    img_array = tf.image.resize_with_pad(
        tf.expand_dims(img_array, axis=-1), 
        target_height=img_height, 
        target_width=img_width
    )
    
    # Add batch dimension
    img_array = tf.expand_dims(img_array, axis=0)
    
    return img_array


def main(image_path):
    """Main function to load model and perform inference."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        # Get the latest model version from MLflow
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("handwriting_recognition_model", stages=["None"])[0].version
        logger.info(f"Loading model version: {latest_version}")
        
        # Load the model
        model_uri = f"models:/handwriting_recognition_model/{latest_version}"
        model = mlflow.tensorflow.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Decode prediction
        predicted_text = decode_prediction(predictions)
        
        logger.info(f"Predicted text: {predicted_text}")
        print(f"\nPredicted text: {predicted_text}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting Recognition Inference")
    parser.add_argument("-P", dest="image_path", type=str, required=True,
                        help="Path to the handwritten image file")
    
    args = parser.parse_args()
    main(args.image_path)