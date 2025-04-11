import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import mlflow.tensorflow
from tensorflow.keras import layers
import base64
from PIL import Image
import os, io, logging, datetime

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

app = Flask(__name__)

# Constants
img_width = 128
img_height = 32
char_vocabulary = list("abcdefghijklmnopqrstuvwxyz0123456789")
num_to_char = layers.StringLookup(
    vocabulary=char_vocabulary, mask_token=None, invert=True
)

# Load the model
try:
    # Get the latest model version from MLflow
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("handwriting_recognition_model", stages=["None"])[0].version
    logger.info(f"Loading model version: {latest_version}")
    
    model_uri = f"models:/handwriting_recognition_model/{latest_version}"
    model = mlflow.tensorflow.load_model(model_uri)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


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


def preprocess_image(image):
    """Preprocess the image for model input."""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to target dimensions
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


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict text from handwriting image."""
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            if 'image' in request.form:
                # Handle base64 encoded image
                encoded_image = request.form['image']
                image_data = base64.b64decode(encoded_image.split(',')[1] if ',' in encoded_image else encoded_image)
                image = Image.open(io.BytesIO(image_data))
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            # Handle file upload
            image_file = request.files['image']
            image = Image.open(image_file)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Decode prediction
        predicted_text = decode_prediction(predictions)
        
        return jsonify({
            'predicted_text': predicted_text
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)