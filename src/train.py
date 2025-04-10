import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# Keras 3 imports. Adjust if using older Keras/TF-Keras
from keras import layers
from keras import ops # Keras 3 operations module
from keras.layers import StringLookup
from keras.utils import set_random_seed
from tensorflow.keras.backend import ctc_batch_cost

import mlflow
import mlflow.keras # Or mlflow.tensorflow
from sklearn.model_selection import train_test_split # Used for splitting indices if needed


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

# Default image dimensions from the Keras example
IMG_WIDTH = 200
IMG_HEIGHT = 50

# --- Data Handling Functions ---

def download_and_extract_data():
    """
    Checks for IAM dataset presence.
    (Assumes manual download or add download commands here).
    """
    logger.info("Checking for IAM Dataset...")
    base_path = "data" # Assuming data directory exists in project root
    words_list_path = os.path.join(base_path, "words.txt")
    words_folder_path = os.path.join(base_path, "words") # Folder containing extracted images

    if not os.path.exists(words_list_path) or not os.path.exists(words_folder_path):
         logger.error(f"Dataset not found. Expected words.txt at '{words_list_path}' and image folder at '{words_folder_path}'.")
         logger.error("Please download the IAM dataset ('words.tgz' and 'words.txt') and place them accordingly.")
         # Example download commands (uncomment and adapt if needed):
         logger.info("Attempting to download dataset...")
         try:
             os.makedirs(base_path, exist_ok=True)
             os.system("wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip -P data/")
             os.system("unzip -qq data/IAM_Words.zip -d data/")
             os.makedirs(words_folder_path, exist_ok=True)
             os.system(f"tar -xf data/IAM_Words/words.tgz -C {words_folder_path}")
             os.system(f"mv data/IAM_Words/words.txt {words_list_path}")
             os.system("rm -rf data/IAM_Words data/IAM_Words.zip") # Clean up archive/intermediate folder
             logger.info("Dataset downloaded and extracted.")
         except Exception as e:
             logger.error(f"Automatic dataset download failed: {e}")
             raise FileNotFoundError("Dataset download/extraction failed.")
         raise FileNotFoundError("Dataset not found and automatic download not enabled/failed.")
    else:
        logger.info("Dataset found.")
    return base_path

def get_image_paths_and_labels(samples, base_image_path):
    """Extracts image paths and labels from the samples list."""
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        try:
            line_split = file_line.strip().split(" ")
            image_name = line_split[0]
            # Construct image path (e.g., data/words/a01/a01-000u/a01-000u-00-00.png)
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(
                base_image_path, partI, partI + "-" + partII, image_name + ".png"
            )
            if os.path.getsize(img_path): # Check if file is not empty
                paths.append(img_path)
                corrected_samples.append(file_line.strip()) # Keep the original line info
            else: # Optional: log skipped empty files
                logger.debug(f"Skipping empty image file: {img_path}")
        except Exception as e:
            logger.warning(f"Error processing line: '{file_line}'. Error: {e}. Skipping.")
            continue # Skip this line if error occurs
    return paths, corrected_samples

def clean_labels(labels):
    """Extracts the actual text label from the sample line."""
    cleaned_labels = []
    for label in labels:
        try:
            text = label.split(" ")[-1].strip()
            cleaned_labels.append(text)
        except IndexError:
             logger.warning(f"Could not parse label from line: '{label}'. Skipping.")
    return cleaned_labels

def distortion_free_resize(image, img_size):
    """Resize image without distortion, maintaining aspect ratio."""
    w, h = img_size
    image = ops.cast(image, dtype=tf.float32) # Cast to float for resizing ops

    # Determine target aspect ratio and current aspect ratio
    target_aspect = w / h
    current_height, current_width = tf.shape(image)[0], tf.shape(image)[1]
    current_aspect = tf.cast(current_width, tf.float32) / tf.cast(current_height, tf.float32)

    # Resize based on aspect ratio comparison
    if current_aspect > target_aspect:
        # Width is the limiting factor
        new_width = tf.cast(w, tf.float32)
        new_height = tf.cast(new_width / current_aspect, tf.float32)
    else:
        # Height is the limiting factor
        new_height = tf.cast(h, tf.float32)
        new_width = tf.cast(new_height * current_aspect, tf.float32)

    # Resize the image
    image = tf.image.resize(image, size=(tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)), preserve_aspect_ratio=True)

    # Pad the image to the target size
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Handle potential off-by-one padding needs
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    image = tf.pad(image, paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0) # Pad with black (0)

    # Ensure the final shape is exactly the target size
    image = tf.image.resize(image, size=(h, w)) # Final resize just in case padding/rounding caused slight deviation

    image = tf.cast(image, tf.uint8) # Cast back to uint8 if necessary downstream
    return image


def encode_single_sample(img_path, label, img_height, img_width, char_to_num, max_label_length):
    """Encodes a single sample: read image, process, vectorize label."""
    try:
        # Load and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)  # Ensure grayscale
        img = distortion_free_resize(img, (img_width, img_height))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize image

        # Process and encode label
        label_tensor = tf.strings.unicode_split(label, 'UTF-8')
        label_encoded = char_to_num(label_tensor)
        
        # Pad or truncate label to ensure consistent length
        label_encoded = tf.pad(label_encoded, [[0, max_label_length - tf.shape(label_encoded)[0]]], constant_values=-1)  # Use -1 for padding
        label_encoded = label_encoded[:max_label_length]  # Truncate if longer than max_label_length

        return {
            "image": img,
            "label": label_encoded,
            "valid": tf.constant(True)
        }
    except Exception as e:
        logger.error(f"Failed to process sample: {img_path}. Error: {e}")
        return {
            "image": tf.zeros((img_width, img_height, 1), dtype=tf.float32),
            "label": tf.constant([], dtype=tf.int64),
            "valid": tf.constant(False)
        }

def prepare_dataset(image_paths, labels, batch_size, img_height, img_width, char_to_num):
    AUTOTUNE = tf.data.AUTOTUNE
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    dataset = dataset.map(
        lambda x, y: encode_single_sample(x, y, img_height, img_width, char_to_num),
        num_parallel_calls=AUTOTUNE
    )
    
    # Filter invalid samples
    dataset = dataset.filter(lambda sample: sample["valid"])
    
    # Format data for the CTC model which expects TWO inputs and a dummy output
    dataset = dataset.map(
        lambda sample: (
            # First element: dictionary with both inputs needed by the model
            {
                "image": sample["image"], 
                "label": sample["label"]
            },
            # Second element: dummy output (model uses CTC loss internally)
            sample["label"]  # Dummy target tensor (not actually used)
        ),
        num_parallel_calls=AUTOTUNE
    )
    
    # Batch and prefetch with padding for variable-length labels
    dataset = dataset.batch(
        batch_size,
        padded_shapes=(
            {"image": [img_width, img_height, 1], "label": [None]},  # Pad labels to max length in batch
            [None]  # Pad dummy outputs (same as labels)
        )
    )
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def load_and_preprocess_data(base_path, batch_size, img_height, img_width, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Loads dataset, cleans labels, splits data, prepares datasets."""
    logger.info("Loading and preprocessing data...")
    set_random_seed(random_seed) # For reproducibility of shuffling/splitting

    words_list = []
    words_path = os.path.join(base_path, "words.txt")
    try:
        with open(words_path, "r") as f:
            words = f.readlines()
        for line in words:
            if line.startswith("#"):
                continue
            # Basic check for format and error tag
            parts = line.strip().split(" ")
            if len(parts) >= 2 and parts[1] != "err":
                words_list.append(line.strip())
            # else: # Optional: log skipped lines
                # logger.debug(f"Skipping invalid/errored line: '{line.strip()}'")

    except FileNotFoundError:
        logger.error(f"'words.txt' not found at {words_path}. Cannot load data.")
        raise
    except Exception as e:
        logger.error(f"Error reading 'words.txt': {e}")
        raise

    logger.info(f"Total valid samples found: {len(words_list)}")
    np.random.shuffle(words_list) # Shuffle before splitting

    # Calculate split indices based on ratios
    total_size = len(words_list)
    if total_size == 0:
        raise ValueError("No valid samples loaded from words.txt.")

    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    train_samples = words_list[:train_end]
    validation_samples = words_list[train_end:val_end]
    # Ensure test ratio consistency (handle potential rounding issues)
    actual_test_ratio = 1.0 - (len(train_samples) + len(validation_samples)) / total_size
    logger.info(f"Requested splits: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    test_samples = words_list[val_end:]
    logger.info(f"Actual data split: Train={len(train_samples)}, Validation={len(validation_samples)}, Test={len(test_samples)} (Ratio: {actual_test_ratio:.2f})")

    base_image_path = os.path.join(base_path, "words")

    # Get paths and filter samples based on existing image files
    train_img_paths, train_samples = get_image_paths_and_labels(train_samples, base_image_path)
    validation_img_paths, validation_samples = get_image_paths_and_labels(validation_samples, base_image_path)
    test_img_paths, test_samples = get_image_paths_and_labels(test_samples, base_image_path)
    logger.info(f"Samples after checking image files: Train={len(train_img_paths)}, Validation={len(validation_img_paths)}, Test={len(test_img_paths)}")


    # Clean labels to get just the text
    train_labels_cleaned = clean_labels(train_samples)
    validation_labels_cleaned = clean_labels(validation_samples)
    test_labels_cleaned = clean_labels(test_samples)

    # --- Build Vocabulary and Mappings (from training data only) ---
    characters = set()
    max_len = 0
    for label in train_labels_cleaned:
        for char in label:
            characters.add(char)
        max_len = max(max_len, len(label))

    characters = sorted(list(characters))
    logger.info(f"Vocabulary Size (from training data): {len(characters)}")
    logger.info(f"Maximum Label Length (from training data): {max_len}")
    mlflow.log_param("vocabulary_size", len(characters))
    mlflow.log_param("max_label_length", max_len)


    # Mapping characters to integers
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None, name="char_to_num_lookup")
    # Mapping integers back to original characters
    num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, name="num_to_char_lookup")

    # --- Create tf.data datasets ---
    logger.info("Creating tf.data datasets...")
    try:
        train_dataset = prepare_dataset(train_img_paths, train_labels_cleaned, batch_size, img_height, img_width, char_to_num)
        validation_dataset = prepare_dataset(validation_img_paths, validation_labels_cleaned, batch_size, img_height, img_width, char_to_num)
        test_dataset = prepare_dataset(test_img_paths, test_labels_cleaned, batch_size, img_height, img_width, char_to_num)
        logger.info("Datasets created successfully.")
    except Exception as e:
        logger.error(f"Failed to create tf.data datasets: {e}", exc_info=True)
        raise

    return train_dataset, validation_dataset, test_dataset, char_to_num, num_to_char, max_len




# --- Model Components ---

class CTCLayer(layers.Layer):
    """
    Custom CTC Layer implementation.
    In Keras 3, this layer mainly exists for adding the loss during training.
    The actual calculation is typically done via the model.compile(loss=...) mechanism.
    However, defining it helps if the Keras example relied on adding loss within the layer.
    For pure Keras 3 style, this layer might not be strictly necessary if loss is handled externally.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost    # Reference the backend function

    def call(self, y_true, y_pred):
        # The loss calculation is the main purpose here when added to the model graph.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        # Calculate input_length and label_length tensors assuming padding if necessary
        # The Keras example assumes y_pred is padded and calculates based on its shape.
        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        # Label length needs to be calculated based on the actual sequence length before padding.
        # This might require passing label lengths explicitly or finding the first padding token.
        # Simplification: Assume label_length can be derived or is handled correctly by ctc_batch_cost if y_true is dense.
        # If y_true contains padding, a more robust way to get label_length is needed.
        # Example: find first occurrence of padding token (e.g., 0 if mask_token=None wasn't used carefully)
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64") # This assumes no padding or handled by backend

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss) # IMPORTANT: Add the loss to the layer

        # Return the prediction itself
        return y_pred

    # get_config needed for saving/loading model
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(img_width, img_height, vocab_size, max_len):
    """Builds the handwriting recognition model based on Keras example."""
    logger.info(f"Building model with Img W={img_width}, H={img_height}, Vocab Size={vocab_size}, Max Len={max_len}")

    input_img = keras.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")


    # CNN Layers (Convolutional Block)
    x = layers.Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1"
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="Pool1")(x)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), name="Pool2")(x)

    # Reshape for RNN layers
    # Output shape of Pool2 will be (batch_size, W/4, H/4, 64)
    # We need (batch_size, Timesteps, Features) for RNN
    # Timesteps = W/4, Features = (H/4) * 64
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNN Layers (Recurrent Block)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer - Dense layer with units = vocab_size + 1 (for CTC blank)
    x = layers.Dense(vocab_size + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating loss and wrap labels
    # Note: Keras 3 best practice is often to define loss function externally
    # and compile the model with it, rather than using a layer that adds loss.
    # However, we follow the Keras example's likely structure for demonstration.
    output = CTCLayer(name="ctc_loss")(labels, x) # Wrap predictions 'x' and true 'labels'

    # Define the model that includes the loss calculation
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_train"
    )

    # Compile the model - Use a dummy loss since CTC loss is added within the layer
    # Optimizer choice can be parameterized
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred) # Dummy loss fn

    # === Create Prediction Model ===
    # This model is used for inference and in the EditDistanceCallback.
    # It takes only the image as input and outputs the softmax predictions.
    prediction_model = keras.models.Model(
        inputs=model.input[0],                           # use the image input only
        outputs=model.get_layer(name="dense2").output, 
        name="handwriting_recognizer_predict"
    )

    logger.info("Model built and compiled successfully.")
    return model, prediction_model


def decode_batch_predictions(pred, num_to_char, max_len):
    """Decodes the predictions from the model."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search (default best path) - Keras backend might have ctc_decode
    # Check TF/Keras version for availability and API
    # Using tf.nn.ctc_greedy_decoder as an example if keras.backend.ctc_decode isn't suitable
    try:
        # tf.compat.v1.nn.ctc_greedy_decoder is one option for older TF versions
        # tf.nn.ctc_greedy_decoder for newer TF
        results = tf.nn.ctc_greedy_decoder(
            ops.transpose(pred, perm=[1, 0, 2]), # Needs time-major order
            tf.cast(input_len, tf.int32)
        )[0][0]
    except AttributeError:
         logger.warning("tf.nn.ctc_greedy_decoder not found, trying keras.backend.ctc_decode (might require TF backend)")
         # Fallback attempt (check compatibility)
         try:
             results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
         except Exception as decode_err:
             logger.error(f"CTC decoding failed with both tf.nn and keras.backend: {decode_err}")
             return ["DECODE_ERROR"] * pred.shape[0] # Return error string

    # Iterate over the results and convert back to symbols
    output_text = []
    if isinstance(results, tf.SparseTensor): # Decoder might return SparseTensor
        results_dense = tf.sparse.to_dense(results).numpy()
    else: # Or it might return dense tensor directly
        results_dense = results.numpy()

    for res in results_dense:
        # Filter out padding values (often -1 or 0 depending on implementation)
        res_filtered = res[res != -1]
        res_text = tf.strings.reduce_join(num_to_char(tf.cast(res_filtered, dtype=tf.int64))).numpy().decode("utf-8")
        output_text.append(res_text)
    return output_text


class EditDistanceCallback(keras.callbacks.Callback):
    """Calculates and logs edit distance on validation set each epoch."""
    def __init__(self, pred_model, validation_dataset, num_to_char, max_len):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_dataset = validation_dataset # The entire tf.data.Dataset
        self.num_to_char = num_to_char
        self.max_len = max_len
        self.edit_distances = [] # Store distances for plotting
        logger.info("EditDistanceCallback initialized.")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Calculating Edit Distance for Epoch {epoch+1}...")
        validation_images = []
        validation_labels_encoded = []

        # Iterate through the validation dataset to get all images and labels
        try:
            for batch in self.validation_dataset:
                validation_images.append(batch["image"])
                validation_labels_encoded.append(batch["label"])

            if not validation_images:
                logger.warning("No data found in validation dataset for edit distance calculation.")
                return

            # Concatenate all batches
            # Handle potential ragged tensors if batch sizes differ (last batch)
            # Using tf.concat which handles lists of tensors
            validation_images = tf.concat(validation_images, axis=0)
            validation_labels_encoded = tf.concat(validation_labels_encoded, axis=0)

        except Exception as e:
            logger.error(f"Error extracting data from validation_dataset in EditDistanceCallback: {e}", exc_info=True)
            return # Cannot proceed if data extraction fails

        # Get predictions from the prediction_model (only takes images)
        try:
            predictions = self.prediction_model.predict(validation_images, verbose=0)
        except Exception as e:
            logger.error(f"Error getting predictions in EditDistanceCallback: {e}", exc_info=True)
            return

        # Decode predictions
        pred_texts = decode_batch_predictions(predictions, self.num_to_char, self.max_len)

        # Decode true labels
        # Need to handle the encoded labels (potentially padded)
        original_texts = []
        for label_enc in validation_labels_encoded.numpy():
             # Filter out padding (assuming padding value is -1 or 0, check StringLookup/Dataset)
             # StringLookup default mask token is '', check if used. If not, padding might be 0.
             # Let's assume padding index is 0 if mask_token=None in StringLookup
             label_enc_filtered = label_enc[label_enc != 0]
             label_text = tf.strings.reduce_join(self.num_to_char(tf.cast(label_enc_filtered, dtype=tf.int64))).numpy().decode("utf-8")
             original_texts.append(label_text)


        # Calculate edit distance for each sample
        epoch_edit_distances = []
        for i in range(len(pred_texts)):
            # Simple edit distance calculation (Levenshtein distance)
            # Using a basic implementation here, consider libraries like 'editdistance' or 'Levenshtein' for efficiency
            s1 = original_texts[i]
            s2 = pred_texts[i]

            if len(s1) < len(s2): s1, s2 = s2, s1 # Ensure s1 is longer or equal
            if len(s2) == 0: dist = len(s1)
            else:
                previous_row = range(len(s2) + 1)
                for r, c1 in enumerate(s1):
                    current_row = [r + 1]
                    for c, c2 in enumerate(s2):
                        insertions = previous_row[c + 1] + 1
                        deletions = current_row[c] + 1
                        substitutions = previous_row[c] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                dist = previous_row[-1]

            epoch_edit_distances.append(dist)

        avg_edit_distance = np.mean(epoch_edit_distances) if epoch_edit_distances else 0
        self.edit_distances.append(avg_edit_distance) # Store for plotting

        logger.info(f"Epoch {epoch+1}: Average Edit Distance = {avg_edit_distance:.4f}")
        if logs is not None:
            logs['avg_edit_distance'] = avg_edit_distance # Add to logs dict for Keras history

        # Log metric to MLflow
        mlflow.log_metric("avg_edit_distance", avg_edit_distance, step=epoch)


# --- Main Training Function ---

def train(config):
    """Runs the full training and MLflow logging process."""
    # Use run_name from config if provided, otherwise let MLflow generate one
    run_name = config.get('run_name')
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow Run: {run_id} (Name: {run_name or 'Default'})")
        logger.info(f"Config: {config}")
        mlflow.log_params(config) # Log all config parameters

        # --- Setup ---
        set_random_seed(config['random_seed'])
        img_height = config.get('img_height', IMG_HEIGHT) # Use defaults if not in config
        img_width = config.get('img_width', IMG_WIDTH)
        batch_size = config['batch_size']

        try:
            # 1. Data Loading and Preprocessing
            logger.info("--- Data Preparation ---")
            base_path = download_and_extract_data()
            train_ds, val_ds, test_ds, char_to_num, num_to_char, max_len = \
                load_and_preprocess_data(
                    base_path,
                    batch_size=batch_size,
                    img_height=img_height,
                    img_width=img_width,
                    train_ratio=config['train_split_ratio'],
                    val_ratio=config['val_split_ratio'],
                    test_ratio=config['test_split_ratio'],
                    random_seed=config['random_seed']
                )

            if train_ds is None or val_ds is None:
                 raise ValueError("Data loading/preprocessing failed.")

            vocab_size = char_to_num.vocabulary_size()
            if vocab_size <= 1: # Should have actual chars + potentially blank
                 raise ValueError("Vocabulary size seems too small. Check data loading.")

            # 2. Model Building
            logger.info("--- Model Building ---")
            model, prediction_model = build_model(img_width, img_height, vocab_size, max_len)
            model.summary(print_fn=logger.info) # Log model summary

            # 3. Callbacks
            logger.info("--- Setting up Callbacks ---")
            edit_distance_callback = EditDistanceCallback(prediction_model, val_ds, num_to_char, max_len)
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config['early_stopping_patience'], restore_best_weights=True, verbose=1
            )
            # Optional: MLflow Autologging (can replace manual metric/model logging)
            # mlflow.keras.autolog()

            # 4. Training
            logger.info(f"--- Starting Model Training (Epochs: {config['epochs']}) ---")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config['epochs'],
                callbacks=[edit_distance_callback, early_stopping],
                verbose=1
            )
            logger.info("--- Training Finished ---")

            # 5. Log Metrics Manually (if not using autolog)
            logger.info("--- Logging Metrics & Plots ---")
            train_loss = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [])
            # Edit distance logged by callback, but retrieve final values for plotting
            avg_edit_dist_history = edit_distance_callback.edit_distances

            for epoch, loss in enumerate(train_loss):
                mlflow.log_metric("train_loss", loss, step=epoch)
            for epoch, loss in enumerate(val_loss):
                mlflow.log_metric("val_loss", loss, step=epoch)
            # Edit distance already logged per epoch by callback

            # Generate and Log Plots
            try:
                epochs_range = range(len(train_loss))

                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(epochs_range, train_loss, label='Training Loss')
                ax_loss.plot(epochs_range, val_loss, label='Validation Loss')
                ax_loss.set_title('Epochs vs Training & Validation Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()
                mlflow.log_figure(fig_loss, "plots/loss_vs_epochs.png")
                plt.close(fig_loss) # Close plot

                if avg_edit_dist_history:
                    fig_edit, ax_edit = plt.subplots()
                    # Ensure history length matches epoch count if early stopping occurred
                    edit_epochs_range = range(len(avg_edit_dist_history))
                    ax_edit.plot(edit_epochs_range, avg_edit_dist_history, label='Avg Edit Distance (Validation)')
                    ax_edit.set_title('Epochs vs Average Edit Distance')
                    ax_edit.set_xlabel('Epoch')
                    ax_edit.set_ylabel('Edit Distance')
                    ax_edit.legend()
                    mlflow.log_figure(fig_edit, "plots/edit_distance_vs_epochs.png")
                    plt.close(fig_edit) # Close plot
                else:
                    logger.warning("Average edit distance data not available for plotting.")

            except Exception as plot_err:
                 logger.error(f"Error generating or logging plots: {plot_err}", exc_info=True)


            # 6. Log Model Artifacts (StringLookups and Keras Model)
            logger.info("--- Logging Model Artifacts ---")
            # Log StringLookup layers needed for inference
            lookup_path = "string_lookups"
            os.makedirs(lookup_path, exist_ok=True)
            char_to_num_path = os.path.join(lookup_path, "char_to_num")
            num_to_char_path = os.path.join(lookup_path, "num_to_char")
            try:
                # Saving StringLookup requires saving the entire layer config/weights
                # A simple way is to save a dummy model containing them
                lookup_model = keras.Sequential([char_to_num, num_to_char], name="lookup_model")
                lookup_model.save(lookup_path, save_format="tf") # Save as TF SavedModel format
                mlflow.log_artifact(lookup_path, artifact_path="string-lookups")
                logger.info(f"StringLookup layers saved and logged from: {lookup_path}")

            except Exception as lookup_err:
                 logger.error(f"Error saving/logging StringLookup layers: {lookup_err}", exc_info=True)


            # Log the *prediction* model (without the CTC loss layer wrapper)
            # This is usually the one needed for deployment/inference.
            registered_model_name = config.get("registered_model_name")
            mlflow.keras.log_model(
                prediction_model, # Log the model suitable for prediction
                artifact_path="prediction-model", # Subfolder in MLflow artifacts
                registered_model_name=registered_model_name, # Register if name provided
                # Signatures can be useful for defining expected input/output
                # signature=infer_signature(validation_images[:5], prediction_model.predict(validation_images[:5]))
            )
            logger.info(f"Prediction model logged to MLflow. Registered as: '{registered_model_name}'")

            # 7. Evaluate on Test Set (Optional but recommended)
            if test_ds:
                logger.info("--- Evaluating on Test Set ---")
                try:
                    # Use the prediction model and EditDistance logic for evaluation
                    test_edit_distance_callback = EditDistanceCallback(prediction_model, test_ds, num_to_char, max_len)
                    # Manually trigger calculation on test set (doesn't need full epoch loop)
                    test_images = []
                    test_labels_encoded = []
                    for batch in test_ds:
                         test_images.append(batch["image"])
                         test_labels_encoded.append(batch["label"])
                    if test_images:
                         test_images = tf.concat(test_images, axis=0)
                         test_labels_encoded = tf.concat(test_labels_encoded, axis=0)
                         test_predictions = prediction_model.predict(test_images, verbose=0)
                         pred_texts = decode_batch_predictions(test_predictions, num_to_char, max_len)

                         original_texts = []
                         for label_enc in test_labels_encoded.numpy():
                              label_enc_filtered = label_enc[label_enc != 0] # Assuming 0 padding
                              label_text = tf.strings.reduce_join(num_to_char(tf.cast(label_enc_filtered, dtype=tf.int64))).numpy().decode("utf-8")
                              original_texts.append(label_text)

                         test_edit_distances = []
                         for i in range(len(pred_texts)):
                              s1 = original_texts[i]
                              s2 = pred_texts[i]
                              if len(s1) < len(s2): s1, s2 = s2, s1
                              if len(s2) == 0: dist = len(s1)
                              else:
                                   previous_row = range(len(s2) + 1)
                                   for r, c1 in enumerate(s1):
                                        current_row = [r + 1]
                                        for c, c2 in enumerate(s2):
                                             insertions = previous_row[c + 1] + 1; deletions = current_row[c] + 1
                                             substitutions = previous_row[c] + (c1 != c2)
                                             current_row.append(min(insertions, deletions, substitutions))
                                        previous_row = current_row
                                   dist = previous_row[-1]
                              test_edit_distances.append(dist)

                         avg_test_edit_distance = np.mean(test_edit_distances) if test_edit_distances else 0
                         logger.info(f"Test Set Average Edit Distance: {avg_test_edit_distance:.4f}")
                         mlflow.log_metric("test_avg_edit_distance", avg_test_edit_distance)
                    else:
                         logger.warning("Test dataset was empty, skipping evaluation.")

                except Exception as eval_err:
                    logger.error(f"Error during test set evaluation: {eval_err}", exc_info=True)

            logger.info(f"--- MLflow Run {run_id} Completed Successfully ---")

        except Exception as e:
            logger.error(f"An error occurred during the training process: {e}", exc_info=True)
            mlflow.end_run(status="FAILED") # Mark run as failed
            logger.info(f"--- MLflow Run {run_id} Marked as FAILED ---")
            # Re-raise the exception to stop the script
            raise

        # end_run() is called automatically when exiting the 'with' block


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Handwriting Recognition Model with MLflow")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs") # Keras example uses 50+, 10 for quick test
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--img_height", type=int, default=IMG_HEIGHT, help="Target image height")
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH, help="Target image width")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, help="Fraction of data for training (default 0.8)")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Fraction of data for validation (default 0.1)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data shuffling and splitting")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (default 10)")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for the MLflow run")
    parser.add_argument("--registered_model_name", type=str, default="HandwritingRecognitionModel", help="Name to register the model under in MLflow Model Registry (set to '' or None to skip registration)")

    args = parser.parse_args()

    # Basic validation
    if args.train_split_ratio <= 0 or args.val_split_ratio <= 0 or args.train_split_ratio + args.val_split_ratio >= 1.0:
        parser.error("Train and validation split ratios must be positive and sum to less than 1.0")
    test_split_ratio = round(1.0 - args.train_split_ratio - args.val_split_ratio, 4) # Calculate test split

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "train_split_ratio": args.train_split_ratio,
        "val_split_ratio": args.val_split_ratio,
        "test_split_ratio": test_split_ratio,
        "random_seed": args.random_seed,
        "early_stopping_patience": args.early_stopping_patience,
        "run_name": args.run_name,
        "registered_model_name": args.registered_model_name if args.registered_model_name else None, # Pass None if empty string
    }

    # Run the training process
    train(config)