import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow.tensorflow
import os, random, logging, datetime, mlflow

# Ensure logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Create a log file name with a timestamp
log_filename = os.path.join(
    logs_dir, f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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

# MLflow experiment setup
mlflow.set_experiment("handwriting_recognition")

# Constants
batch_size = 16
img_width = 128
img_height = 32
max_length = 21
downsample_factor = 4
char_to_num = layers.StringLookup(vocabulary=list("abcdefghijklmnopqrstuvwxyz0123456789"), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def split_data(images, labels, train_size=0.8, val_size=0.1, test_size=0.1):
    """Split the data into training, validation, and test sets."""
    assert train_size + val_size + test_size == 1.0, "Split proportions must sum to 1"
    
    dataset_size = len(images)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_end = int(train_size * dataset_size)
    val_end = int((train_size + val_size) * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_images = [images[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def distortion_free_resize(image, img_size):
    """Resize without distortion."""
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    
    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    
    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    
    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2
    
    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(img_width, img_height)):
    """Preprocess the images with a check for empty input."""
    # Read file from disk (returns a string tensor)
    image_data = tf.io.read_file(image_path)
    
    # Check the length of the file's content
    file_length = tf.strings.length(image_data)
    
    # Use tf.cond to check for an empty file.
    # Both branches must return a tensor of type uint8.
    image = tf.cond(
        tf.equal(file_length, 0),
        # True branch: if empty, return a dummy image tensor of zeros.
        lambda: tf.zeros((img_size[1], img_size[0], 1), dtype=tf.uint8),
        # False branch: decode the PNG image.
        lambda: tf.image.decode_png(image_data, channels=1)
    )
    
    # Resize the image without distortion.
    image = distortion_free_resize(image, img_size)
    
    # Cast to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image

def vectorize_label(label):
    """Convert label to a vector with empty label handling."""
    # Convert to lowercase and split into characters
    label = tf.strings.lower(label)
    label_chars = tf.strings.unicode_split(label, 'UTF-8')
    
    # Convert to numerical tokens
    label_encoded = char_to_num(label_chars)
    
    # Handle empty labels: Check if label_encoded contains any elements.
    non_empty = tf.greater(tf.size(label_encoded), 0)
    
    # Use tf.cond to decide whether to pad or fill with zeros.
    label_encoded = tf.cond(
        non_empty,
        lambda: tf.pad(label_encoded, [[0, max_length - tf.shape(label_encoded)[0]]], constant_values=0),
        lambda: tf.fill([max_length], tf.constant(0, dtype=tf.int64))  # Ensure int64 output
    )
    
    return label_encoded[:max_length]  # Ensure fixed length

def process_images_labels(image_path, label):
    """Process images and labels."""
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(images, labels):
    """Prepare the dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Batch size
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        # Time steps for predictions
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        # Compute actual label length: count non-zero tokens along sequence axis.
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype="int64"), axis=1, keepdims=True)
        
        # Build constant tensors for input_length: one for each sample.
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_model():
    """Build the model architecture."""
    # Inputs
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image")
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # RNN layers
    x = layers.Reshape(target_shape=((img_width // 4), (img_height // 4 * 64)), name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="lstm1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name="lstm2")(x)
    
    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)
    
    # Define the model
    model = keras.Model(inputs=input_img, outputs=x, name="handwriting_recognition")
    
    # CTC Layer for training
    labels = layers.Input(name="label", shape=(None,), dtype="int64")
    output = CTCLayer(name="ctc_loss")(labels, x)
    
    # Define the model with CTC
    train_model = keras.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognition_training"
    )
    
    # Optimizer
    opt = keras.optimizers.Adam()
    
    # Compile the model
    train_model.compile(optimizer=opt)
    
    return train_model


def decode_batch_predictions(pred):
    """Decode the predictions."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you might want beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.squeeze(res)
        res = num_to_char(res)
        res = tf.strings.reduce_join(res).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


class EditDistanceCallback(keras.callbacks.Callback):
    """Custom callback to calculate edit distance using a separate prediction model."""
    
    def __init__(self, prediction_model, validation_dataset):
        super().__init__()
        self.prediction_model = prediction_model
        self.validation_dataset = validation_dataset
        self.edit_distances = []
        
    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        for batch in self.validation_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            # Use the prediction model (which takes a single image input)
            preds = self.prediction_model.predict(batch_images)
            
            pred_texts = decode_batch_predictions(preds)
            for i in range(len(pred_texts)):
                # Extract the ground-truth label for the i-th sample from batch_labels.
                # Remove padding (assuming padding value is 0) and decode using num_to_char.
                label_tensor = batch_labels[i]
                # Filter out padded values.
                label_nonzero = tf.boolean_mask(label_tensor, tf.not_equal(label_tensor, 0))
                # Convert numeric tokens to characters.
                label_chars = num_to_char(label_nonzero)
                # Join the characters to form a string.
                label_text = tf.strings.reduce_join(label_chars).numpy().decode("utf-8")
                
                # Convert predicted text to a SparseTensor.
                dense_pred = tf.convert_to_tensor([list(pred_texts[i])], dtype=tf.string)
                sparse_pred = tf.sparse.from_dense(dense_pred)
                
                # Convert the ground truth label text to a SparseTensor.
                dense_label = tf.convert_to_tensor([list(label_text)], dtype=tf.string)
                sparse_label = tf.sparse.from_dense(dense_label)
                
                # Compute edit distance using SparseTensors.
                edit_distance = tf.edit_distance(sparse_pred, sparse_label).numpy()[0]
                edit_distances.append(edit_distance)

        
        mean_edit_distance = np.mean(edit_distances)
        self.edit_distances.append(mean_edit_distance)
        if logs is not None:
            logs["val_edit_distance"] = mean_edit_distance
        mlflow.log_metric("val_edit_distance", mean_edit_distance, step=epoch)
        logger.info(f"Mean edit distance: {mean_edit_distance}")


def find_words_file(base_dir):
    """Find the words.txt file in the dataset directory structure."""
    # Try different possible locations
    possible_paths = [
        os.path.join(base_dir, "IAM_Words", "words", "words.txt"),
        os.path.join(base_dir, "IAM_Words_extracted", "IAM_Words", "words", "words.txt"),
        os.path.join(base_dir, "IAM_Words_extracted", "IAM_Words", "words.txt")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found words.txt at: {path}")
            return path
    
    # If we can't find it, search for it
    logger.info("Searching for words.txt file...")
    for root, dirs, files in os.walk(base_dir):
        if "words.txt" in files:
            path = os.path.join(root, "words.txt")
            logger.info(f"Found words.txt at: {path}")
            return path
    
    raise FileNotFoundError("Could not find words.txt file in the dataset directory")


def find_words_directory(base_dir, words_file_path):
    """Find the directory containing word images."""
    # The words directory should be in the same directory as words.txt or in a subdirectory
    words_dir = os.path.dirname(words_file_path)
    
    # Check if this directory contains PNG files, if yes, then use it
    if any(file.endswith('.png') for file in os.listdir(words_dir)):
        return words_dir
    
    # Otherwise, check for a subdirectory that contains PNG files
    for root, dirs, files in os.walk(os.path.dirname(words_file_path)):
        if any(file.endswith('.png') for file in files):
            return root
    
    raise FileNotFoundError("Could not find directory with word images")


def main(train_size=0.8, val_size=0.1, test_size=0.1):
    """Main function to train the model."""
    try:
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Log parameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("img_width", img_width)
            mlflow.log_param("img_height", img_height)
            mlflow.log_param("max_length", max_length)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("test_size", test_size)
            
            # Download and prepare the dataset
            logger.info("Downloading and preparing IAM dataset...")
            words_ds = tf.keras.utils.get_file(
                "IAM_Words.zip",
                origin="https://git.io/J0fjL",
                extract=True,
                cache_dir=".",
                cache_subdir="data",
            )

            # Find the directory containing the dataset
            base_dir = "data"
            extracted_dir = os.path.join(base_dir, "IAM_Words_extracted", "IAM_Words")
            
            # Find the words.txt file
            words_txt_path = os.path.join(extracted_dir, "words.txt")
            
            if not os.path.exists(words_txt_path):
                raise FileNotFoundError(f"Could not find words.txt at {words_txt_path}")
            
            logger.info(f"Found words.txt at: {words_txt_path}")
            
            # Find the words directory containing images
            words_dir = os.path.join(extracted_dir, "words")
            
            if not os.path.exists(words_dir):
                raise FileNotFoundError(f"Words directory not found at {words_dir}")
            
            logger.info(f"Found words directory at: {words_dir}")
            
            # Read the words.txt file and process the dataset
            images = []
            labels = []
            
            words_list = open(words_txt_path, "r").readlines()
            for line in words_list:
                if line.startswith("#"):
                    continue
                    
                line_split = line.strip().split(" ")
                if len(line_split) >= 9:
                    image_name = line_split[0]
                    image_text = line_split[8].lower()
                    
                    # Filter out samples with characters not in vocabulary
                    if (len(image_text) > 0 and 
                        all(char in "abcdefghijklmnopqrstuvwxyz0123456789" for char in image_text)):
                        # For the IAM dataset, the image path follows this pattern:
                        # a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
                        parts = image_name.split("-")
                        if len(parts) >= 2:
                            writer_id = parts[0]
                            form_id = f"{writer_id}-{parts[1]}"
                            image_path = os.path.join(words_dir, writer_id, form_id, f"{image_name}.png")
                            
                            if os.path.exists(image_path):
                                images.append(image_path)
                                labels.append(image_text)
                            else:
                                logger.debug(f"Image not found: {image_path}")
            
            logger.info(f"Found {len(images)} valid images with labels")
            
            if len(images) == 0:
                raise ValueError("No valid images found. Check the dataset structure.")
            
            # Split the data
            (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = \
                split_data(images, labels, train_size, val_size, test_size)
            
            logger.info(f"Number of training samples: {len(train_images)}")
            logger.info(f"Number of validation samples: {len(val_images)}")
            logger.info(f"Number of test samples: {len(test_images)}")
            
            mlflow.log_param("train_samples", len(train_images))
            mlflow.log_param("val_samples", len(val_images))
            mlflow.log_param("test_samples", len(test_images))
            
            # Prepare datasets
            train_dataset = prepare_dataset(train_images, train_labels)
            validation_dataset = prepare_dataset(val_images, val_labels)
            test_dataset = prepare_dataset(test_images, test_labels)
            
            # Build and compile the model
            logger.info("Building model...")
            model = build_model()
            model.summary()
            
            prediction_model = keras.models.Model(
                inputs=model.get_layer(name="image").input, 
                outputs=model.get_layer(name="dense2").output,
                name="handwriting_recognition_prediction"
            )
                
            # Custom callback for edit distance using the prediction model
            edit_distance_callback = EditDistanceCallback(prediction_model, validation_dataset)

            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            
            # Train the model
            logger.info("Training model...")
            epochs = 2
            mlflow.log_param("epochs", epochs)
            
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[early_stopping, edit_distance_callback]
            )
            
            # Evaluate the model
            logger.info("Evaluating model...")
            prediction_model = keras.models.Model(
                model.get_layer(name="image").input, model.get_layer(name="dense2").output
            )
            
            # Log performance metrics
            for epoch, (loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
                mlflow.log_metric("loss", loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # Plot training graphs
            logger.info("Generating plots...")
            
            # Loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            loss_plot_path = "plots\loss_plot.png"
            plt.savefig(loss_plot_path)
            mlflow.log_artifact(loss_plot_path)
            
            # Edit distance plot
            plt.figure(figsize=(10, 5))
            plt.plot(edit_distance_callback.edit_distances, label="Average Edit Distance")
            plt.xlabel("Epochs")
            plt.ylabel("Edit Distance")
            plt.title("Average Edit Distance per Epoch")
            plt.legend()
            edit_distance_plot_path = "plots\edit_distance_plot.png"
            plt.savefig(edit_distance_plot_path)
            mlflow.log_artifact(edit_distance_plot_path)
            
            # Create directory for models if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save and log the model
            logger.info("Saving model...")
            model_path = "models/handwriting_model"
            prediction_model.save(model_path)
            mlflow.tensorflow.log_model(prediction_model, "handwriting_model")
            
            # Register the model in MLflow Model Registry
            model_name = "handwriting_recognition_model"
            model_version = mlflow.register_model(f"runs:/{run_id}/handwriting_model", model_name)
            
            logger.info(f"Model registered as: {model_name} version {model_version.version}")
            
            # Save a few test examples for inference demonstration
            test_examples_dir = "test_examples"
            os.makedirs(test_examples_dir, exist_ok=True)
            
            for i, img_path in enumerate(test_images[:5]):
                import shutil
                dest_path = os.path.join(test_examples_dir, f"test_example_{i}.png")
                shutil.copy(img_path, dest_path)
                mlflow.log_artifact(dest_path)
                
            logger.info("Run completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the main function with different data splits
    # First run
    logger.info("Starting Run 1 with default splits")
    main(train_size=0.8, val_size=0.1, test_size=0.1)
    
    # Second run
    logger.info("Starting Run 2 with adjusted splits")
    main(train_size=0.7, val_size=0.15, test_size=0.15)
    
    # Third run
    logger.info("Starting Run 3 with adjusted splits")
    main(train_size=0.75, val_size=0.15, test_size=0.1)