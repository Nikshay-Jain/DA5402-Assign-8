import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow
import logging
import random
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    """Preprocess the images."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    """Convert label to a vector."""
    label = char_to_num(tf.strings.lower(label))
    length = tf.shape(label)[0]
    pad_amount = max_length - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=0)
    return label


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
    """CTC Layer to compute loss and decoder."""
    
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Return decoded sequence for metric calculation
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
    """Custom callback to calculate edit distance."""
    
    def __init__(self, validation_dataset):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.edit_distances = []
        
    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        for batch in self.validation_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            preds = self.model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)
            
            for i in range(len(pred_texts)):
                label = tf.gather(batch_labels[i], tf.where(tf.not_equal(batch_labels[i], 0)))
                label = tf.squeeze(label).numpy()
                label_text = ''.join([num_to_char(c).numpy().decode('utf-8') for c in label])
                
                edit_distance = tf.edit_distance(
                    tf.convert_to_tensor([list(pred_texts[i])], dtype=tf.string),
                    tf.convert_to_tensor([list(label_text)], dtype=tf.string)
                ).numpy()[0]
                
                edit_distances.append(edit_distance)
        
        mean_edit_distance = np.mean(edit_distances)
        self.edit_distances.append(mean_edit_distance)
        logs["val_edit_distance"] = mean_edit_distance
        mlflow.log_metric("val_edit_distance", mean_edit_distance, step=epoch)
        logger.info(f"Mean edit distance: {mean_edit_distance}")
        return


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
                cache_subdir="datasets",
            )

            data_dir = os.path.join(os.path.dirname(words_ds), "IAM_Words", "words")
            images = []
            labels = []
            
            words_list = open(f"{data_dir}/words.txt", "r").readlines()
            for line in words_list:
                if line.startswith("#"):
                    continue
                    
                line_split = line.strip().split(" ")
                if len(line_split) >= 9:
                    image_name = line_split[0]
                    image_text = line_split[8].lower()
                    
                    # Filter out samples with characters not in vocabulary
                    if all(char in "abcdefghijklmnopqrstuvwxyz0123456789" for char in image_text):
                        image_path = os.path.join(data_dir, image_name + ".png")
                        if os.path.exists(image_path):
                            images.append(image_path)
                            labels.append(image_text)
            
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
            
            # Custom callback for edit distance
            edit_distance_callback = EditDistanceCallback(validation_dataset)
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            
            # Train the model
            logger.info("Training model...")
            epochs = 50
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
            loss_plot_path = "loss_plot.png"
            plt.savefig(loss_plot_path)
            mlflow.log_artifact(loss_plot_path)
            
            # Edit distance plot
            plt.figure(figsize=(10, 5))
            plt.plot(edit_distance_callback.edit_distances, label="Average Edit Distance")
            plt.xlabel("Epochs")
            plt.ylabel("Edit Distance")
            plt.title("Average Edit Distance per Epoch")
            plt.legend()
            edit_distance_plot_path = "edit_distance_plot.png"
            plt.savefig(edit_distance_plot_path)
            mlflow.log_artifact(edit_distance_plot_path)
            
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
