# src/utils.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from keras import backend as K
from keras.utils import pad_sequences
from keras.datasets import mnist
from keras.preprocessing.sequence import pad_sequences as pad_seq


# Load IAM or placeholder dataset - for now using EMNIST dataset for handwriting

def get_data(seed=42):
    # For prototype purposes using EMNIST-balanced dataset
    import tensorflow_datasets as tfds
    ds_train, ds_info = tfds.load('emnist/letters', split='train', with_info=True, as_supervised=True)

    # Convert to numpy and preprocess
    def preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.transpose(img)  # simulate handwriting format
        return img, label - 1

    ds = ds_train.map(preprocess).cache().shuffle(1000, seed=seed).batch(64).prefetch(tf.data.AUTOTUNE)

    total = ds.cardinality().numpy()
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_data = ds.take(train_size)
    val_data = ds.skip(train_size).take(val_size)
    test_data = ds.skip(train_size + val_size)

    char_to_num = keras.layers.StringLookup(vocabulary=list("abcdefghijklmnopqrstuvwxyz"), oov_token="?")
    num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True)

    return (train_data, val_data, test_data), char_to_num, num_to_char


def build_model():
    input_img = layers.Input(shape=(28, 28, 1), name='image')
    x = layers.Rescaling(1.0 / 255)(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    new_shape = (-1, x.shape[2] * x.shape[3])
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(27, activation='softmax')(x)  # 26 letters + blank

    model = keras.models.Model(inputs=input_img, outputs=x, name="handwriting_crnn")
    return model


def plot_metrics(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig("metrics.png")


def calculate_edit_distance(model, test_data, num_to_char):
    total_dist = 0
    total_count = 0

    for batch in test_data:
        images, labels = batch
        preds = model.predict(images)
        pred_text = tf.argmax(preds, axis=-1)
        pred_text = tf.strings.reduce_join(num_to_char(pred_text), axis=1)
        label_text = tf.strings.reduce_join(num_to_char(labels), axis=1)

        edit_dist = tf.edit_distance(tf.strings.unicode_split(pred_text, 'UTF-8'),
                                     tf.strings.unicode_split(label_text, 'UTF-8'),
                                     normalize=True)
        total_dist += tf.reduce_sum(edit_dist)
        total_count += edit_dist.shape[0]

    return (total_dist / total_count).numpy()