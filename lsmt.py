import pprint
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Sequential
from keras.src.layers import LSTM, Embedding
from tensorflow import keras

print(tf.__version__)
# Set device to cpu
# tf.config.set_visible_devices([], "GPU")
print("Devices: ", tf.config.get_visible_devices())
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

DEBUG = False
pretty_printer = pprint.PrettyPrinter(indent=2)


def pp(x):
    if DEBUG:
        pprint.pprint(x)


# get data files
train_file_path = "data/train-data.tsv"
validate_file_path = "data/valid-data.tsv"

# preprocess
EPOCHS = 10
BATCH_SIZE = 32


def load_data(path: str) -> pd.DataFrame:
    """Load data from a file and return a DataFrame."""
    try:
        pp(path.split("/")[-1])
        data = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["spam", "sms"],
        )  # .drop_duplicates(["sms"])
        return data
    except Exception as e:
        raise e


train_data = load_data(train_file_path)
test_data = load_data(validate_file_path)


# Model architecture and normalization of inputs


def create_text_vectorization_layer(
    train_x: pd.Series,
) -> keras.layers.TextVectorization:
    vectorizer = keras.layers.TextVectorization(
        output_mode="int",
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(np.array(train_x.values))
    return vectorizer


def create_model(train_x: pd.DataFrame) -> keras.Model:
    vectorizer = create_text_vectorization_layer(train_x["sms"])
    vocab_size = len(vectorizer.get_vocabulary())
    # Reduce embedding dimensions
    embedding_dim = min(vocab_size // 4, 100)

    model = Sequential(
        [
            vectorizer,
            # Reduced embedding dimension
            keras.layers.Embedding(vocab_size, embedding_dim),
            # Add BatchNormalization
            keras.layers.BatchNormalization(),
            # First LSTM layer
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            # Second LSTM layer
            keras.layers.LSTM(32),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            # Final dense layers
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Add learning rate scheduling
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


model = create_model(train_data)


def train_model(model: keras.Model, x: pd.DataFrame, y: pd.Series) -> keras.Model:

    assert "sms" in x.columns, "SMS column not found in input data"
    assert len(x) == len(y), "Features and labels must have same length"

    x_array = np.array(x["sms"].values)
    y_array = np.array([1 if label == "spam" else 0 for label in y.values])

    pp(x_array)
    pp(y_array)

    print(f"Input shape: {x_array.shape}")
    print(f"Labels shape: {y_array.shape}")

    model.fit(
        x_array,
        y_array,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=[tensorboard_callback],
    )
    return model


x = train_data.copy()
y = train_data.pop("spam")

model = train_model(model, x, y)
model.summary()


# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    prediction = model.predict([pred_text])
    return [prediction[0][0], "spam" if prediction[0][0] > 0.5 else "ham"]


pred_text = "how are you doing today?"

prediction = predict_message(pred_text)


# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too",
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")


test_predictions()
