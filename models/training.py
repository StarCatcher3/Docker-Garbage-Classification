import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_position
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("GARBAGE_MODEL_DIR", os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "models")))
DB_URL = "http://db-service:8000"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

spark = SparkSession.builder \
    .appName("GarbageClassificationML") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

def load_and_prepare_data():

    # 1. Fetch from API
    response = requests.get(f"{DB_URL}/data")
    response.raise_for_status()

    payload = response.json()

    print("Response received")

    # 2. Parse JSON → Pandas
    train_df = pd.read_json(
        StringIO(payload["train"]),
        orient="records"
    )

    test_df = pd.read_json(
        StringIO(payload["test"]),
        orient="records"
    )

    #print(train_df.head())
    #print(test_df.head())

    # 3. Convert label → integer class index
    classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

    def encode_label(df):
        y = np.array(
            [classes.index(c) for c in df["class"]],
            dtype=np.int32
        )
        return y

    # 4. Extract X (your image arrays)
    def extract_x(df, split):
        X = np.array(df[f"x_{split}"].tolist(), dtype=np.float32)
        return X.reshape(X.shape + (1,))  # (N, H, W, 1)

    # 5. Build train/test
    X_train = extract_x(train_df, "train")
    y_train = encode_label(train_df)

    X_test = extract_x(test_df, "test")
    y_test = encode_label(test_df)

    return X_train, y_train, X_test, y_test

CONFIG = {
    "img_size": 64,
    "batch_size": 64,
    "epochs": 25,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "seed": 42
}

def data_augmentation():
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.05)
    ], name="data_augmentation")
    return data_aug


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_model(model, name):
    filename = f"{name}.keras"
    save_path = os.path.join(MODEL_DIR, filename)
    model.save(save_path)
    print(f"Modèle sauvegardé : {save_path}")

def create_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation(),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="CNN")
    
    return compile_model(model)


def main():
    tf.keras.utils.set_random_seed(CONFIG["seed"])

    X_train, y_train, X_val, y_val = load_and_prepare_data()

    input_shape = X_train.shape[1:]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Distribution des images sur les labels: {np.bincount(y_val)}")

    num_classes = int(max(y_train.max(), y_val.max()) + 1)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weight = {
        i: float(len(y_train) / (num_classes * count))
        for i, count in enumerate(class_counts) if count > 0
    }

    model = create_cnn(input_shape, num_classes)

    model.summary()

    #Entraînement
    log_name = f"{model.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    current_log_path = os.path.join("..", "models", "logs", log_name)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max"
            ),
            tf.keras.callbacks.TensorBoard(log_dir=current_log_path)
        ],
        verbose=2
    )
    val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    save_model(model, "final_CNN")
    spark.catalog.clearCache()


main()
