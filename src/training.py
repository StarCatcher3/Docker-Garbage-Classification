# %% [markdown]
# # Classification de déchets avec PySpark MLlib
# 
# Ce notebook montre comment entraîner un modèle de Machine Learning pour identifier le type de déchet à partir de vos images préalablement traitées par `data_tansformation.py`.

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_position
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
from datetime import datetime

spark = SparkSession.builder \
    .appName("GarbageClassificationML") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark


# %% [markdown]
# ### Chargement et formatage des données
# Keras nécessite que les features soient sous forme de `Veteurs`. Nos pixels sont stockés au format texte dans un .parquet

# %%
def load_and_prepare_data(path, split):
    df = spark.read.parquet(path)
    label_col = f"y_{split}"
    feature_col = f"x_{split}"

    df = df.withColumn(
        "label",
        array_position(col(label_col), 1) - 1
    )

    return df.select("label", col(feature_col).alias("features")).dropna()

# Test
train_data = load_and_prepare_data("../data/train_data.parquet", "train")
test_data = load_and_prepare_data("../data/test_data.parquet", "test")
print("Train data :")
train_data.show(5)
print("Test data :")
test_data.show(5)


# %%
CONFIG = {
    "img_size": 64,
    "batch_size": 64,
    "epochs": 100,
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

def spark_to_numpy(df):

    pdf = df.toPandas()

    
    y = pdf["label"].to_numpy(dtype=np.int32)

    
    X = np.array(pdf["features"].tolist(), dtype=np.float32)

    X = X.reshape(X.shape + (1,))

    return X, y

def save_model(model, name):
    filename = f"{name}.keras"
    save_path = os.path.join("..", "models", filename)
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

    train_data = load_and_prepare_data("../data/train_data.parquet", "train")
    test_data = load_and_prepare_data("../data/test_data.parquet", "test")


    X_train, y_train = spark_to_numpy(train_data)
    X_val, y_val = spark_to_numpy(test_data)

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
    save_model(model, f"final_{model.name}_val_acc_{val_accuracy:.4f}")
    spark.catalog.clearCache()


main()


# %%


# %%



