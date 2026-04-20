from PIL import Image
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, array_max, array_position, array, element_at
from pyspark.sql.types import ArrayType, StructField, StructType, IntegerType, StringType, DoubleType
import numpy as np
from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = Path(os.getenv("GARBAGE_RAW_DATA_DIR", os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "data")))
PARQUET_DIR = Path(os.getenv("GARBAGE_PARQUET_DIR", os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "data")))
MODEL_PATH = Path(os.getenv("GARBAGE_MODEL_PATH", BASE_DIR / "models" / "final_CNN.keras"))
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Initialize Spark Session
spark = SparkSession.builder.appName("GarbageClassification").master(SPARK_MASTER).getOrCreate()

schema = StructType([
    StructField("file", StringType(), True)
])
file_size = (64, 64)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(str(MODEL_PATH))
classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

def predict_images_to_parquet(raw_data_path, parquet_path, sub_folder):
    directory = Path(raw_data_path) / sub_folder
    archive_dir = Path(raw_data_path) / "archive"

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    directory_str = str(directory)
    files = sorted(file.name for file in directory.iterdir() if file.is_file())

    if not files:
        print(f"No files to process in {directory}")
        return

    file_data = ((file,) for file in files)

    # Create DataFrame with file paths
    df = spark.createDataFrame(file_data, schema=schema)

    # Define UDF to process images
    def process_image(filename):
        img = Image.open(os.path.join(directory_str, filename))
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(file_size)  # Resize for vector
        arr = np.array(img).astype(int)
        return arr.tolist()
        
    process_image_udf = udf(process_image, ArrayType(ArrayType(IntegerType())))

    def predict(vector):
        X = np.array(vector)
        X = X.reshape((1,) + X.shape + (1,))
        prediction = model.predict(X)
        return prediction.flatten().tolist()
    
    predict_udf = udf(predict, ArrayType(DoubleType()))

    # Apply transformation
    df = df.withColumn("vector", process_image_udf(col("file")))
    df = df.withColumn("prediction", predict_udf(col("vector")))
    df = df.withColumn("confidence", array_max(col("prediction")))
    df = df.withColumn("class_id", array_position(col("prediction"), col("confidence")).astype("INT"))
    df = df.withColumn("class", element_at(array(*[lit(c) for c in classes]), col("class_id")))

    df = df.select("file", "prediction", "class_id", "class", "confidence")
    df.show()

    # Save as Parquet
    output_path = Path(parquet_path) / "prediction_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("append").parquet(str(output_path))

    for file in files:
        os.replace(directory / file, archive_dir / file)

if __name__ == "__main__":
    predict_images_to_parquet(RAW_DATA_DIR, PARQUET_DIR, "input")