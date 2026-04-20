from PIL import Image
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat, lit
from pyspark.sql.types import ArrayType, StructField, StructType, IntegerType, StringType
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = Path(os.getenv("GARBAGE_RAW_DATA_DIR", os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "data")))
DB_DIR = Path(os.getenv("GARBAGE_DB_DIR", os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "database")))
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("GarbageClassification") \
    .master(SPARK_MASTER) \
    .config("spark.hadoop.fs.permissions.umask-mode", "000") \
    .config("spark.speculation", "false") \
    .getOrCreate()
schema = StructType([
    StructField("class", StringType(), True),
    StructField("file", StringType(), True)
])
file_size = (64, 64)

def transform_images_to_parquet(raw_data_path, parquet_path, train_test):
    directory = Path(raw_data_path) / train_test
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    directory_str = str(directory)
    folders = sorted(folder.name for folder in directory.iterdir() if folder.is_dir())
    file_data = (
        (folder, file.name)
        for folder in folders
        for file in (directory / folder).iterdir()
        if file.is_file()
    )

    # Create DataFrame with file paths
    df = spark.createDataFrame(file_data, schema=schema)


    # Define UDF to process images
    def process_image(filename):
        img = Image.open(os.path.join(directory_str, filename))
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(file_size)  # Resize for vector
        arr = np.array(img).astype(int).tolist()
        return arr
        
    process_image_udf = udf(process_image, ArrayType(ArrayType(IntegerType())))

    # Define UDF to assign one-hot encoding based on class name
    def get_one_hot(class_name):
        classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
        one_hot = [0] * len(classes)
        index = classes.index(class_name)
        one_hot[index] = 1
        return one_hot

    get_one_hot_udf = udf(get_one_hot, ArrayType(IntegerType()))

    # Apply transformation
    df = df.withColumn(f"x_{train_test}", process_image_udf(concat(col("class"), lit("/"), col("file"))))
    df = df.withColumn(f"y_{train_test}", get_one_hot_udf(col("class")))

    df = df.select(f"x_{train_test}", f"y_{train_test}", "class").filter(col(f"x_{train_test}").isNotNull())

    # Save as Parquet
    output_path = Path(parquet_path) / f"{train_test}_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output_path))

if __name__ == "__main__":
    for data_type in ["train", "test"]:
        transform_images_to_parquet(RAW_DATA_DIR, DB_DIR, data_type)