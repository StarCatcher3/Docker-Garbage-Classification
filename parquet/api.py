from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import pandas as pd
from pathlib import Path
import numpy as np
from io import StringIO

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = Path(
    os.getenv(
        "GARBAGE_RAW_DATA_DIR",
        os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "data")
    )
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    return obj

@app.get("/data")
def get_parquet_data():
    train = pd.read_parquet("/app/data/train_data.parquet")
    test = pd.read_parquet("/app/data/test_data.parquet")
    #print("parquet read")

    train_json = train.to_json(orient="records")
    test_json = test.to_json(orient="records")

    return {"train": train_json, "test": test_json}

@app.post("/save")
async def save_vector(payload: dict):
    #print("payload received")
    df_json = payload["data"]
    train_test = payload["train_test"]

    df = pd.read_json(StringIO(df_json), orient="records")
    #print(df.head(10))

    df.to_parquet(Path(RAW_DATA_DIR) / f"{train_test}_data.parquet")

    return {"status": "parquet saved"}