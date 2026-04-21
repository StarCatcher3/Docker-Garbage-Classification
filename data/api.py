from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import pandas as pd
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = Path(
    os.getenv(
        "GARBAGE_RAW_DATA_DIR",
        os.getenv("GARBAGE_DATA_DIR", BASE_DIR / "data")
    )
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/data")
async def get_file():

    df = pd.read_parquet(RAW_DATA_DIR / "prediction_data.parquet")[["file", "class", "confidence"]]

    if df.empty:
        raise HTTPException(status_code=404, detail="No prediction found")

    return {
        "status": "success",
        "data": df.to_dict(orient="records")
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = RAW_DATA_DIR / "input" / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"status": "saved", "filename": file.filename}

@app.post("/get")
async def get_file(file: UploadFile = File(...)):

    file_path = RAW_DATA_DIR / "archive" / file.filename

    # ❌ CASE 1: file does NOT exist
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found in archive")

    # ✅ CASE 2: file exists → load parquet

    print("Trying to read:", RAW_DATA_DIR / "prediction_data.parquet")

    try:
        df = pd.read_parquet(RAW_DATA_DIR / "prediction_data.parquet")
        # filter row for that file
        row = df[df["file"] == file.filename]

        if row.empty:
            raise HTTPException(status_code=404, detail="No prediction found")

        # return result
        return {
            "status": "success",
            "class_name": row["class"].iloc[0],
            "confidence": row["confidence"].iloc[0]
        }

    except Exception as e:
        print("error", str(e))
        return {
            "error": str(e),
            "path": str(RAW_DATA_DIR / "prediction_data.parquet")
        }

    