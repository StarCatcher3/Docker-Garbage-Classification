from fastapi import FastAPI, UploadFile, File
import os
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


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = RAW_DATA_DIR / "input" / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"status": "saved", "filename": file.filename}

@app.get("/get")
async def upload_image(file: UploadFile = File(...)):
    file_path = RAW_DATA_DIR / "archive" / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"status": "predicted", "filename": file.filename}