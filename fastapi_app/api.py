from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from typing import List
import shutil
import os

app = FastAPI()

model = YOLO('model/best2.pt')

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Prediction(BaseModel):
    label: str
    confidence: float
    box: List[float]


@app.post("/predict", response_model=List[Prediction])
async def predict_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(source=file_path, conf=0.1)

    predictions = []

    for result in results:
        for box in result.boxes:
            predictions.append({
                "label": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy.tolist()
            })

    os.remove(file_path)

    return JSONResponse(content=predictions)


@app.get("/")
def read_root():
    return {"message": "API is running. Use POST /predict to make predictions."}
