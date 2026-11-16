import pickle
import numpy as np
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

project_dir = Path(__file__).resolve().parent
model_path = project_dir / "model.bin"

with open(model_path, "rb") as file:
    dv, model = pickle.load(file)

app = FastAPI(title="Goodbooks Rating Predictor API")


class BookFeatures(BaseModel):
    weighted_average_rating: float = Field(..., example=4.49)
    tag_name: str = Field(..., example="fantasy")
    authors: str = Field(..., example="Brandon Sanderson")


class PredictionResponse(BaseModel):
    predicted_rating: float
    recommended: bool


@app.get("/")
def health_check():
    return {"status": "ok", "model": "Ridge(alpha=100.0)"}


@app.post("/predict", response_model=PredictionResponse)
def predict(book: BookFeatures):
    book_dict = {
        "weighted_average_rating": book.weighted_average_rating,
        "tag_name": book.tag_name.lower().strip(),
        "authors": book.authors.strip(),
    }

    X = dv.transform([book_dict])
    pred = float(model.predict(X)[0])

    return PredictionResponse(
        predicted_rating=round(pred, 4),
        recommended=pred >= 4.0,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
