""" FastAPI service for classifying emails """

from contextlib import asynccontextmanager
from pathlib import Path
import os

import pandas as pd

from fastapi import FastAPI, Request

from capstone.features import engineer_features
from capstone.predictor import Predictor, load_predictor
from capstone.schemas import Message, ScoreRequest, ScoreResult, ScoreResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    manifest_path = Path(os.environ["MODEL_MANIFEST_PATH"])
    app.state.predictor = load_predictor(manifest_path)
    yield

app = FastAPI(lifespan=lifespan)

def build_features(messages: list[Message], feature_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Subject": [m.subject for m in messages],
        "Body": [m.body for m in messages]
    })
    df = engineer_features(df)
    df["text"] = df["Subject"] + " " + df["Body"]
    return df[["text", *feature_cols]]

@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    predictor : Predictor = request.app.state.predictor
    df = build_features(payload.messages, predictor.feature_cols)
    probabilities = predictor.predict(df)

    return ScoreResponse(
        results=[ScoreResult(spam_probability=float(p)) for p in probabilities]
    )

@app.get("/health")
def health(request : Request) -> dict:
    predictor: Predictor = request.app.state.predictor
    return {
        "status": "ok",
        "model_type": predictor.model_type,
    }