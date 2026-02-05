import base64
import tempfile
import joblib
import numpy as np
import os

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from python.feature_extractor import extract_features


app = FastAPI(title="AI vs Human Voice Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return RedirectResponse(url="/docs")


API_KEY = os.getenv("API_KEY")

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


model = joblib.load("model/voice_model.pkl")


# ✅ Hackathon-compatible request body
class AudioRequest(BaseModel):
    language: str | None = None
    audioFormat: str | None = None
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        populate_by_name = True


@app.post("/detect")
def detect_voice(req: AudioRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    try:
        audio_bytes = base64.b64decode(req.audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            features = extract_features(tmp.name)
            features = np.array(features).reshape(1, -1)

            prob = model.predict_proba(features)[0]
            label = int(np.argmax(prob))

            return {
                "result": "AI_GENERATED" if label == 1 else "HUMAN",
                "confidence": round(float(prob[label]), 4)
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
