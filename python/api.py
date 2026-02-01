import base64
import tempfile
import joblib
import numpy as np
import os

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from python.feature_extractor import extract_features

# ---------------- API KEY CONFIG ----------------
API_KEY = os.getenv("API_KEY", "hackathon-secret-key")
api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ------------------------------------------------

app = FastAPI()

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

# Load trained model
model = joblib.load("model/voice_model.pkl")

class AudioRequest(BaseModel):
    audio_base64: str


@app.post("/detect", dependencies=[Security(verify_api_key)])
def detect_voice(req: AudioRequest):
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
                "confidence": float(prob[label])
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
