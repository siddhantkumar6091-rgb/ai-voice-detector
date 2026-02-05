import base64
import tempfile
import joblib
import numpy as np
import os
import gc

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from python.feature_extractor import extract_features


# =========================
# APP INIT
# =========================
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


# =========================
# API KEY
# =========================
API_KEY = os.getenv("API_KEY")

def verify_key(x_api_key: str = Header(...)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# =========================
# LOAD MODEL (ONCE)
# =========================
model = joblib.load("model/voice_model.pkl")
EXPECTED_FEATURES = model.n_features_in_   # 🔥 AUTO SYNC


# =========================
# REQUEST SCHEMA
# =========================
class AudioRequest(BaseModel):
    language: str | None = None
    audioFormat: str | None = None
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        populate_by_name = True


# =========================
# MAIN API
# =========================
@app.post("/detect")
def detect_voice(req: AudioRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    tmp_path = None
    features = None
    audio_bytes = None

    try:
        # 🔹 Decode Base64
        audio_bytes = base64.b64decode(req.audio_base64)

        # 🔹 Write temp WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 🔹 Extract features
        features = extract_features(tmp_path)
        features = np.array(features, dtype=np.float32)

        # 🔥 FEATURE COUNT FIX
        if features.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Feature mismatch: got {features.shape[0]}, expected {EXPECTED_FEATURES}"
            )

        features = features.reshape(1, -1)

        # 🔹 Predict
        prob = model.predict_proba(features)[0]
        label = int(np.argmax(prob))

        return {
            "result": "AI_GENERATED" if label == 1 else "HUMAN",
            "confidence": round(float(prob[label]), 4)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        # 🔥 CLEANUP (VERY IMPORTANT FOR RENDER)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

        del features
        del audio_bytes
        gc.collect()
