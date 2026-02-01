import base64
import tempfile
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from python.feature_extractor import extract_features


app = FastAPI()

model = joblib.load("model/voice_model.pkl")

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect")
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
                "result": "AI_GENERATED" if label ==1 else "HUMAN",
                "confidence": float(prob[label])
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
