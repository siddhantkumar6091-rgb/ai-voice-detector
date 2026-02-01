# ai-voice-detector
🎙️ AI vs Human Voice Detection API

A Machine Learning powered FastAPI service that detects whether an audio sample is:

✅ Human Voice
🤖 AI Generated Voice (Text-to-Speech)

Built for hackathon demonstration of AI audio forensics and voice authenticity detection.

🚀 Live API (Deployed)

Base URL

https://ai-voice-detector-h1fq.onrender.com


Swagger Docs

https://ai-voice-detector-h1fq.onrender.com/docs

🔐 API Authentication

This API is protected using an API key.

Header	Value
x-api-key	hackathon-secret-key

In Swagger → Click Authorize → Paste the key.

🧠 How It Works

Audio (.wav) is converted to features using MFCC + spectral features

A trained RandomForest ML model classifies voice as:

HUMAN

AI_GENERATED

Returns prediction + confidence score

📡 API Endpoint
POST /detect

Detect whether voice is AI or Human.

Headers
x-api-key: hackathon-secret-key
Content-Type: application/json

Request Body
{
  "audio_base64": "BASE64_ENCODED_WAV_FILE"
}

Response
{
  "result": "AI_GENERATED",
  "confidence": 0.97
}


or

{
  "result": "HUMAN",
  "confidence": 0.92
}

🛠️ Tech Stack

Python

FastAPI

Scikit-Learn

Librosa (audio feature extraction)

RandomForest Classifier

Render (Deployment)

🧪 Testing via Swagger

Open /docs

Click 🔐 Authorize

Enter API key

Use /detect endpoint

🧩 Use Cases

Detect AI generated deepfake voices

Voice authenticity verification

Audio forensic analysis

Anti-spoofing systems

Call center fraud detection

🧑‍💻 Local Setup
pip install -r requirements.txt
uvicorn python.api:app --reload

🏁 Hackathon Ready

This project demonstrates:

End-to-end ML pipeline

Audio feature engineering

Model training

API development

Production deployment

API security

👨‍💻 Author

Siddhant Kumar
