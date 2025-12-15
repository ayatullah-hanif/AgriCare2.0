"""
AgriCare – Disease Detection API
--------------------------------

Production-grade AI backend for cassava disease detection.

Features:
✓ ONNX EfficientNet-B3 inference
✓ Proper softmax probabilities
✓ English-first medical guidance
✓ Multilingual translation via N-ATLaS
✓ Human-in-the-loop escalation
"""

import os
import io
import logging
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# -------------------------------------------------------
# App Setup
# -------------------------------------------------------
app = FastAPI(title="AgriCare Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agricare_api")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AgriCare Disease Detection API",
        "endpoint": "/predict"
    }

# -------------------------------------------------------
# Model Setup
# -------------------------------------------------------
MODEL_PATH = "cassava_efficientnetb3_fp16.onnx"
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

INPUT_NAME = sess.get_inputs()[0].name
MODEL_DTYPE = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32

CLASS_NAMES = [
    "Cassava Bacterial Blight",
    "Cassava Brown Streak Disease",
    "Cassava Green Mottle",
    "Cassava Mosaic Disease",
    "Healthy Leaf"
]

IMG_SIZE = 300
LOW_CONF_THRESHOLD = 0.60

# -------------------------------------------------------
# Disease Recommendation Dictionary (ENGLISH SOURCE OF TRUTH)
# -------------------------------------------------------
DISEASE_RECOMMENDATIONS = {
    "Cassava Bacterial Blight": (
        "Cassava Bacterial Blight was detected. "
        "Remove and destroy infected plants. "
        "Use clean disease-free planting materials. "
        "Apply copper-based bactericides such as Copper Oxychloride. "
        "Avoid overhead irrigation to reduce spread."
    ),

    "Cassava Brown Streak Disease": (
        "Cassava Brown Streak Disease was detected. "
        "There is no chemical cure for this disease. "
        "Control whiteflies using insecticides like Imidacloprid or Thiamethoxam. "
        "Plant resistant cassava varieties and remove infected plants early."
    ),

    "Cassava Green Mottle": (
        "Cassava Green Mottle was detected. "
        "Control aphids and whiteflies using Lambda-cyhalothrin or Cypermethrin. "
        "Maintain field hygiene and use certified disease-free cuttings."
    ),

    "Cassava Mosaic Disease": (
        "Cassava Mosaic Disease was detected. "
        "There is no direct chemical cure. "
        "Control whiteflies using Imidacloprid or Acetamiprid. "
        "Uproot and destroy infected plants immediately. "
        "Plant resistant varieties recommended by extension officers."
    ),

    "Healthy Leaf": (
        "The cassava leaf is healthy. "
        "No treatment is required. "
        "Continue regular monitoring and good farm hygiene."
    )
}

# -------------------------------------------------------
# Hugging Face – N-ATLaS (TEXT TRANSLATION ONLY)
# -------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
NATLAS_URL = "https://router.huggingface.co/hf-inference/models/NCAIR1/N-ATLaS"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :].astype(MODEL_DTYPE)

def translate_text(text: str, language: str) -> str:
    if language.lower() == "english":
        return text

    prompt = f"""
Translate the following agricultural advice into {language}.
Keep it simple, clear, and farmer-friendly.

Text:
{text}
"""

    try:
        r = requests.post(
            NATLAS_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=20
        )

        if r.status_code != 200:
            logger.error(f"N-ATLaS error {r.status_code}: {r.text}")
            return text

        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", text)

    except Exception as e:
        logger.error(f"N-ATLaS translation failed: {e}")

    return text

# -------------------------------------------------------
# API Endpoint
# -------------------------------------------------------

from fastapi.responses import HTMLResponse

async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/docs" />
            <title>AgriCare API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    margin-top: 20%;
                }
            </style>
        </head>
        <body>
            <h2>AgriCare Disease Detection API</h2>
            <p>Redirecting to API documentation…</p>
            <p>
                If not redirected,
                <a href="/docs">click here to open Swagger Docs</a>.
            </p>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("english")
):
    image_bytes = await file.read()
    arr = preprocess(image_bytes)

    logits = np.squeeze(sess.run(None, {INPUT_NAME: arr})[0])
    probs = softmax(logits)

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted = CLASS_NAMES[idx]

    base_text = DISEASE_RECOMMENDATIONS[predicted]
    final_text = translate_text(base_text, language)

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "language": language,
        "recommendation_text": final_text,
        "probabilities": probs.tolist()
    }

# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=7860)
