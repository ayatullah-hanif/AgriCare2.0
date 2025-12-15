"""
AgriCare – Disease Detection API
--------------------------------

Production-grade AI backend for cassava disease detection.

Features:
✓ ONNX EfficientNet-B3 inference
✓ Proper softmax probabilities
✓ English-first agricultural guidance
✓ Multilingual translation via N-ATLaS
✓ Human-in-the-loop escalation

Designed for real-world Nigerian agriculture.
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
# Disease Recommendations (ENGLISH SOURCE OF TRUTH)
# -------------------------------------------------------
DISEASE_RECOMMENDATIONS = {
    "Cassava Bacterial Blight": {
        "english": (
            "Cassava Bacterial Blight was detected.\n"
            "• Remove and destroy infected plants.\n"
            "• Use clean, disease-free planting materials.\n"
            "• Apply copper-based bactericides such as Copper Oxychloride.\n"
            "• Avoid overhead irrigation to reduce disease spread."
        )
    },

    "Cassava Brown Streak Disease": {
        "english": (
            "Cassava Brown Streak Disease was detected.\n"
            "• There is no chemical cure for this disease.\n"
            "• Control whiteflies using insecticides like Imidacloprid or Thiamethoxam.\n"
            "• Plant resistant cassava varieties.\n"
            "• Remove and destroy infected plants early."
        )
    },

    "Cassava Green Mottle": {
        "english": (
            "Cassava Green Mottle was detected.\n"
            "• Control insect vectors such as aphids and whiteflies.\n"
            "• Use insecticides like Lambda-cyhalothrin or Cypermethrin.\n"
            "• Maintain field hygiene.\n"
            "• Use certified disease-free planting materials."
        )
    },

    "Cassava Mosaic Disease": {
        "english": (
            "Cassava Mosaic Disease was detected.\n"
            "• No direct chemical cure exists.\n"
            "• Control whiteflies using Imidacloprid or Acetamiprid.\n"
            "• Uproot and destroy infected plants immediately.\n"
            "• Plant resistant cassava varieties."
        )
    },

    "Healthy Leaf": {
        "english": (
            "The cassava leaf is healthy.\n"
            "• No treatment is required.\n"
            "• Continue monitoring your farm.\n"
            "• Maintain good agricultural practices."
        )
    }
}

# -------------------------------------------------------
# Hugging Face – N-ATLaS (TEXT TRANSLATION ONLY)
# -------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

HF_BASE = "https://router.huggingface.co/hf-inference/models"
NATLAS_TEXT_URL = f"{HF_BASE}/NCAIR1/N-ATLaS"

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

# -------------------------------------------------------
# Translation via N-ATLaS
# -------------------------------------------------------
def translate_text(text: str, language: str) -> str:
    if language.lower() == "english":
        return text

    prompt = f"""
    Translate the following agricultural advice into {language}.
    Keep it simple and farmer-friendly.

    Text:
    {text}
    """

    try:
        r = requests.post(
            NATLAS_TEXT_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=20
        )

        r.raise_for_status()

        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", text)

        return text

    except Exception as e:
        logger.error(f"N-ATLaS translation failed: {e}")
        return text

# -------------------------------------------------------
# API Endpoint
# -------------------------------------------------------
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

    # English source text
    base_text = DISEASE_RECOMMENDATIONS[predicted]["english"]

    # Translate if needed
    final_text = translate_text(base_text, language)

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "language": language,
        "recommendation_text": final_text,
        "audio_available": False,
        "probabilities": probs.tolist()
    }

# -------------------------------------------------------
# Run (HF-compatible)
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=7860)
