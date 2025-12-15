"""
AgriCare – Disease Detection API
--------------------------------

Production-grade AI backend for cassava disease detection.

✓ ONNX EfficientNet-B3 inference
✓ Deterministic English advice (dict-based)
✓ N-ATLaS translation + audio generation
✓ Human-in-the-loop escalation
✓ Hugging Face Spaces ready
"""

import os
import io
import logging
import base64
import numpy as np
import onnxruntime as ort
import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
# English Advice Dictionary (SOURCE OF TRUTH)
# -------------------------------------------------------
ENGLISH_ADVICE = {
    "Cassava Bacterial Blight": (
        "Cassava bacterial blight causes leaf wilting and stem rot. "
        "Remove infected plants, avoid overhead watering, and use resistant varieties."
    ),
    "Cassava Brown Streak Disease": (
        "Cassava brown streak disease damages roots and reduces yield. "
        "Use certified disease-free cuttings and control whiteflies."
    ),
    "Cassava Green Mottle": (
        "Cassava green mottle causes leaf discoloration. "
        "Remove affected plants early and maintain good farm hygiene."
    ),
    "Cassava Mosaic Disease": (
        "Cassava mosaic disease leads to distorted leaves and stunted growth. "
        "Plant resistant varieties and control whitefly populations."
    ),
    "Healthy Leaf": (
        "Your cassava plant appears healthy. "
        "Continue good farming practices and monitor regularly."
    )
}

# -------------------------------------------------------
# Hugging Face – N-ATLaS
# -------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required and must be set in Hugging Face Secrets.")

HF_BASE = "https://router.huggingface.co/hf-inference/models"
NATLAS_TEXT_URL = f"{HF_BASE}/NCAIR1/N-ATLaS"
NATLAS_TTS_URL  = f"{HF_BASE}/NCAIR1/N-ATLaS-TTS"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

LANG_CODE_MAP = {
    "english": "en",
    "hausa": "ha",
    "igbo": "ig",
    "yoruba": "yo"
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
# N-ATLaS Translation
# -------------------------------------------------------
def translate_text(text: str, target_language: str) -> str:
    if target_language == "english":
        return text

    prompt = f"""
    Translate the following agricultural advice into {target_language}.
    Keep it clear and farmer-friendly.

    Text:
    {text}
    """

    r = requests.post(
        NATLAS_TEXT_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=25
    )

    r.raise_for_status()
    data = r.json()
    return data[0]["generated_text"]

# -------------------------------------------------------
# N-ATLaS Audio
# -------------------------------------------------------
def generate_audio(text: str, language: str):
    lang_code = LANG_CODE_MAP.get(language, "en")

    r = requests.post(
        NATLAS_TTS_URL,
        headers=HEADERS,
        json={
            "inputs": text,
            "parameters": {"language": lang_code}
        },
        timeout=25
    )

    r.raise_for_status()
    return r.json().get("audio")  # base64 WAV

# -------------------------------------------------------
# Root (HF requirement)
# -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>AgriCare API is running</h2><p>Visit <a href='/docs'>/docs</a></p>"

# -------------------------------------------------------
# Prediction Endpoint
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
    disease = CLASS_NAMES[idx]

    # English source text
    english_text = ENGLISH_ADVICE[disease]

    # Translation
    final_text = translate_text(english_text, language.lower())

    # Audio
    audio = generate_audio(final_text, language.lower())

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": disease,
        "confidence": round(confidence, 4),
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "language": language,
        "text": final_text,
        "audio_base64": audio,
        "probabilities": probs.tolist()
    }

# -------------------------------------------------------
# Run (Local)
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)
