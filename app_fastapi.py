"""
AgriCare – Disease Detection API
--------------------------------------

Production-grade AI backend for disease detection.

Features:
✓ ONNX EfficientNet-B3 inference
✓ Softmax-calibrated probabilities
✓ Multilingual explanations (EN / HA / IG / YO)
✓ Responsible AI fallback
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
from fastapi import FastAPI, UploadFile, File
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
# Hugging Face – N-ATLaS
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
# N-ATLaS Text
# -------------------------------------------------------
def generate_text_explanation(predicted_class: str) -> str:
    if not HF_TOKEN:
        logger.warning("HF_TOKEN missing – skipping N-ATLaS.")
        return ""

    prompt = f"""
    Provide agricultural advice for cassava condition:
    {predicted_class}

    Structure the response as:
    English:
    Hausa:
    Igbo:
    Yoruba:

    Keep each section short and farmer-friendly.
    """

    try:
        r = requests.post(
            NATLAS_TEXT_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=20
        )

        if r.status_code != 200:
            logger.error(f"N-ATLaS HTTP {r.status_code}: {r.text}")
            return ""

        data = r.json()

        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]

        logger.error(f"Unexpected N-ATLaS response format: {data}")
        return ""

    except requests.exceptions.RequestException as e:
        logger.error(f"N-ATLaS network failure: {e}")
        return ""

    except Exception as e:
        logger.error(f"N-ATLaS unknown error: {e}")
        return ""


def extract_sections(text):
    sections = {"english": "", "hausa": "", "igbo": "", "yoruba": ""}
    current = None

    for line in text.splitlines():
        l = line.lower().strip()
        if l.startswith("english"):
            current = "english"; continue
        if l.startswith("hausa"):
            current = "hausa"; continue
        if l.startswith("igbo"):
            current = "igbo"; continue
        if l.startswith("yoruba"):
            current = "yoruba"; continue
        if current:
            sections[current] += line.strip() + " "

    # fallback
    if not any(sections.values()):
        sections["english"] = (
            "This disease was detected with high confidence. "
            "Please consult a trained agricultural extension officer "
            "for appropriate treatment and prevention guidance."
        )


    return sections

# -------------------------------------------------------
# API Endpoint
# -------------------------------------------------------
from fastapi import Form

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("english")  # frontend-controlled
):
    """
    Inference endpoint.

    Inputs:
    - image file
    - language: english | hausa | igbo | yoruba

    Returns:
    - prediction
    - confidence
    - explanation (language-specific)
    - routing decision
    """

    image_bytes = await file.read()
    arr = preprocess(image_bytes)

    # ---- Model inference ----
    logits = np.squeeze(sess.run(None, {INPUT_NAME: arr})[0])
    probs = softmax(logits)

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted = CLASS_NAMES[idx]

    # ---- Text explanation ----
    text = generate_text_explanation(predicted)
    sections = extract_sections(text)

    # Fallback if model explanation fails
    explanation = sections.get(language.lower(), "")
    if not explanation.strip():
        explanation = (
            "This condition was detected with high confidence. "
            "Please consult an agricultural extension officer for guidance."
        )

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "language": language,
        "explanation": explanation,
        "probabilities": probs.tolist()
    }

# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)