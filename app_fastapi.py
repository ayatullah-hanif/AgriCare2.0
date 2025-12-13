"""
AgriCare – Disease Detection API
--------------------------------

Production-grade AI backend for cassava disease detection.

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
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# =====================================================
# App Setup
# =====================================================
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


# =====================================================
# ONNX Model Setup
# =====================================================
MODEL_PATH = "cassava_efficientnetb3_fp16.onnx"

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = sess.get_inputs()[0].name
MODEL_DTYPE = (
    np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
)

CLASS_NAMES = [
    "Cassava Bacterial Blight",
    "Cassava Brown Streak Disease",
    "Cassava Green Mottle",
    "Cassava Mosaic Disease",
    "Healthy Leaf"
]

IMG_SIZE = 300
LOW_CONF_THRESHOLD = 0.60


# =====================================================
# Hugging Face – N-ATLaS (Local Model)
# =====================================================
HF_TOKEN = os.getenv("HF_TOKEN")
NATLAS_MODEL_NAME = "NCAIR1/N-ATLaS"

tokenizer = None
llm_model = None

logger.info(f"Loading N-ATLaS model: {NATLAS_MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        NATLAS_MODEL_NAME,
        token=HF_TOKEN
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        NATLAS_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=HF_TOKEN
    )
    logger.info("N-ATLaS model loaded successfully.")
except Exception as e:
    logger.error(f"N-ATLaS load failed: {e}")


# =====================================================
# Helper Functions
# =====================================================
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))

    return arr[np.newaxis, :].astype(MODEL_DTYPE)


def format_chat_prompt(messages: list) -> str:
    if tokenizer is None:
        return ""

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        date_string=datetime.now().strftime("%d %b %Y")
    )


def generate_text_explanation(predicted_class: str) -> str:
    if llm_model is None or tokenizer is None:
        logger.warning("N-ATLaS unavailable — using fallback.")
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural extension officer providing advice "
                "in English, Hausa, Igbo, and Yoruba."
            )
        },
        {
            "role": "user",
            "content": (
                f"Provide short, farmer-friendly advice for the condition: "
                f"{predicted_class}. Format exactly as:\n"
                "English:\nHausa:\nIgbo:\nYoruba:"
            )
        }
    ]

    prompt = format_chat_prompt(messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.12,
        use_cache=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()


def extract_sections(text: str) -> dict:
    sections = {"english": "", "hausa": "", "igbo": "", "yoruba": ""}
    current = None

    for line in text.splitlines():
        key = line.lower().strip()

        if key.startswith("english"):
            current = "english"; continue
        if key.startswith("hausa"):
            current = "hausa"; continue
        if key.startswith("igbo"):
            current = "igbo"; continue
        if key.startswith("yoruba"):
            current = "yoruba"; continue

        if current:
            sections[current] += line.strip() + " "

    if not any(sections.values()):
        sections["english"] = (
            "This disease was detected with high confidence. "
            "Please consult a trained agricultural extension officer "
            "for proper treatment and prevention."
        )

    return sections


# =====================================================
# API Routes
# =====================================================
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>AgriCare API</title></head>
        <body>
            <h1>AgriCare API is running</h1>
            <p>Use <a href="/docs">/docs</a> for API documentation.</p>
            <p>POST images to <b>/predict</b></p>
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

    raw_text = generate_text_explanation(predicted)
    sections = extract_sections(raw_text)

    explanation = sections.get(language.lower(), "").strip()
    if not explanation:
        explanation = sections["english"]

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "language": language,
        "explanation": explanation,
        "probabilities": probs.tolist()
    }


# =====================================================
# Local Run 
# =====================================================
# uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)
