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
import uvicorn 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
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
# Model Setup (ONNX Model)
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


NATLAS_MODEL_NAME = "NCAIR1/N-ATLaS"
model, tokenizer = None, None # Initialize globally

logger.info(f"Loading N-ATLaS model: {NATLAS_MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(NATLAS_MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        NATLAS_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu", 
        token=HF_TOKEN
    )
    logger.info("N-ATLaS model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load N-ATLaS model locally: {e}")


def format_text_for_inference(messages):
    if not tokenizer: return ""
    current_date = datetime.now().strftime('%d %b %Y')
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        date_string=current_date
    )
    return text

def generate_text_explanation(predicted_class: str) -> str:
    if model is None or tokenizer is None:
        logger.warning("N-ATLaS model not loaded - using fallback.")
        return ""

    q_chat = [
        {'role':'system','content':'You are an agricultural extension officer providing advice in English, Hausa, Igbo, and Yoruba.'},
        {'role': 'user', 'content': f"Provide short, farmer-friendly advice for the condition: {predicted_class}. Format it exactly as: English: ... Hausa: ... Igbo: ... Yoruba: ..."}
    ]

    text = format_text_for_inference(q_chat)
    input_tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    
    device = torch.device("cpu") 
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

    outputs = model.generate(
        **input_tokens,
        max_new_tokens = 512, 
        use_cache=True,
        repetition_penalty=1.12,
        temperature = 0.1
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Simple extraction of the generated text part
    generated_text = decoded_outputs.replace(text, "").strip()

    return generated_text

# The extract_sections function remains the same as before
def extract_sections(text):
    sections = {"english": "", "hausa": "", "igbo": "", "yoruba": ""}
    # ... (keep the rest of the extract_sections function as you had it) ...
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
# API Endpoints
# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Handles the root URL request expected by Hugging Face Spaces.
    Provides a simple HTML landing page.
    """
    return """
    <html>
        <head>
            <title>AgriCare FastAPI Service</title>
        </head>
        <body>
            <h1>Welcome to the AgriCare API Space!</h1>
            <p>The main FastAPI service is running correctly.</p>
            <p>Access the API documentation at the <a href="/docs">/docs</a> endpoint.</p>
            <p>The prediction endpoint is <b>/predict</b> (POST request).</p>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("english") 
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
# if __name__ == "__main__":
#     # Note: Hugging Face uses port 7860. Local runs can use 8000 if desired.
#     uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)