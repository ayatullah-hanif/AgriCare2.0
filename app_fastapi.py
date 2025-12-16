import os
import io
import logging
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
# Root → Docs
# -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head>
        <meta http-equiv="refresh" content="0; url=/docs">
        <title>AgriCare API</title>
      </head>
      <body>
        <p>Redirecting to <a href="/docs">/docs</a>...</p>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

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
# English Source of Truth
# -------------------------------------------------------
DISEASE_RECOMMENDATIONS = {
    "Cassava Bacterial Blight":
        "Cassava Bacterial Blight was detected. Remove and destroy infected plants. "
        "Use clean disease-free planting materials. Apply copper-based bactericides "
        "such as Copper Oxychloride. Avoid overhead irrigation.",

    "Cassava Brown Streak Disease":
        "Cassava Brown Streak Disease was detected. There is no chemical cure. "
        "Control whiteflies using Imidacloprid or Thiamethoxam. "
        "Plant resistant varieties and remove infected plants early.",

    "Cassava Green Mottle":
        "Cassava Green Mottle was detected. Control aphids and whiteflies using "
        "Lambda-cyhalothrin or Cypermethrin. Maintain field hygiene.",

    "Cassava Mosaic Disease":
        "Cassava Mosaic Disease was detected. There is no direct chemical cure. "
        "Control whiteflies using Imidacloprid or Acetamiprid. "
        "Uproot and destroy infected plants immediately.",

    "Healthy Leaf":
        "The cassava leaf is healthy. No treatment is required. "
        "Continue regular monitoring and good farm hygiene."
}

# -------------------------------------------------------
# Hugging Face – N-ATLaS
# -------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
NATLAS_URL = "https://router.huggingface.co/hf-inference/models/NCAIR1/N-ATLaS"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

LANGUAGE_MAP = {
    "yoruba": "Yoruba language",
    "hausa": "Hausa language",
    "igbo": "Igbo language"
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

    target_lang = LANGUAGE_MAP.get(language.lower(), language)

    prompt = (
        f"Translate the following agricultural advice into {target_lang}. "
        f"Do NOT answer in English.\n\n{text}"
    )

    r = requests.post(
        NATLAS_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=30
    )

    r.raise_for_status()
    data = r.json()

    translated = data[0].get("generated_text", text)

    return translated.strip()

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
    predicted = CLASS_NAMES[idx]

    base_text = DISEASE_RECOMMENDATIONS[predicted]
    final_text = translate_text(base_text, language)

    return {
        "status": "low_confidence" if confidence < LOW_CONF_THRESHOLD else "ok",
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "language": language,
        "recommendation_text": final_text,
        "route_to_expert": confidence < LOW_CONF_THRESHOLD,
        "probabilities": probs.tolist()
    }

# -------------------------------------------------------
# Run (HF Compatible)
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=7860)
