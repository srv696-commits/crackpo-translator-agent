# ===============================================
# CrackPO Translator Agent (Free-Tier Optimized)
# ===============================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from langdetect import detect
from cachetools import TTLCache
from hashlib import sha256
import os

# === Configuration ===
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")

# Cache to speed up repeated translations
translation_cache = TTLCache(maxsize=500, ttl=3600)

# Initialize FastAPI
app = FastAPI(title="CrackPO Translator (Lightweight Version)")

# Supported lightweight Helsinki models
model_map = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
    "te": "Helsinki-NLP/opus-mt-en-te",
}
reverse_model_map = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "kn": "Helsinki-NLP/opus-mt-kn-en",
    "te": "Helsinki-NLP/opus-mt-te-en",
}

# === Utility Functions ===
def make_cache_key(text: str, src: str, tgt: str):
    """Create a unique key for caching translations."""
    return sha256(f"{src}:{tgt}:{text}".encode()).hexdigest()

def get_translator(src_lang: str, tgt_lang: str):
    """Lazy load the smallest translation model per language pair."""
    if src_lang == "en" and tgt_lang in model_map:
        model_name = model_map[tgt_lang]
    elif tgt_lang == "en" and src_lang in reverse_model_map:
        model_name = reverse_model_map[src_lang]
    else:
        raise ValueError(f"Unsupported language pair: {src_lang}->{tgt_lang}")
    return pipeline("translation", model=model_name)

# === Routes ===
@app.get("/")
def home():
    """Root endpoint."""
    return {"status": "ok", "message": "CrackPO Translator running."}

@app.get("/health")
def health():
    """Health endpoint for Render uptime checks."""
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        # 1️⃣ API key validation
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        # 2️⃣ Parse request
        data = await request.json()
        text = data.get("content_md", "").strip()
        target = data.get("target_lang", "hi").strip().lower()

        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing 'content_md' field"})

        # 3️⃣ Detect source language (simple heuristic)
        try:
            detected = detect(text)
        except Exception:
            detected = "en"
        src_lang = detected if detected in ["en", "hi", "kn", "te"] else "en"

        # Prevent same-language translations
        if src_lang == target:
            return {"status": "success", "cached": True, "data": {
                "detected_lang": src_lang,
                "from": src_lang,
                "to": target,
                "translated_text": text,
            }}

        # 4️⃣ Cache lookup
        key = make_cache_key(text, src_lang, target)
        if key in translation_cache:
            return {"status": "success", "cached": True, "data": translation_cache[key]}

        # 5️⃣ Run translation (load only one model at a time)
        translator = get_translator(src_lang, target)
        translated = translator(text, max_length=400)[0]["translation_text"]

        result = {
            "detected_lang": src_lang,
            "from": src_lang,
            "to": target,
            "translated_text": translated,
        }

        # Cache and respond
        translation_cache[key] = result
        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
