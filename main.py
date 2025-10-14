# ===============================================
# CrackPO Translator Agent (Free-Tier Stable Build)
# ===============================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from langdetect import detect
from cachetools import TTLCache
from hashlib import sha256
import torch
import threading
import os

# === Config ===
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")
translation_cache = TTLCache(maxsize=500, ttl=3600)

app = FastAPI(title="CrackPO Translator (Stable Free-Tier Version)")

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

def make_cache_key(text: str, src: str, tgt: str):
    return sha256(f"{src}:{tgt}:{text}".encode()).hexdigest()

def get_translator(src_lang: str, tgt_lang: str):
    if src_lang == "en" and tgt_lang in model_map:
        model_name = model_map[tgt_lang]
    elif tgt_lang == "en" and src_lang in reverse_model_map:
        model_name = reverse_model_map[src_lang]
    else:
        raise ValueError(f"Unsupported language pair {src_lang}->{tgt_lang}")
    return pipeline("translation", model=model_name)

@app.get("/")
def home():
    return {"status": "ok", "message": "Translator running."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        data = await request.json()
        text = data.get("content_md", "").strip()
        target = data.get("target_lang", "hi").strip().lower()
        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing content_md"})

        detected = detect(text)
        src_lang = detected if detected in ["en", "hi", "kn", "te"] else "en"
        if src_lang == target:
            return {"status": "success", "cached": True, "data": {"translated_text": text}}

        key = make_cache_key(text, src_lang, target)
        if key in translation_cache:
            return {"status": "success", "cached": True, "data": translation_cache[key]}

        translator = get_translator(src_lang, target)

        # timeout safeguard
        result_holder = {}

        def run_translation():
            try:
                out = translator(text, max_length=400)[0]["translation_text"]
                result_holder["output"] = out
            except Exception as e:
                result_holder["error"] = str(e)

        t = threading.Thread(target=run_translation)
        t.start()
        t.join(timeout=60)

        if "error" in result_holder:
            raise RuntimeError(result_holder["error"])
        if "output" not in result_holder:
            raise TimeoutError("Translation timed out")

        translated = result_holder["output"]
        result = {
            "detected_lang": src_lang,
            "from": src_lang,
            "to": target,
            "translated_text": translated,
        }
        translation_cache[key] = result
        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Preload a dummy model (non-blocking)
def preload_model():
    try:
        pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    except Exception:
        pass

threading.Thread(target=preload_model, daemon=True).start()
