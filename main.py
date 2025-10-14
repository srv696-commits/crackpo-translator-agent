from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from langdetect import detect
from cachetools import TTLCache
from hashlib import sha256
import os

# === Configuration ===
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")
translation_cache = TTLCache(maxsize=500, ttl=3600)

# FastAPI app
app = FastAPI(title="CrackPO Translator (Lightweight Version)")

# Supported language models (small, public)
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

# === Utilities ===
def make_cache_key(text: str, src: str, tgt: str):
    return sha256(f"{src}:{tgt}:{text}".encode()).hexdigest()

def get_translator(src_lang: str, tgt_lang: str):
    """Load lightweight model dynamically."""
    # Map to correct model based on src→tgt
    if src_lang == "en" and tgt_lang in model_map:
        model_name = model_map[tgt_lang]
    elif tgt_lang == "en" and src_lang in reverse_model_map:
        model_name = reverse_model_map[src_lang]
    else:
        raise ValueError("Unsupported language pair")
    return pipeline("translation", model=model_name)

# === Routes ===
@app.get("/")
def home():
    return {"status": "ok", "message": "CrackPO Translator running."}

@app.post("/translate")
async def translate(request: Request):
    try:
        # 1️⃣ Security check
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        # 2️⃣ Parse input
        data = await request.json()
        text = data.get("content_md", "").strip()
        target = data.get("target_lang", "hi").strip().lower()
        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing 'content_md' field"})

        # 3️⃣ Auto-detect language
        detected = detect(text)
        src_lang = detected if detected in ["en", "hi", "kn", "te"] else "en"

        # Flip direction if already in target language
        if src_lang == target:
            src_lang, target = target, "en"

        # 4️⃣ Cache check
        key = make_cache_key(text, src_lang, target)
        if key in translation_cache:
            return {"status": "success", "cached": True, "data": translation_cache[key]}

        # 5️⃣ Run translation (model loaded only for needed pair)
        translator = get_translator(src_lang, target)
        translated = translator(text)[0]["translation_text"]

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
