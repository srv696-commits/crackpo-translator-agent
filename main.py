# ===========================================================
# CrackPO Translator (Hugging Face API Version with Debug Logs)
# ===========================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langdetect import detect
from cachetools import TTLCache
import requests, os, json, hashlib

# === CONFIG ===
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Hugging Face API token
HF_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi",
    "kn": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-kn",
    "te": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te",
}

# Cache translations for 1 hour
translation_cache = TTLCache(maxsize=500, ttl=3600)
app = FastAPI(title="CrackPO Translator (Hugging Face API)")

# === UTILITIES ===
def make_cache_key(text: str, tgt: str) -> str:
    return hashlib.sha256(f"{tgt}:{text}".encode()).hexdigest()

# === ROUTES ===
@app.get("/")
def home():
    return {"status": "ok", "message": "Translator running via Hugging Face API."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        # 1️⃣ Authenticate
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        # 2️⃣ Parse request
        data = await request.json()
        text = data.get("content_md", "").strip()
        tgt = data.get("target_lang", "hi").strip().lower()
        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing content_md"})

        # 3️⃣ Detect source language
        try:
            src_lang = detect(text)
        except Exception:
            src_lang = "en"

        if src_lang == tgt:
            return {"status": "success", "cached": True,
                    "data": {"detected_lang": src_lang, "from": src_lang, "to": tgt, "translated_text": text}}

        # 4️⃣ Cache check
        key = make_cache_key(text, tgt)
        if key in translation_cache:
            return {"status": "success", "cached": True, "data": translation_cache[key]}

        # 5️⃣ Call Hugging Face Inference API
        model_url = HF_ENDPOINTS.get(tgt)
        if not model_url:
            raise ValueError(f"Unsupported target language: {tgt}")

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({"inputs": text})

        resp = requests.post(model_url, headers=headers, data=payload, timeout=60)

        # Debug logs (visible in Render → Logs)
        print("HF STATUS:", resp.status_code)
        print("HF RAW:", resp.text[:500])

        # 6️⃣ Parse response
        try:
            out = resp.json()
        except Exception as err:
            raise ValueError(f"Hugging Face returned invalid JSON ({err}): {resp.text[:200]}")

        if isinstance(out, list) and "translation_text" in out[0]:
            translated = out[0]["translation_text"]
        elif isinstance(out, dict) and "error" in out:
            raise RuntimeError(out["error"])
        else:
            raise ValueError(f"Unexpected Hugging Face response: {out}")

        result = {
            "detected_lang": src_lang,
            "from": src_lang,
            "to": tgt,
            "translated_text": translated,
        }

        translation_cache[key] = result
        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
