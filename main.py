# ===========================================================
# CrackPO Translator — FastAPI (HF Inference API + CORS + Cache)
# ===========================================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect
from cachetools import TTLCache
import requests, os, json, hashlib, time

# ---------- Configuration ----------
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")

# Hugging Face token with "Inference Providers" permission
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# Allow your Lovable domain (comma-separated) or "*" while testing
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Supported target languages → HF endpoints (en → target)
HF_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi",
    "kn": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-kn",
    "te": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te",
}

# Cache translations for 1 hour (keyed by text+lang)
translation_cache = TTLCache(maxsize=500, ttl=3600)

# ---------- App ----------
app = FastAPI(title="CrackPO Translator (HF API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,         # e.g. "https://your-lovable.app"
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)

# ---------- Helpers ----------
def cache_key(text: str, tgt: str) -> str:
    return hashlib.sha256(f"{tgt}:{text}".encode()).hexdigest()

def call_hf_with_retry(model_url: str, text: str, hf_token: str, attempts: int = 3, timeout: int = 60) -> str:
    """
    Call Hugging Face Inference API with small exponential backoff to handle cold starts.
    Returns the translated string or raises an Exception.
    """
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"inputs": text})

    for i in range(attempts):
        try:
            resp = requests.post(model_url, headers=headers, data=payload, timeout=timeout)
            # Debug (visible in Render logs)
            print("HF STATUS:", resp.status_code)
            print("HF RAW:", resp.text[:500])

            # If HF is waking up the model
            if resp.status_code in (503, 504):
                # backoff: 1s, 2s, 4s
                time.sleep(2 ** i)
                continue

            # Parse JSON
            out = resp.json()

            # Expected success: list with {"translation_text": "..."}
            if isinstance(out, list) and out and "translation_text" in out[0]:
                return out[0]["translation_text"]

            # Error payloads are often dicts with "error"
            if isinstance(out, dict) and "error" in out:
                # If it's a "model loading" message, retry
                if "loading" in out["error"].lower() and i < attempts - 1:
                    time.sleep(2 ** i)
                    continue
                raise RuntimeError(out["error"])

            raise ValueError(f"Unexpected HF response: {out}")

        except requests.exceptions.Timeout:
            if i < attempts - 1:
                time.sleep(2 ** i)
                continue
            raise TimeoutError("Hugging Face request timed out")
        except requests.exceptions.RequestException as e:
            if i < attempts - 1:
                time.sleep(2 ** i)
                continue
            raise RuntimeError(f"Network error contacting Hugging Face: {str(e)}")

    # If we fall through attempts
    raise RuntimeError("Unable to get translation after retries")

# ---------- Routes ----------
@app.get("/")
def home():
    return {"status": "ok", "message": "Translator running via Hugging Face API."}

@app.get("/health")
def health():
    # lightweight check that doesn’t call HF
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        # 1) API key guard
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        # 2) Parse input
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})

        text = (payload.get("content_md") or "").strip()
        tgt = (payload.get("target_lang") or "hi").strip().lower()

        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing 'content_md'"})
        if tgt not in HF_ENDPOINTS:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Unsupported target_lang '{tgt}'"})

        # 3) Detect source language (best-effort)
        try:
            src_lang = detect(text)
        except Exception:
            src_lang = "en"

        # If user asks for same language as input, just return original to avoid nonsense
        if src_lang == tgt:
            return {
                "status": "success",
                "cached": True,
                "data": {
                    "detected_lang": src_lang,
                    "from": src_lang,
                    "to": tgt,
                    "translated_text": text,
                },
            }

        # 4) Cache check
        k = cache_key(text, tgt)
        if k in translation_cache:
            return {"status": "success", "cached": True, "data": translation_cache[k]}

        # 5) Call Hugging Face (en -> tgt)
        if not HF_TOKEN:
            return JSONResponse(status_code=500, content={"status": "error", "message": "HF_API_TOKEN is not configured"})

        model_url = HF_ENDPOINTS[tgt]
        translated = call_hf_with_retry(model_url, text, HF_TOKEN)

        result = {
            "detected_lang": src_lang,
            "from": "en" if src_lang == "en" else src_lang,  # informative only
            "to": tgt,
            "translated_text": translated,
        }
        translation_cache[k] = result

        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        # Never leak stack traces to clients; put details in logs instead if needed
        print("SERVER ERROR:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
