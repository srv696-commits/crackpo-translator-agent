# ===========================================================
# CrackPO Translator — HF + OpenAI fallback, CORS, chunking, cache
# ===========================================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect
from cachetools import TTLCache
import requests, os, json, hashlib, time, math

# ---------- Config ----------
API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")
HF_TOKEN = os.getenv("HF_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Order can be "hf,openai" (default) or "openai,hf"
PROVIDER_ORDER = [p.strip() for p in os.getenv("PROVIDER_ORDER", "hf,openai").split(",")]

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

HF_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi",
    "kn": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-kn",
    "te": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te",
}

LANG_NAME = {"hi": "Hindi", "kn": "Kannada", "te": "Telugu"}

translation_cache = TTLCache(maxsize=800, ttl=3600)

# ---------- App ----------
app = FastAPI(title="CrackPO Translator (HF + OpenAI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)

# ---------- Helpers ----------
def cache_key(text: str, tgt: str, provider: str) -> str:
    return hashlib.sha256(f"{tgt}:{provider}:{text}".encode()).hexdigest()

def chunk_text(text: str, max_len: int = 1500):
    """Split text into manageable chunks on paragraph boundaries."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_len:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_len:
                buf = p
            else:
                # hard-split very long paragraphs
                for i in range(0, len(p), max_len):
                    part = p[i : i + max_len]
                    if i == 0 and not buf:
                        buf = part
                    else:
                        chunks.append(buf)
                        buf = part
    if buf:
        chunks.append(buf)
    return chunks or [text]

def call_hf(model_url: str, text: str, attempts: int = 3, timeout: int = 60) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = json.dumps({"inputs": text})
    last_error = None
    for i in range(attempts):
        try:
            resp = requests.post(model_url, headers=headers, data=payload, timeout=timeout)
            print("HF STATUS:", resp.status_code)
            print("HF RAW:", resp.text[:300])
            if resp.status_code in (503, 504):
                time.sleep(2 ** i)
                continue
            out = resp.json()
            if isinstance(out, list) and out and "translation_text" in out[0]:
                return out[0]["translation_text"]
            if isinstance(out, dict) and "error" in out:
                last_error = out["error"]
                # retry if model is loading
                if "loading" in out["error"].lower():
                    time.sleep(2 ** i)
                    continue
                break
            last_error = f"Unexpected HF response: {out}"
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** i)
    raise RuntimeError(last_error or "HF translation failed")

def call_openai(text: str, tgt: str) -> str:
    """
    Use OpenAI as a fallback (or primary if requested).
    Requires OPENAI_API_KEY.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    # lazy import to keep cold start small
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are a precise translation engine. "
        "Translate the user's content into the specified target language. "
        "Preserve markdown formatting, lists, headings, math/LaTeX, and code blocks. "
        "Do not add explanations or extra text; return only the translated content."
    )
    target_name = LANG_NAME.get(tgt, tgt)

    # Chunk to avoid context limits and HF-like failures
    chunks = chunk_text(text, max_len=1500)
    translated_chunks = []

    for idx, ch in enumerate(chunks, start=1):
        msg = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Target language: {target_name} ({tgt}).\n\nText:\n{ch}"},
        ]
        # Use a small, cheap, strong model
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.2,
        )
        out = resp.choices[0].message.content.strip()
        translated_chunks.append(out)

    return "\n\n".join(translated_chunks)

def translate_with_provider(provider: str, text: str, tgt: str) -> str:
    if provider == "hf":
        if not HF_TOKEN:
            raise RuntimeError("HF_API_TOKEN is not configured")
        model_url = HF_ENDPOINTS.get(tgt)
        if not model_url:
            raise RuntimeError(f"Unsupported target_lang '{tgt}' for HF")
        return call_hf(model_url, text)
    elif provider == "openai":
        return call_openai(text, tgt)
    else:
        raise RuntimeError(f"Unknown provider '{provider}'")

def translate_smart(text: str, tgt: str, force_provider: str | None = None) -> (str, str):
    """
    Try providers in order until one succeeds. Returns (translated_text, used_provider).
    """
    providers = [force_provider] if force_provider else PROVIDER_ORDER
    errors = []
    for p in providers:
        if not p:
            continue
        try:
            result = translate_with_provider(p, text, tgt)
            return result, p
        except Exception as e:
            print(f"Provider {p} failed:", str(e))
            errors.append(f"{p}: {str(e)}")
    raise RuntimeError(" / ".join(errors))

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "message": "Translator running (HF + OpenAI fallback)."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        # Auth
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

        # Parse
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})

        text = (body.get("content_md") or "").strip()
        tgt = (body.get("target_lang") or "hi").strip().lower()
        force_provider = (body.get("provider") or "").strip().lower() or None  # optional override

        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing 'content_md'"})
        if tgt not in LANG_NAME:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Unsupported target_lang '{tgt}'"})

        # Detect source
        try:
            src = detect(text)
        except Exception:
            src = "en"

        if src == tgt:
            return {
                "status": "success",
                "cached": True,
                "provider": "none",
                "data": {"detected_lang": src, "from": src, "to": tgt, "translated_text": text},
            }

        # Cache (provider-aware: because outputs differ slightly across engines)
        k = cache_key(text, tgt, force_provider or ",".join(PROVIDER_ORDER))
        if k in translation_cache:
            payload = translation_cache[k] | {"cached": True}
            return {"status": "success", **payload}

        # Translate (smart order or forced)
        result_text, used_provider = translate_smart(text, tgt, force_provider)

        data = {
            "detected_lang": src,
            "from": src,
            "to": tgt,
            "translated_text": result_text,
        }
        translation_cache[k] = {"provider": used_provider, "data": data}

        return {"status": "success", "cached": False, "provider": used_provider, "data": data}

    except Exception as e:
        print("SERVER ERROR:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
