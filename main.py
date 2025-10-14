# ===============================================
# CrackPO Translator (Hugging Face API Version)
# ===============================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langdetect import detect
import requests, os, hashlib, json
from cachetools import TTLCache

API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")
HF_TOKEN = os.getenv("HF_API_TOKEN")  # <-- add this in Render
HF_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi",
    "kn": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-kn",
    "te": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te",
}

app = FastAPI(title="CrackPO Translator via HF API")
cache = TTLCache(maxsize=500, ttl=3600)

def cache_key(text, tgt): return hashlib.sha256(f"{tgt}:{text}".encode()).hexdigest()

@app.get("/")
def root(): return {"status": "ok"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/translate")
async def translate(request: Request):
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status":"error","message":"Unauthorized"})
        data = await request.json()
        text = data.get("content_md","").strip()
        tgt = data.get("target_lang","hi").strip().lower()
        if not text: return {"status":"error","message":"Empty text"}
        key = cache_key(text,tgt)
        if key in cache: return {"status":"success","cached":True,"data":cache[key]}
        src = detect(text)
        model_url = HF_ENDPOINTS.get(tgt)
        headers = {"Authorization": f"Bearer {HF_TOKEN}","Content-Type":"application/json"}
        payload = json.dumps({"inputs": text})
        resp = requests.post(model_url, headers=headers, data=payload, timeout=60)
        out = resp.json()
        if isinstance(out,list) and "translation_text" in out[0]:
            translated = out[0]["translation_text"]
        else:
            raise ValueError(str(out))
        result = {"detected_lang": src, "from": src, "to": tgt, "translated_text": translated}
        cache[key] = result
        return {"status":"success","cached":False,"data":result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})
