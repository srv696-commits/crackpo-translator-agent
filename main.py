from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
from cachetools import TTLCache
from hashlib import sha256
import os

API_KEY = os.getenv("TRANSLATOR_API_KEY", "crackpo123")

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

lang_map = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu", "en": "eng_Latn"}
detect_map = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu", "en": "eng_Latn"}
translation_cache = TTLCache(maxsize=500, ttl=3600)

app = FastAPI(title="CrackPO AI Translator")

def make_cache_key(text, src, tgt):
    return sha256(f"{src}:{tgt}:{text}".encode()).hexdigest()

@app.post("/translate")
async def translate(request: Request):
    try:
        data = await request.json()
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"status":"error","message":"Unauthorized"})
        text = data.get("content_md","").strip()
        lang = data.get("target_lang","hi").strip().lower()
        if not text: return JSONResponse(status_code=400,content={"status":"error","message":"Missing content"})
        tgt_lang = lang_map.get(lang,"hin_Deva")
        detected = detect(text)
        src_lang = detect_map.get(detected,"eng_Latn")
        if src_lang == tgt_lang: src_lang, tgt_lang = tgt_lang, "eng_Latn"
        key = make_cache_key(text, src_lang, tgt_lang)
        if key in translation_cache:
            return {"status":"success","cached":True,"data":translation_cache[key]}
        out = translator(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)[0]["translation_text"]
        result = {"detected_lang":detected,"from":src_lang,"to":tgt_lang,"translated_text":out}
        translation_cache[key] = result
        return {"status":"success","cached":False,"data":result}
    except Exception as e:
        return JSONResponse(status_code=500,content={"status":"error","message":str(e)})
