import os
import json
import time
import threading
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ------------------ CONFIG ------------------
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY", "crackpo123")

PROVIDER_ORDER = os.getenv("PROVIDER_ORDER", "openai,hf").split(",")
FAST_FALLBACK_MS = int(os.getenv("FAST_FALLBACK_MS", "6000"))  # 6 sec
MAX_RETRIES = 3

HF_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi",
    "kn": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-kn",
    "te": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
}

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ HELPERS ------------------

def chunk_text(text, max_len=1000):
    """Split long text into manageable pieces."""
    chunks, cur = [], []
    count = 0
    for line in text.splitlines(True):
        count += len(line)
        cur.append(line)
        if count >= max_len:
            chunks.append("".join(cur))
            cur, count = [], 0
    if cur:
        chunks.append("".join(cur))
    return chunks


def call_hf(model_url: str, text: str, timeout: int = 60) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = json.dumps({"inputs": text})
    r = requests.post(model_url, headers=headers, data=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")
    data = r.json()
    if isinstance(data, list) and len(data) and "translation_text" in data[0]:
        return data[0]["translation_text"]
    raise RuntimeError(f"Unexpected HF output: {data}")


def call_openai(text: str, tgt: str) -> str:
    """Use GPT model for translation (fast + stable)."""
    prompt = f"Translate this English text to {tgt} (keep markdown):\n\n{text}"
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def translate_with_provider(text: str, tgt: str, provider: str) -> tuple[str, str]:
    """Explicitly choose provider."""
    if provider == "hf":
        model = HF_ENDPOINTS.get(tgt)
        if not model:
            raise RuntimeError(f"No HF model for {tgt}")
        return call_hf(model, text), "hf"
    elif provider == "openai":
        return call_openai(text, tgt), "openai"
    raise RuntimeError(f"Invalid provider: {provider}")


def translate_hf_with_fast_fallback(text: str, tgt: str):
    """Try HF for FAST_FALLBACK_MS, then immediately fall back to OpenAI if no result."""
    result = {"text": None, "err": None}

    def run_hf():
        try:
            model = HF_ENDPOINTS.get(tgt)
            if not model:
                raise RuntimeError(f"No HF model for {tgt}")
            result["text"] = call_hf(model, text)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=run_hf, daemon=True)
    t.start()
    t.join(FAST_FALLBACK_MS / 1000.0)
    if result["text"]:
        return result["text"], "hf"
    return call_openai(text, tgt), "openai"


def translate_smart(text: str, tgt: str, force_provider=None):
    """Select provider per config order, with retries."""
    if force_provider:
        return translate_with_provider(text, tgt, force_provider)

    order = [p.strip() for p in PROVIDER_ORDER]
    for prov in order:
        try:
            if prov == "hf":
                if FAST_FALLBACK_MS > 0:
                    return translate_hf_with_fast_fallback(text, tgt)
                else:
                    model = HF_ENDPOINTS.get(tgt)
                    return call_hf(model, text), "hf"
            elif prov == "openai":
                return call_openai(text, tgt), "openai"
        except Exception as e:
            print(f"[WARN] {prov} failed: {e}")
            continue
    raise RuntimeError("All providers failed")


# ------------------ ROUTES ------------------

@app.route("/translate", methods=["POST"])
def translate():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        data = request.get_json(force=True)
        content_md = data.get("content_md", "").strip()
        tgt = data.get("target_lang", "").strip()
        force_provider = data.get("provider")

        if not content_md or tgt not in HF_ENDPOINTS:
            return jsonify({"status": "error", "message": "Invalid input"}), 400

        chunks = chunk_text(content_md)
        print(f"🔹 Translating {len(chunks)} chunk(s) to {tgt} via provider={force_provider or 'auto'}")

        full_output = []
        used_provider = None
        for i, ch in enumerate(chunks):
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    out, prov = translate_smart(ch, tgt, force_provider)
                    full_output.append(out)
                    used_provider = prov
                    break
                except Exception as e:
                    retries += 1
                    print(f"[Retry {retries}] {e}")
                    time.sleep(1.5)
            else:
                raise RuntimeError("Translation failed after retries")

        translated = "\n".join(full_output)
        return jsonify({
            "status": "success",
            "provider": used_provider or "none",
            "data": {"translated_text": translated}
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/warm")
def warm():
    """Ping endpoint to pre-warm the Render dyno."""
    return jsonify({"status": "warm"})


@app.get("/")
def root():
    return jsonify({"status": "ok", "message": "CrackPO Translator Agent active"})


# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Translator on port {port} | Providers: {PROVIDER_ORDER}")
    app.run(host="0.0.0.0", port=port)
