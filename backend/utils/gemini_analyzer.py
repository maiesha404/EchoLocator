"""
EchoLocator — Gemini Audio Analysis Module (google.genai SDK)

Uses Google Gemini to detect AI-generated voice — works on ALL modern TTS
(ElevenLabs, OpenAI, Google WaveNet, VITS, StyleTTS2).

SETUP: Create backend/.env with:
  GEMINI_API_KEY=your_api_key_here
  Get free key: https://aistudio.google.com/apikey

QUOTA NOTE: Free tier has per-minute rate limits on audio (large input tokens).
  - The system auto-retries with back-off on 429 errors.
  - Falls back to acoustic heuristics when quota is exhausted.
  - Models tried in order: gemini-2.0-flash → gemini-2.0-flash-lite → gemini-2.5-flash

IMPORTANT: Audio chunks (2s WAV) consume ~500–1000 input tokens each.
  Free tier allows ~15 RPM but audio token quota is lower.
  For production use, enable billing at https://console.cloud.google.com.
"""

import os
import json
import re
import time

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Track which models are quota-exceeded and when to retry them
_model_backoff = {}   # model_name → unix timestamp when safe to retry
_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]


def _get_api_key():
    """Load API key from env var or .env file."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()

    for env_path in [
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        ".env",
    ]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                            return v.strip().strip('"').strip("'")
    return None


def gemini_available():
    """True if google.genai is installed and a valid API key is configured."""
    if not HAS_GENAI:
        return False
    key = _get_api_key()
    if not key:
        return False
    if "your_" in key or "placeholder" in key.lower() or len(key) < 20:
        return False
    return True


def _pick_model():
    """Return the first model not in backoff window."""
    now = time.time()
    for m in _MODELS:
        if now >= _model_backoff.get(m, 0):
            return m
    # All in backoff — return the one whose backoff expires soonest
    return min(_MODELS, key=lambda m: _model_backoff.get(m, 0))


def _parse_gemini_json(text):
    """Robustly extract JSON from Gemini response text."""
    # Strip code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try full JSON object
    json_match = re.search(r"\{[\s\S]*?\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Greedy match
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Partial field extraction fallback
    ai_m = re.search(r'"ai_probability"\s*:\s*(\d+)', text)
    ver_m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
    conf_m = re.search(r'"confidence"\s*:\s*"(\w+)"', text)
    ind_m = re.findall(r'"([^"]{10,80})"', text)

    if ai_m:
        return {
            "ai_probability": int(ai_m.group(1)),
            "verdict": ver_m.group(1).upper() if ver_m else "UNCERTAIN",
            "confidence": conf_m.group(1).upper() if conf_m else "MEDIUM",
            "key_indicators": ind_m[:3] if ind_m else [],
        }

    return None


_ANALYSIS_PROMPT = """You are an expert AI voice forensics system.

Analyze this audio and determine: is this voice AI-generated (TTS/synthetic) or a real human?

Key AI voice signs: robotically perfect intonation, no breathing, no hesitations ("um"/"uh"), unnaturally consistent energy, hyper-smooth pitch, no vocal fry, silent background with no room acoustics.

Key human signs: natural pitch jitter, breath sounds, dynamic energy, slight imperfections, natural room reverb.

Reply with ONLY this JSON (no markdown, no extra text):
{"ai_probability": <0-100>, "verdict": "<AI|HUMAN|UNCERTAIN>", "confidence": "<HIGH|MEDIUM|LOW>", "key_indicators": ["<indicator 1>", "<indicator 2>", "<indicator 3>"]}

Score: 85-100=clearly AI, 65-84=likely AI, 40-64=uncertain, 15-39=likely human, 0-14=clearly human."""


def analyze_audio_gemini(audio_bytes, mime_type="audio/wav"):
    """
    Send audio to Gemini for AI voice detection.
    Automatically rotates models on quota errors.

    Returns:
        dict: {ai_probability, verdict, confidence, indicators}
        None: if all models are rate-limited or error occurs
    """
    if not HAS_GENAI:
        return None

    api_key = _get_api_key()
    if not api_key:
        return None

    model = _pick_model()
    now = time.time()

    if now < _model_backoff.get(model, 0):
        wait_left = int(_model_backoff[model] - now)
        print(f"[Gemini] All models in backoff. Retry in {wait_left}s. Using acoustic fallback.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model=model,
            contents=[_ANALYSIS_PROMPT, audio_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        text = response.text.strip() if response.text else ""
        data = _parse_gemini_json(text)

        if not data:
            print(f"[Gemini] Could not parse response from {model}: {text[:120]}")
            return None

        ai_prob    = max(0, min(100, int(data.get("ai_probability", 50))))
        verdict    = str(data.get("verdict", "UNCERTAIN")).upper()
        confidence = str(data.get("confidence", "MEDIUM")).upper()
        indicators = data.get("key_indicators", [])
        if isinstance(indicators, str):
            indicators = [indicators]

        print(f"[Gemini/{model}] ai={ai_prob}%, verdict={verdict}, confidence={confidence}")
        return {
            "ai_probability": ai_prob,
            "verdict":        verdict,
            "confidence":     confidence,
            "indicators":     [str(i) for i in indicators[:3]],
        }

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            # Extract retry delay from error message
            delay_match = re.search(r'retry.*?(\d+)', err, re.IGNORECASE)
            delay = int(delay_match.group(1)) + 5 if delay_match else 65
            _model_backoff[model] = time.time() + delay
            print(f"[Gemini] {model} quota exceeded. Backoff {delay}s. "
                  f"Trying next model next time.")
            # Immediately try next model in this call
            next_models = [m for m in _MODELS if m != model and
                           time.time() >= _model_backoff.get(m, 0)]
            if next_models:
                print(f"[Gemini] Trying fallback model: {next_models[0]}")
                try:
                    client2 = genai.Client(api_key=api_key)
                    audio_part2 = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                    resp2 = client2.models.generate_content(
                        model=next_models[0],
                        contents=[_ANALYSIS_PROMPT, audio_part2],
                        config=types.GenerateContentConfig(
                            temperature=0.1, max_output_tokens=512),
                    )
                    text2 = resp2.text.strip() if resp2.text else ""
                    data2 = _parse_gemini_json(text2)
                    if data2:
                        ai_prob = max(0, min(100, int(data2.get("ai_probability", 50))))
                        verdict = str(data2.get("verdict", "UNCERTAIN")).upper()
                        conf    = str(data2.get("confidence", "MEDIUM")).upper()
                        inds    = data2.get("key_indicators", [])
                        print(f"[Gemini/{next_models[0]}] ai={ai_prob}%, verdict={verdict}")
                        return {"ai_probability": ai_prob, "verdict": verdict,
                                "confidence": conf, "indicators": inds[:3]}
                except Exception as e2:
                    if "429" in str(e2):
                        _model_backoff[next_models[0]] = time.time() + 65
                    print(f"[Gemini] Fallback also failed: {e2}")
        else:
            print(f"[Gemini] Error with {model}: {type(e).__name__}: {err[:120]}")
        return None
