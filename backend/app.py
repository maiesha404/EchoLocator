"""
EchoLocator Backend — Real-Time AI Voice Phishing Detection API
"""

import os
import traceback
import uuid
from flask import Flask, request, jsonify

from utils.audio import (
    remove_background_noise,
    extract_audio_features,
    preprocess_for_aasist,
)
from utils.detector import compute_acoustic_signals, compute_risk_from_scores
from utils.predictor import (
    predictor,
    hybrid_predict,
    has_speech,
    scores_to_risk,
    session_buffer,
)
from utils.session_smoothing import smoother
from utils.call_history import call_tracker

app = Flask(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "EchoLocator Backend",
        "model_available": predictor.available,
    })


@app.route("/analyze-chunk", methods=["POST"])
def analyze_chunk():
    """
    Analyze a 2-second audio chunk for AI voice / spoof detection.

    Expects multipart/form-data:
        audio        (file) — WAV audio chunk
        session_id   (str)  — session identifier
        phone_number (str)  — optional caller number
        timestamp    (str)  — optional
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    session_id   = request.form.get("session_id", str(uuid.uuid4()))
    phone_number = request.form.get("phone_number", "")

    temp_path  = f"temp_{session_id[:8]}.wav"
    clean_path = f"clean_{session_id[:8]}.wav"
    audio_file.save(temp_path)

    try:
        # ── Step 1: Noise reduction ──
        clean_path = remove_background_noise(temp_path, clean_path)

        # ── Step 2: Load waveform for AASIST ──
        waveform = preprocess_for_aasist(clean_path)

        # ── Step 2b: Read clean WAV bytes for Gemini ──
        audio_bytes = None
        try:
            with open(clean_path, "rb") as f:
                audio_bytes = f.read()
        except Exception:
            try:
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
            except Exception:
                pass

        # ── Step 3: Voice Activity Detection ──
        if waveform is None or not has_speech(waveform):
            return jsonify({
                "trust_score": 90,
                "ai_likelihood": 5,
                "risk": "SAFE",
                "confidence": 30,
                "signals": ["No voiced speech detected — awaiting speaker input"],
                "repeated_number": False,
                "repeat_call_count": 0,
            })

        # ── Step 4: Extract acoustic features ──
        features = extract_audio_features(clean_path) or {}

        # ── Step 5: Hybrid prediction (Gemini primary + acoustics fallback) ──
        prediction = hybrid_predict(session_id, waveform, features, audio_bytes=audio_bytes)

        ai_likelihood = prediction["ai_likelihood"]
        trust_score   = prediction["trust_score"]
        signals       = prediction["signals"]

        # ── Step 6: Risk classification ──
        raw_risk = scores_to_risk(ai_likelihood)

        # ── Step 7: Temporal smoothing ──
        smoothed = smoother.smooth(
            session_id=session_id,
            raw_trust_score=trust_score,
            raw_ai_likelihood=ai_likelihood,
            raw_risk=raw_risk,
            signals=signals,
        )

        # ── Step 8: Repeat caller check ──
        call_info = call_tracker.get_call_info(phone_number)
        if call_info["repeated"]:
            smoothed["signals"].append(
                f"Repeated caller behavior detected — {call_info['count']} calls"
            )

        return jsonify({
            "trust_score":        smoothed["trust_score"],
            "ai_likelihood":      smoothed["ai_likelihood"],
            "risk":               smoothed["risk"],
            "confidence":         smoothed["confidence"],
            "signals":            smoothed["signals"],
            "repeated_number":    call_info["repeated"],
            "repeat_call_count":  call_info["count"],
        })

    except Exception as e:
        print(f"[App] Error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        for f in [temp_path, clean_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


@app.route("/record-call", methods=["POST"])
def record_call():
    data = request.json or {}
    phone_number = data.get("phone_number", "")
    if phone_number:
        call_tracker.record_call(phone_number)
    return jsonify(call_tracker.get_call_info(phone_number))


@app.route("/end-session", methods=["POST"])
def end_session():
    """Clear session audio buffer and smoothing state when call ends."""
    data = request.json or {}
    session_id = data.get("session_id", "")
    if session_id:
        session_buffer.clear(session_id)
        smoother.reset_session(session_id)
    return jsonify({"cleared": True})


if __name__ == "__main__":
    from utils.gemini_analyzer import gemini_available
    gemini_on = gemini_available()
    print("=" * 60)
    print("  EchoLocator Backend — Starting")
    print(f"  AASIST Model: {'LOADED' if predictor.available else 'NOT FOUND'}")
    print(f"  Gemini AI:    {'ENABLED (real TTS detection)' if gemini_on else 'NOT CONFIGURED (acoustic fallback)'}")
    if not gemini_on:
        print("  Get free key: https://aistudio.google.com/apikey")
        print("  Add to backend/.env: GEMINI_API_KEY=your_key")
    print("=" * 60)
    app.run(debug=True, port=5001, host="0.0.0.0")