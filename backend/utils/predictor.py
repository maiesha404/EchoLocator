"""
EchoLocator — Detection Engine v4

PRIMARY:   Gemini API audio analysis (if API key configured)
           → Gemini actually HEARS the audio and understands naturalness,
             prosody, vocoder artifacts — works on ANY modern TTS.
FALLBACK:  Acoustic heuristics (if no API key)
           → Pitch CV, Spectral Flatness, ZCR stability.
             Calibrated on simulated TTS/human signals.

HOW TO ENABLE GEMINI:
  Create backend/.env with:
    GEMINI_API_KEY=your_key_here
  Or set environment variable: export GEMINI_API_KEY=your_key
"""

import numpy as np
import os

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "aasist.onnx")
AASIST_SR = 16000
AASIST_SAMPLES = 64600


# ══════════════════════════════════════════════════════════════════
# SESSION AUDIO BUFFER
# ══════════════════════════════════════════════════════════════════
class SessionBuffer:
    def __init__(self, max_samples=AASIST_SAMPLES * 2):
        self._buffers = {}
        self._max = max_samples

    def append(self, session_id, waveform):
        if session_id not in self._buffers:
            self._buffers[session_id] = np.array([], dtype=np.float32)
        buf = np.concatenate([self._buffers[session_id], waveform])
        if len(buf) > self._max:
            buf = buf[-self._max:]
        self._buffers[session_id] = buf
        return buf

    def get(self, session_id):
        return self._buffers.get(session_id, np.array([], dtype=np.float32))

    def clear(self, session_id):
        self._buffers.pop(session_id, None)


session_buffer = SessionBuffer()


# ══════════════════════════════════════════════════════════════════
# AASIST ONNX (supplemental — for future signal text)
# ══════════════════════════════════════════════════════════════════
class AASISTPredictor:
    def __init__(self):
        self.session = None
        self.available = False
        self._try_load()

    def _try_load(self):
        if not HAS_ORT:
            return
        abs_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(abs_path):
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(abs_path, opts)
            self.available = True
            print(f"[Predictor] AASIST loaded: {abs_path}")
        except Exception as e:
            print(f"[Predictor] AASIST load failed: {e}")

    def predict_raw(self, waveform):
        if not self.available:
            return None
        try:
            x = waveform.astype(np.float32)
            if len(x) < AASIST_SAMPLES:
                pad = np.zeros(AASIST_SAMPLES - len(x), dtype=np.float32)
                x = np.concatenate([x, pad])
            else:
                x = x[-AASIST_SAMPLES:]
            raw = self.session.run(None, {"audio": x[np.newaxis, :]})[0][0]
            e = np.exp(raw - raw.max())
            p = e / e.sum()
            return float(p[0]), float(p[1])
        except Exception:
            return None


predictor = AASISTPredictor()


# ══════════════════════════════════════════════════════════════════
# VOICE ACTIVITY DETECTION
# ══════════════════════════════════════════════════════════════════
def has_speech(waveform, sr=AASIST_SR, energy_threshold=0.0005):
    """Returns True if chunk contains audible voice (low threshold catches speakers)."""
    if len(waveform) == 0:
        return False
    try:
        rms = librosa.feature.rms(y=waveform.astype(np.float32))[0]
        return float(np.mean(rms)) > energy_threshold
    except Exception:
        return float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2))) > energy_threshold


# ══════════════════════════════════════════════════════════════════
# ACOUSTIC HEURISTICS (FALLBACK when no Gemini key)
# Calibrated: TTS pitch_cv < 0.04, flatness < 0.010
# ══════════════════════════════════════════════════════════════════
def compute_acoustic_ai_score(waveform, sr=AASIST_SR):
    """
    3-feature acoustic detector.
    Returns 0–100 (0=human, 100=AI).

    NOTE: Works for synthetic TTS signals. For real modern neural TTS
    (ElevenLabs, Google WaveNet, OpenAI TTS), prefer Gemini API.
    """
    if not HAS_LIBROSA:
        return 50

    y = waveform.astype(np.float32)
    if len(y) < 1600:
        return 0

    # Normalize amplitude
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        return 0
    y = y / peak

    score = 0

    # ── Pitch CV (45 pts) ──
    try:
        f0 = librosa.yin(y, fmin=60, fmax=500, hop_length=80)
        valid = f0[(~np.isnan(f0)) & (f0 > 50) & (f0 < 500)]
        if len(valid) >= 10:
            cv = float(np.std(valid)) / max(float(np.mean(valid)), 1e-6)
            if cv < 0.008:
                score += 45
            elif cv < 0.020:
                score += 40
            elif cv < 0.040:
                score += 30
            elif cv < 0.060:
                score += 15
            elif cv < 0.080:
                score += 5
        else:
            score += 8
    except Exception:
        pass

    # ── Spectral Flatness (35 pts) ──
    try:
        flat = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=80)[0]
        mf = float(np.mean(flat))
        if mf < 0.003:
            score += 35
        elif mf < 0.010:
            score += 28
        elif mf < 0.025:
            score += 18
        elif mf < 0.040:
            score += 8
    except Exception:
        pass

    # ── ZCR Stability (20 pts) ──
    try:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=80)[0]
        mean_zcr = float(np.mean(zcr))
        if mean_zcr > 1e-4:
            zcr_cv = float(np.std(zcr)) / mean_zcr
            if zcr_cv < 0.05:
                score += 20
            elif zcr_cv < 0.08:
                score += 14
            elif zcr_cv < 0.12:
                score += 6
    except Exception:
        pass

    return min(100, max(0, score))


def get_feature_values(waveform, sr=AASIST_SR):
    """Extract raw feature values for signal text."""
    y = waveform.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak

    result = {"pitch_cv": None, "flatness": None, "zcr_cv": None}
    try:
        f0 = librosa.yin(y, fmin=60, fmax=500, hop_length=80)
        valid = f0[(~np.isnan(f0)) & (f0 > 50) & (f0 < 500)]
        if len(valid) >= 10:
            result["pitch_cv"] = float(np.std(valid)) / max(float(np.mean(valid)), 1e-6)
    except Exception:
        pass
    try:
        flat = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=80)[0]
        result["flatness"] = float(np.mean(flat))
    except Exception:
        pass
    try:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=80)[0]
        m = float(np.mean(zcr))
        if m > 1e-4:
            result["zcr_cv"] = float(np.std(zcr)) / m
    except Exception:
        pass
    return result


# ══════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
def scores_to_risk(ai_likelihood):
    if ai_likelihood >= 55:
        return "HIGH"
    elif ai_likelihood >= 30:
        return "SUSPICIOUS"
    else:
        return "SAFE"


# ══════════════════════════════════════════════════════════════════
# SIGNAL TEXT BUILDER
# ══════════════════════════════════════════════════════════════════
def build_signals_from_gemini(gemini_result, acoustic_score, raw_features):
    """Build signal list when Gemini analysis is available."""
    signals = []
    ai_prob = gemini_result["ai_probability"]
    verdict = gemini_result["verdict"]
    confidence = gemini_result["confidence"]

    # Primary Gemini verdict
    if verdict == "AI":
        signals.append(
            f"🤖 Gemini AI: Voice identified as synthetic/AI-generated "
            f"({ai_prob}% probability, {confidence.lower()} confidence)"
        )
    elif verdict == "HUMAN":
        signals.append(
            f"✅ Gemini AI: Voice identified as genuine human speech "
            f"({100-ai_prob}% human probability)"
        )
    else:
        signals.append(
            f"⚠️ Gemini AI: Voice authenticity uncertain "
            f"({ai_prob}% AI probability)"
        )

    # Gemini indicators — check both key names the parser may use
    raw_indicators = gemini_result.get("indicators") or gemini_result.get("key_indicators", [])
    for indicator in raw_indicators[:2]:
        # Skip if it looks like a raw JSON field name (short, no spaces)
        if indicator and len(str(indicator)) > 8 and ' ' in str(indicator):
            signals.append(str(indicator))

    # Acoustic supplement
    pitch_cv = raw_features.get("pitch_cv")
    if pitch_cv is not None:
        if pitch_cv < 0.04:
            signals.append(
                f"Acoustic analysis: Pitch regularity confirms synthetic pattern "
                f"(CV={pitch_cv:.3f})"
            )
        else:
            signals.append(
                f"Acoustic analysis: Natural pitch variation detected "
                f"(CV={pitch_cv:.3f})"
            )

    return signals[:5]


def build_signals_acoustic(acoustic_score, raw_features):
    """Build signal list from acoustic heuristics only."""
    signals = []
    pitch_cv = raw_features.get("pitch_cv")
    flatness  = raw_features.get("flatness")
    zcr_cv    = raw_features.get("zcr_cv")

    if pitch_cv is not None:
        if pitch_cv < 0.020:
            signals.append(f"Pitch variation is robotically stable (CV={pitch_cv:.3f}) — TTS indicator")
        elif pitch_cv < 0.050:
            signals.append(f"Pitch variation is below typical human range (CV={pitch_cv:.3f})")
        else:
            signals.append(f"Natural pitch micro-fluctuations detected (CV={pitch_cv:.3f})")

    if flatness is not None:
        if flatness < 0.010:
            signals.append(f"Spectral fingerprint is hyper-smooth (flatness={flatness:.4f}) — vocoder artifact")
        elif flatness < 0.030:
            signals.append(f"Spectral harmonics are unusually clean (flatness={flatness:.4f})")
        else:
            signals.append(f"Natural harmonic variation present (flatness={flatness:.4f})")

    if zcr_cv is not None:
        if zcr_cv < 0.07:
            signals.append("Phoneme transitions are unnaturally uniform — TTS pattern")
        else:
            signals.append("Natural phoneme transition irregularity detected")

    if acoustic_score >= 55:
        signals.append("⚠️ Multiple AI voice signatures detected — high risk call")
    elif acoustic_score >= 30:
        signals.append("Acoustic patterns partially consistent with synthetic voice")
    else:
        signals.append("No significant AI voice signatures detected in this segment")

    return signals[:5]


# ══════════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ══════════════════════════════════════════════════════════════════
def hybrid_predict(session_id, new_waveform, features=None, audio_bytes=None):
    """
    Primary detection function.

    Detection pipeline:
      IF Gemini API available (GEMINI_API_KEY set):
        → Gemini analysis (70%) + acoustic heuristics (30%)
        → Works on ALL modern TTS (ElevenLabs, OpenAI, Google WaveNet)
      ELSE:
        → Acoustic heuristics only
        → Works on basic/synthetic TTS

    Args:
        session_id:   str
        new_waveform: np.ndarray float32 at 16kHz
        features:     dict (unused, kept for API compat)
        audio_bytes:  bytes or None — raw WAV bytes for Gemini

    Returns:
        dict: ai_likelihood, trust_score, signals
    """
    from utils.gemini_analyzer import analyze_audio_gemini, gemini_available

    # Session buffer for AASIST
    accumulated = session_buffer.append(session_id, new_waveform)

    # Always compute acoustic score (fast, no API)
    acoustic_score = compute_acoustic_ai_score(new_waveform)
    raw_features   = get_feature_values(new_waveform)

    # ── Route: Gemini (primary) or acoustic (fallback) ──
    gemini_result = None
    if audio_bytes and gemini_available():
        gemini_result = analyze_audio_gemini(audio_bytes, mime_type="audio/wav")

    if gemini_result is not None:
        # ── Gemini available: 70% Gemini + 30% acoustic ──
        g_score = gemini_result["ai_probability"]
        confidence_boost = {"HIGH": 0, "MEDIUM": 5, "LOW": 15}.get(
            gemini_result.get("confidence", "MEDIUM"), 5
        )
        ai_likelihood = round(0.70 * g_score + 0.30 * acoustic_score)
        ai_likelihood = max(0, min(100, ai_likelihood - confidence_boost))
        signals = build_signals_from_gemini(gemini_result, acoustic_score, raw_features)
        print(f"[Detect] Gemini={g_score}, Acoustic={acoustic_score}, Final={ai_likelihood}")
    else:
        # ── Fallback: acoustic only ──
        if gemini_available() and not audio_bytes:
            print("[Detect] No audio_bytes provided for Gemini — acoustic fallback")
        elif not gemini_available():
            print("[Detect] No Gemini API key — acoustic fallback. "
                  "Create backend/.env with GEMINI_API_KEY=... to enable AI detection.")
        ai_likelihood = acoustic_score
        signals = build_signals_acoustic(acoustic_score, raw_features)

    ai_likelihood = max(0, min(100, ai_likelihood))
    trust_score   = max(0, 100 - ai_likelihood)

    return {
        "ai_likelihood": ai_likelihood,
        "trust_score":   trust_score,
        "signals":       signals,
        "gemini_used":   gemini_result is not None,
    }


def compute_heuristic_scores(features):
    """Legacy fallback."""
    return {"ai_likelihood": 0, "trust_score": 95}
