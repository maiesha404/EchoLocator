"""
Acoustic signal analysis for EchoLocator.

Analyzes audio features and generates human-readable signal descriptions
for the trust score panel. Focuses on real acoustic properties rather than
keyword detection.

Signal dimensions:
- AI / synthetic voice likelihood
- Acoustic smoothness
- Pitch stability / variation
- Energy behavior
- Background noise realism
- Speech naturalness
"""


def compute_acoustic_signals(features):
    """
    Analyze extracted audio features and generate signal descriptions.

    Args:
        features: dict from extract_audio_features()

    Returns:
        list of signal description strings
    """
    if not features or not features.get("is_voiced"):
        return ["No voiced speech detected in this chunk"]

    signals = []

    # ── Pitch Analysis ──
    pitch_std = features.get("pitch_std", 100)
    if pitch_std < 3.0:
        signals.append("Pitch variation is unusually stable — characteristic of synthetic speech")
    elif pitch_std < 8.0:
        signals.append("Pitch variation is lower than typical human speech")
    elif pitch_std < 15.0:
        signals.append("Pitch variation is slightly reduced")
    else:
        signals.append("Natural pitch micro-fluctuations detected")

    # ── Spectral Flatness ──
    mean_flatness = features.get("mean_flatness", 0)
    if mean_flatness > 0.08:
        signals.append("Voice appears over-smoothed — missing natural harmonic complexity")
    elif mean_flatness > 0.04:
        signals.append("Spectral texture shows some synthetic characteristics")
    else:
        signals.append("Harmonic structure appears natural and complex")

    # ── Energy Behavior ──
    energy_var = features.get("energy_var", 0.001)
    mean_energy = features.get("mean_energy", 0.01)
    if energy_var < 0.00005:
        signals.append("Energy profile is unnaturally consistent — possible vocoder artifact")
    elif energy_var < 0.0002:
        signals.append("Energy variation is slightly below natural speech range")
    else:
        signals.append("Energy dynamics are within natural speech range")

    # ── Background Noise Realism ──
    mean_zcr = features.get("mean_zcr", 0.05)
    if mean_zcr < 0.02:
        signals.append("Background environment appears unusually clean")
    elif mean_zcr > 0.15:
        signals.append("High-frequency noise detected — possible processing artifact")
    else:
        signals.append("Background noise profile consistent with real environment")

    # ── Overall Energy Level ──
    if mean_energy < 0.002:
        signals.append("Signal amplitude is very low — possible noise-gated processing")

    return signals


def compute_risk_from_scores(ai_likelihood, trust_score):
    """
    Determine risk level from AI likelihood and trust score.

    Args:
        ai_likelihood: 0-100 probability of synthetic voice
        trust_score: 0-100 trust measure

    Returns:
        str: "SAFE", "SUSPICIOUS", or "HIGH"
    """
    if ai_likelihood >= 70 or trust_score <= 30:
        return "HIGH"
    elif ai_likelihood >= 40 or trust_score <= 60:
        return "SUSPICIOUS"
    else:
        return "SAFE"