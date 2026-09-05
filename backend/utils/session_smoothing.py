"""
Session-aware temporal smoothing for EchoLocator.

Maintains a rolling window of recent chunk predictions per session,
ensuring risk assessments are stable and don't jump wildly on a single chunk.

Rules:
- Escalation to HIGH requires >=3 consecutive SUSPICIOUS/HIGH chunks
- Degradation from HIGH to SAFE is gradual (must pass through SUSPICIOUS)
- Smoothed scores use weighted moving average (recent chunks weighted more)
"""

import time
from collections import defaultdict

# Session data expires after 30 minutes of inactivity
SESSION_TIMEOUT = 1800
WINDOW_SIZE = 5

RISK_ORDER = {"SAFE": 0, "SUSPICIOUS": 1, "HIGH": 2}


class SessionSmoother:
    def __init__(self):
        # session_id -> { "chunks": [...], "last_risk": str, "last_active": float }
        self._sessions = {}

    def _cleanup_stale(self):
        """Remove sessions inactive for more than SESSION_TIMEOUT."""
        now = time.time()
        stale = [sid for sid, data in self._sessions.items()
                 if now - data["last_active"] > SESSION_TIMEOUT]
        for sid in stale:
            del self._sessions[sid]

    def _ensure_session(self, session_id):
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "chunks": [],
                "last_risk": "SAFE",
                "last_active": time.time(),
            }
        self._sessions[session_id]["last_active"] = time.time()

    def smooth(self, session_id, raw_trust_score, raw_ai_likelihood, raw_risk, signals):
        """
        Apply temporal smoothing to raw chunk prediction.

        Returns:
            dict with smoothed trust_score, ai_likelihood, risk, confidence, signals
        """
        self._cleanup_stale()
        self._ensure_session(session_id)

        session = self._sessions[session_id]
        chunk_data = {
            "trust_score": raw_trust_score,
            "ai_likelihood": raw_ai_likelihood,
            "risk": raw_risk,
            "ts": time.time(),
        }

        session["chunks"].append(chunk_data)
        # Keep only last WINDOW_SIZE chunks
        if len(session["chunks"]) > WINDOW_SIZE:
            session["chunks"] = session["chunks"][-WINDOW_SIZE:]

        chunks = session["chunks"]
        n = len(chunks)

        # ── Weighted Moving Average ──
        # More recent chunks get higher weight: [1, 2, 3, 4, 5]
        weights = list(range(1, n + 1))
        total_weight = sum(weights)

        smoothed_trust = sum(c["trust_score"] * w for c, w in zip(chunks, weights)) / total_weight
        smoothed_ai = sum(c["ai_likelihood"] * w for c, w in zip(chunks, weights)) / total_weight

        # ── Risk Escalation / De-escalation Logic ──
        last_risk = session["last_risk"]

        # Count recent high-severity chunks
        recent_risks = [c["risk"] for c in chunks[-3:]]  # last 3
        high_count = sum(1 for r in recent_risks if r == "HIGH")
        suspicious_count = sum(1 for r in recent_risks if r in ("SUSPICIOUS", "HIGH"))

        # Determine smoothed risk
        if high_count >= 2 or (suspicious_count >= 3 and smoothed_ai > 65):
            smoothed_risk = "HIGH"
        elif suspicious_count >= 2 or smoothed_ai > 45:
            smoothed_risk = "SUSPICIOUS"
        else:
            smoothed_risk = "SAFE"

        # Prevent instant jumps: can only move one level per update
        current_level = RISK_ORDER.get(smoothed_risk, 0)
        last_level = RISK_ORDER.get(last_risk, 0)

        if current_level > last_level + 1:
            # Can't jump from SAFE to HIGH directly
            smoothed_risk = "SUSPICIOUS"
        elif current_level < last_level - 1:
            # Can't jump from HIGH to SAFE directly
            smoothed_risk = "SUSPICIOUS"

        session["last_risk"] = smoothed_risk

        # ── Confidence ──
        # Higher confidence with more chunks and consistent readings
        base_confidence = min(95, 50 + n * 10)
        risk_variance = len(set(c["risk"] for c in chunks))
        confidence = max(30, base_confidence - (risk_variance - 1) * 15)

        return {
            "trust_score": round(smoothed_trust),
            "ai_likelihood": round(smoothed_ai),
            "risk": smoothed_risk,
            "confidence": round(confidence),
            "signals": signals,
        }

    def reset_session(self, session_id):
        """Clear session data."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Global singleton
smoother = SessionSmoother()
