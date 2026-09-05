"""
In-memory call history tracking for repeat number detection.

Stores recent calls by phone number with timestamps.
Flags numbers that have called multiple times within the tracking window.
"""

import time
from collections import defaultdict

# Track calls within the last 24 hours
TRACKING_WINDOW = 86400  # 24 hours in seconds


class CallHistory:
    def __init__(self):
        # phone_number -> [timestamp, timestamp, ...]
        self._history = defaultdict(list)

    def _cleanup(self, phone_number):
        """Remove entries older than TRACKING_WINDOW."""
        cutoff = time.time() - TRACKING_WINDOW
        self._history[phone_number] = [
            ts for ts in self._history[phone_number] if ts > cutoff
        ]

    def record_call(self, phone_number):
        """Record a call from this phone number."""
        if not phone_number:
            return
        self._cleanup(phone_number)
        self._history[phone_number].append(time.time())

    def get_call_info(self, phone_number):
        """
        Get repeat call information for a phone number.

        Returns:
            dict with `repeated` (bool) and `count` (int)
        """
        if not phone_number:
            return {"repeated": False, "count": 0}

        self._cleanup(phone_number)
        count = len(self._history[phone_number])
        return {
            "repeated": count > 1,
            "count": count,
        }

    def get_all_recent(self, limit=20):
        """Get all recently tracked numbers with their call counts."""
        result = []
        for number, timestamps in self._history.items():
            self._cleanup(number)
            if timestamps:
                result.append({
                    "phone_number": number,
                    "count": len(timestamps),
                    "last_call": max(timestamps),
                })
        result.sort(key=lambda x: x["last_call"], reverse=True)
        return result[:limit]


# Global singleton
call_tracker = CallHistory()
