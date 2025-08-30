from datetime import datetime
from typing import Dict, Any


def record_training_event(event_name: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a simple training event payload for logging or storage.

    This is a lightweight placeholder; integrate with your DB or metrics later.
    """
    return {
        "event": event_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "details": details or {},
    }


