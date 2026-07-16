import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditTrail:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "audit_log.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record_event(self, event_type: str, details: Optional[Dict[str, Any]] = None, user: Optional[str] = None, status: str = "info") -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "status": status,
            "details": details or {},
            "user": user,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return event

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        return lines[-limit:]
