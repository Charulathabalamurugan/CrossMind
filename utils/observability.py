import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class StructuredMetrics:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "metrics.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "value": value,
            "labels": labels or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return payload

    def get_recent_metrics(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        return lines[-limit:]
