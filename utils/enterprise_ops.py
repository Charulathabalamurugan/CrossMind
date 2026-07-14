import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ObservabilityRecorder:
    """Record lightweight operational events for traceability and monitoring."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "artifacts", "observability.jsonl"
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def record(self, event_type: str, details: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details or {},
            "metadata": metadata or {},
        }
        self.events.append(event)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return event

    def summary(self) -> str:
        if not self.events:
            return "No operational events recorded yet."
        lines = ["### Operational Events"]
        for event in self.events[-10:]:
            details = json.dumps(event.get("details", {}), ensure_ascii=False)
            lines.append(f"- {event['timestamp']} | {event['event_type']} | {details}")
        return "\n".join(lines)


class GovernanceEngine:
    """Simple policy engine for low-confidence result handling and review routing."""

    def __init__(self, policy_name: str = "research"):
        self.policy_name = policy_name

    def evaluate_confidence(self, confidence: float, evidence_count: int = 0, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        source_count = len(set(sources or []))
        review_required = confidence < 0.6 or evidence_count < 2 or source_count < 2
        risk_level = "low" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "high"
        return {
            "policy": self.policy_name,
            "confidence": float(confidence),
            "evidence_count": int(evidence_count),
            "source_count": source_count,
            "review_required": bool(review_required),
            "risk_level": risk_level,
        }
