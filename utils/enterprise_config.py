import os
from typing import Dict, Optional


class EnterpriseSettings:
    def __init__(self) -> None:
        self.app_env = os.getenv("CROSSMIND_APP_ENV", "development")
        self.secret_key = os.getenv("CROSSMIND_SECRET_KEY", "dev-secret-key")
        self.auth_enabled = os.getenv("CROSSMIND_AUTH_ENABLED", "true").lower() == "true"
        self.allow_demo_auth = os.getenv("CROSSMIND_ALLOW_DEMO_AUTH", "true").lower() == "true"
        self.audit_log_path = os.getenv("CROSSMIND_AUDIT_LOG", os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "audit_log.jsonl"))
        self.metrics_log_path = os.getenv("CROSSMIND_METRICS_LOG", os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "metrics.jsonl"))
        self.queue_backend = os.getenv("CROSSMIND_QUEUE_BACKEND", "memory")

    def to_dict(self) -> Dict[str, object]:
        return {
            "app_env": self.app_env,
            "auth_enabled": self.auth_enabled,
            "allow_demo_auth": self.allow_demo_auth,
            "audit_log_path": self.audit_log_path,
            "metrics_log_path": self.metrics_log_path,
            "queue_backend": self.queue_backend,
        }


def get_settings() -> EnterpriseSettings:
    return EnterpriseSettings()


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)
