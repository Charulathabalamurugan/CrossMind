from typing import Dict, Optional


DEMO_USERS = {
    "admin": {"password": "admin123", "roles": ["admin", "reviewer", "researcher"]},
    "reviewer": {"password": "reviewer123", "roles": ["reviewer", "researcher"]},
    "researcher": {"password": "researcher123", "roles": ["researcher"]},
}


def authenticate_user(username: str, password: str, allow_demo: bool = True) -> Optional[Dict[str, object]]:
    if not username:
        return None
    user = DEMO_USERS.get(username.lower())
    if user and user["password"] == password and allow_demo:
        return {"username": username.lower(), "roles": list(user["roles"])}
    return None


def user_has_role(user: Optional[Dict[str, object]], role: str) -> bool:
    if not user:
        return False
    roles = user.get("roles", []) or []
    return role in roles


def get_default_user() -> Dict[str, object]:
    return {"username": "demo", "roles": ["researcher"]}
