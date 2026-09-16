# # auth.py
# import json
# import bcrypt
# from pathlib import Path

# class AuthManager:
#     def __init__(self, path: str = "users.json"):
#         self.path = Path(path)
#         self.users = self._load()

#     def _load(self) -> dict:
#         if self.path.exists():
#             return json.loads(self.path.read_text(encoding="utf-8"))
#         return {}

#     def _save(self):
#         self.path.write_text(
#             json.dumps(self.users, ensure_ascii=False, indent=2), encoding="utf-8")

#     def add_user(self, username: str, password: str,
#                  openrouter_api_key: str, default_model: str = "meta-llama/llama-3.1-8b-instruct"):
#         username = username.strip()
#         if not username or not password:
#             raise ValueError("Username and password are required.")
#         if not openrouter_api_key.startswith("sk-or-"):
#             raise ValueError("OpenRouter API keys start with 'sk-or-'.")
#         if username in self.users:
#             raise ValueError(f"User '{username}' already exists.")
#         self.users[username] = {
#             "hashed_password": bcrypt.hashpw(
#                 password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
#             "openrouter_api_key": openrouter_api_key,
#             "default_model": default_model,
#         }
#         self._save()

#     def verify(self, username: str, password: str):
#         """Return a safe profile (no password hash) or None."""
#         rec = self.users.get(username.strip())
#         if not rec:
#             return None
#         if bcrypt.checkpw(password.encode("utf-8"),
#                           rec["hashed_password"].encode("utf-8")):
#             return {
#                 "username": username.strip(),
#                 "openrouter_api_key": rec["openrouter_api_key"],
#                 "default_model": rec["default_model"],
#             }
#         return None

# auth.py
import json
import bcrypt
from pathlib import Path
from typing import Optional


class AuthManager:
    def __init__(self, path: str = "users.json"):
        self.path = Path(path)
        self.users = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self):
        self.path.write_text(
            json.dumps(self.users, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def add_user(
        self,
        username: str,
        password: str,
        key_alias: Optional[str] = None,
        default_model: str = "meta-llama/llama-3.1-8b-instruct"
    ):
        username = username.strip()
        key_alias = (key_alias or username).strip()

        if not username or not password:
            raise ValueError("Username and password are required.")

        if username in self.users:
            raise ValueError(f"User '{username}' already exists.")

        self.users[username] = {
            "hashed_password": bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8"),
            "key_alias": key_alias,
            "default_model": default_model,
        }

        self._save()

    def verify(self, username: str, password: str):
        """
        Return a safe profile or None.

        IMPORTANT:
        This should never return the OpenRouter API key.
        """
        rec = self.users.get(username.strip())

        if not rec:
            return None

        if bcrypt.checkpw(
            password.encode("utf-8"),
            rec["hashed_password"].encode("utf-8")
        ):
            return {
                "username": username.strip(),
                "key_alias": rec.get("key_alias", username.strip()),
                "default_model": rec.get(
                    "default_model",
                    "meta-llama/llama-3.1-8b-instruct"
                ),
            }

        return None