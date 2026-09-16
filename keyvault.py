# keyvault.py
import json
import os
from pathlib import Path
from typing import Optional


class KeyVault:
    """
    Stores OpenRouter API keys separately from user accounts.

    Keys are stored in secrets/openrouter_keys.json.
    This file should never be exposed to the user or committed to Git.
    """

    def __init__(self, path: str = "secrets/openrouter_keys.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self):
        self.path.write_text(
            json.dumps(self.keys, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Best effort: restrict file permissions on Linux/macOS.
        try:
            os.chmod(self.path, 0o600)
        except Exception:
            pass

    def set_key(self, alias: str, api_key: str):
        alias = alias.strip()

        if not alias:
            raise ValueError("Key alias is required.")

        if not api_key.startswith("sk-or-"):
            raise ValueError("OpenRouter API keys start with 'sk-or-'.")

        self.keys[alias] = api_key.strip()
        self._save()

    def get_key(self, alias: str) -> Optional[str]:
        if not alias:
            return None
        return self.keys.get(alias.strip())

    def get_for_user(self, username: str, alias: Optional[str] = None) -> Optional[str]:
        lookup_alias = alias or username
        return self.get_key(lookup_alias)