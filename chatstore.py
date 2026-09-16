# chatstore.py
import json
import uuid
from datetime import datetime
from pathlib import Path

class ChatStore:
    """Per-user persistent chat history. user_data/<username>/conversations.json"""

    def __init__(self, base_dir: str = "user_data"):
        self.base_dir = Path(base_dir)

    def _file(self, username: str) -> Path:
        d = self.base_dir / username
        d.mkdir(parents=True, exist_ok=True)
        return d / "conversations.json"

    def load_all(self, username: str) -> dict:
        f = self._file(username)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return {}

    def _save(self, username: str, data: dict):
        self._file(username).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def new_conversation(self, username: str, title: str = "New chat") -> str:
        cid = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat()
        data = self.load_all(username)
        data[cid] = {"title": title, "created": now, "updated": now, "messages": []}
        self._save(username, data)
        return cid

    def append_message(self, username: str, cid: str, message: dict):
        data = self.load_all(username)
        conv = data.get(cid)
        if conv is None:
            return
        conv["messages"].append(message)
        if message.get("role") == "user" and conv["title"] in ("New chat", ""):
            conv["title"] = message["content"][:40]
        conv["updated"] = datetime.now().isoformat()
        self._save(username, data)

    def delete_conversation(self, username: str, cid: str):
        data = self.load_all(username)
        data.pop(cid, None)
        self._save(username, data)