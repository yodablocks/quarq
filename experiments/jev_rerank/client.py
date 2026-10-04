"""Tiny Jev HTTP client with an on-disk cache and per-call latency. Throwaway."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import requests

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
CACHE = Path(__file__).resolve().parent / ".cache" / "jev.sqlite"


class JevClient:
    def __init__(self) -> None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(CACHE, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS r (key TEXT PRIMARY KEY, response TEXT, latency_ms REAL)"
        )
        self._lock = threading.Lock()
        self.latencies_ms: list[float] = []
        self.input_tokens = 0
        self.fresh_calls = 0
        self.cache_hits = 0

    def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        body = {"state": state, "model": MODEL, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT response FROM r WHERE key=?", (key,)).fetchone()
            if row:
                self.cache_hits += 1
                return json.loads(row[0])
        headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
        for attempt in range(6):
            started = time.time()
            resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=90)
            ms = (time.time() - started) * 1000
            if resp.status_code == 200:
                payload = resp.json()
                with self._lock:
                    self._db.execute("INSERT OR REPLACE INTO r VALUES (?,?,?)",
                                     (key, json.dumps(payload, ensure_ascii=False), ms))
                    self._db.commit()
                    self.latencies_ms.append(ms)
                    self.fresh_calls += 1
                    self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
                return payload
            if resp.status_code in (429, 529) or resp.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("Jev: gave up after 6 attempts")

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN
