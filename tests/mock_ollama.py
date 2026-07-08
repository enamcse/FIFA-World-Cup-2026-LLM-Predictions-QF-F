#!/usr/bin/env python3
"""Minimal fake Ollama server for smoke-testing the pipeline without a GPU.

Implements /api/version, /api/show, /api/chat. Scores are deterministic
functions of (model, match teams, seed), so the whole pipeline - including
aggregation and scoring - is reproducible in tests.

    python3 tests/mock_ollama.py 12435
"""

import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep test output clean

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0.0-mock"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/show":
            self._send({"details": {"family": "mock", "parameter_size": "0B"},
                        "parameters": "mock"})
            return
        if self.path != "/api/chat":
            self._send({"error": "not found"}, 404)
            return

        user_msg = next(m["content"] for m in req["messages"] if m["role"] == "user")
        m = re.search(r"Match: (.+) vs (.+)", user_msg)
        home, away = m.group(1).strip(), m.group(2).strip()
        seed = req.get("options", {}).get("seed", 0)
        temperature = req.get("options", {}).get("temperature", 0)
        # greedy (temperature 0) is seed-independent, like a real model
        key = f"{req['model']}|{home}|{away}|{seed if temperature > 0 else 'greedy'}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        home_goals, away_goals = h % 4, (h // 7) % 3
        advances = home if (home_goals, away_goals) >= (away_goals, h % 2) else away
        content = json.dumps({
            "home_goals": home_goals,
            "away_goals": away_goals,
            "advances": advances,
            "one_line_reasoning": f"mock deterministic pick for seed {seed}",
        })
        self._send({
            "model": req["model"],
            "message": {"role": "assistant", "content": content},
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 40,
            "total_duration": 123456789,
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12435
    print(f"mock ollama on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
