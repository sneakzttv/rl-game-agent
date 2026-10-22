"""Loopback-only educational HTTP API and static frontend (Python stdlib)."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from gridworld import (
    ACTIONS, GOAL, MAX_STEPS, PITS, SIZE, START, STATES, WALLS,
    evaluate, greedy_action, train, transition,
)


STATIC = Path(__file__).resolve().parent / "static"
# The demo has one shared in-memory model. A lock prevents a request observing
# half-written weights; training constructs a new table before publishing it.
MODEL_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()
MODEL = None
MODEL_VERSION = 0


def integer(data, name, default, low, high):
    value = data.get(name, default)
    # bool subclasses int in Python, but JSON true is not a valid episode count.
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return value


def serialize_policy(table):
    return [
        {"state": list(state), "action": greedy_action(table, state),
         "values": table[state]}
        for state in STATES
    ]


class Handler(BaseHTTPRequestHandler):
    server_version = "GridWorldDemo/1.0"

    def setup(self):
        super().setup()
        # Bound body/header reads so stalled local clients cannot wait forever.
        self.connection.settimeout(10)

    def reply(self, status, data, content_type="application/json; charset=utf-8"):
        body = json.dumps(data, allow_nan=False).encode() if isinstance(data, dict) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        # Reject DNS-rebinding hostnames even though the listener is loopback.
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}" or (
            self.headers.get("Host") == f"localhost:{self.server.server_port}"
        )

    def do_GET(self):
        if not self.valid_host():
            self.reply(403, {"error": "Use the loopback URL printed by the server."})
            return
        if self.path == "/api/config":
            self.reply(200, {
                "size": SIZE, "walls": sorted(WALLS), "pits": sorted(PITS),
                "goal": GOAL, "start": START, "actions": ACTIONS,
                "max_steps": MAX_STEPS,
            })
            return
        # Explicit routing avoids directory traversal and accidental source leaks.
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path not in routes:
            self.reply(404, {"error": "Not found."})
            return
        filename, mime = routes[self.path]
        self.reply(200, (STATIC / filename).read_bytes(), mime)

    def do_POST(self):
        if not self.valid_host():
            self.reply(403, {"error": "Invalid Host header."})
            return
        # Only same-origin browser requests may mutate the shared local model.
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers.get('Host')}":
            self.reply(403, {"error": "Cross-origin requests are not allowed."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.reply(415, {"error": "Send application/json."})
            return
        if self.headers.get("Transfer-Encoding"):
            self.reply(400, {"error": "Transfer encoding is not supported."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                raise ValueError("Request body must contain 1 to 4096 bytes.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            self.dispatch(data)
        except (ValueError, UnicodeDecodeError) as exc:
            self.reply(400, {"error": str(exc)})

    def dispatch(self, data):
        global MODEL, MODEL_VERSION
        if self.path == "/api/train":
            episodes = integer(data, "episodes", 3000, 1, 10000)
            seed = integer(data, "seed", 42, 0, 2147483647)
            # Reject overlapping work rather than silently queueing huge jobs.
            if not TRAIN_LOCK.acquire(blocking=False):
                self.reply(409, {"error": "Training is already running. Try again shortly."})
                return
            try:
                table, history = train(episodes=episodes, seed=seed)
                result = {
                    "history": history,
                    "policy": serialize_policy(table),
                    "evaluation": evaluate(table),
                    "baseline": evaluate(),
                    "episodes": episodes, "seed": seed,
                }
                with MODEL_LOCK:
                    MODEL = table
                    MODEL_VERSION += 1
                    result["model_version"] = MODEL_VERSION
            finally:
                TRAIN_LOCK.release()
            self.reply(200, result)
        elif self.path == "/api/evaluate":
            episodes = integer(data, "episodes", 200, 1, 1000)
            seed = integer(data, "seed", 9001, 0, 2147483647)
            with MODEL_LOCK:
                table, version = MODEL, MODEL_VERSION
            if table is None:
                self.reply(409, {"error": "Train a model first."})
                return
            self.reply(200, {
                "evaluation": evaluate(table, episodes=episodes, seed=seed),
                "baseline": evaluate(episodes=episodes, seed=seed),
                "model_version": version,
            })
        elif self.path == "/api/step":
            state = data.get("state")
            if (
                not isinstance(state, list) or len(state) != 2
                or any(type(coord) is not int for coord in state)
                or tuple(state) not in STATES
            ):
                raise ValueError("state must be a safe nonterminal [row, column].")
            steps = integer(data, "steps", 0, 0, MAX_STEPS - 1)
            if data.get("mode", "human") not in ("human", "agent"):
                raise ValueError("mode must be human or agent.")
            if data.get("mode") == "agent":
                with MODEL_LOCK:
                    table, version = MODEL, MODEL_VERSION
                if table is None:
                    self.reply(409, {"error": "Train a model before using agent mode."})
                    return
                if "model_version" in data and data["model_version"] != version:
                    self.reply(409, {"error": "Another tab replaced the model. Train again in this tab."})
                    return
                action = greedy_action(table, tuple(state))
                values = table[tuple(state)]
            else:
                action = integer(data, "action", -1, 0, 3)
                values, version = None, None
            nxt, reward, terminal, event = transition(tuple(state), action)
            truncated = not terminal and steps + 1 >= MAX_STEPS
            self.reply(200, {
                "state": nxt, "reward": reward, "terminal": terminal,
                "truncated": truncated, "event": "timeout" if truncated else event,
                "steps": steps + 1, "action": action, "q_values": values,
                "model_version": version,
            })
        else:
            self.reply(404, {"error": "Not found."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Cannot start server: {exc}. Try --port 8001.\n")
    print(f"GridWorld demo: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GridWorld.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
