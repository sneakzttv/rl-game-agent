"""Exercise the real loopback HTTP server, not a mocked frontend/backend seam."""

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

import server


class QuietHandler(server.Handler):
    def log_message(self, format, *args):
        # Tests report assertions; routine access logs would hide useful failures.
        pass


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        with server.MODEL_LOCK:
            server.MODEL = None
            server.MODEL_VERSION = 0

    def request(self, method, path, data=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=15)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        body = raw if raw is not None else (json.dumps(data) if data is not None else None)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            payload = response.read()
            return response.status, dict(response.getheaders()), payload
        finally:
            connection.close()

    def post(self, path, data):
        status, _, body = self.request("POST", path, data)
        return status, json.loads(body)

    def test_static_and_config(self):
        for path, marker in [
            ("/", b"Small world."), ("/app.js", b"/api/train"),
            ("/style.css", b":root"), ("/api/config", b'"size": 6'),
        ]:
            status, headers, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(marker, body)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        for path in ("/server.py", "/../server.py", "/unknown"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_real_train_evaluate_and_agent_step(self):
        self.assertEqual(self.post("/api/evaluate", {})[0], 409)
        self.assertEqual(self.post("/api/step", {"state": [0, 0], "mode": "agent"})[0], 409)
        status, trained = self.post("/api/train", {"episodes": 3000, "seed": 42})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(trained["evaluation"]["success_rate"], 0.95)
        self.assertEqual(len(trained["policy"]), len(server.STATES))
        status, step = self.post("/api/step", {
            "state": [0, 0], "mode": "agent", "model_version": trained["model_version"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(len(step["q_values"]), 4)
        self.assertEqual(step["steps"], 1)
        status, evaluated = self.post("/api/evaluate", {"episodes": 100, "seed": 12345})
        self.assertEqual(status, 200)
        self.assertEqual(evaluated["evaluation"]["episodes"], 100)
        self.assertEqual(evaluated["model_version"], trained["model_version"])
        self.assertEqual(self.post("/api/step", {
            "state": [0, 0], "mode": "agent", "model_version": -1,
        })[0], 409)

    def test_manual_move_and_timeout(self):
        status, step = self.post("/api/step", {"state": [0, 0], "action": 1})
        self.assertEqual(status, 200)
        self.assertEqual(step["state"], [0, 1])
        self.assertEqual(step["reward"], -0.02)
        self.assertFalse(step["terminal"])
        status, step = self.post("/api/step", {"state": [0, 0], "action": 0, "steps": 59})
        self.assertEqual(status, 200)
        self.assertTrue(step["truncated"])
        self.assertEqual(step["event"], "timeout")

    def test_invalid_inputs_return_clear_client_errors(self):
        for path, data in [
            ("/api/train", {"episodes": True}), ("/api/train", {"seed": -1}),
            ("/api/train", {"episodes": 10001}), ("/api/evaluate", {"episodes": 0}),
            ("/api/step", {"state": [True, 0], "action": 1}),
            ("/api/step", {"state": [1, 1], "action": 1}),
            ("/api/step", {"state": [5, 5], "action": 1}),
            ("/api/step", {"state": [0, 0], "action": 4}),
            ("/api/step", {"state": [0, 0], "action": 1, "steps": 60}),
            ("/api/step", {"state": [0, 0], "mode": "unknown"}),
        ]:
            status, body = self.post(path, data)
            self.assertEqual(status, 400, (path, data))
            self.assertIn("error", body)
        for raw in ("{", "[]", "x" * 4097):
            self.assertEqual(self.request("POST", "/api/train", raw=raw)[0], 400)
        self.assertEqual(self.request("POST", "/api/train", raw="{}",
                                      headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post("/api/unknown", {})[0], 404)

    def test_origin_host_and_overlapping_training_guards(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.invalid"})[0], 403)
        self.assertEqual(self.request("POST", "/api/train", {},
                                      headers={"Origin": "http://evil.invalid"})[0], 403)
        server.TRAIN_LOCK.acquire()
        try:
            self.assertEqual(self.post("/api/train", {})[0], 409)
        finally:
            server.TRAIN_LOCK.release()


if __name__ == "__main__":
    unittest.main()
