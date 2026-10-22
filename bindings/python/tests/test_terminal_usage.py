"""Host error compatibility and real loopback native terminal accounting."""
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import markitai
from markitai.api import _result


class TerminalUsageTests(unittest.TestCase):
    def test_old_errors_keep_categories_and_optional_accounting(self):
        classes = {
            "fetch_error": markitai.FetchError,
            "no_model_configured": markitai.NoModelConfiguredError,
            "invalid_input": ValueError, "invalid_json": ValueError,
            # Python markitai's name; still a ValueError.
            "config_error": markitai.ConfigFileError, "not_found": FileNotFoundError,
            "is_directory": IsADirectoryError, "io_error": OSError,
            "conversion_error": markitai.ConversionError,
            "unsupported": markitai.ConversionError,
        }
        old = markitai.ConversionError("old message", code="old_code")
        self.assertEqual((str(old), old.code), ("old message", "old_code"))
        self.assertIsNone(old.usage)
        paid_zero = {"cost_usd": 0.0, "requests": 1, "input_tokens": 0,
                     "output_tokens": 0, "by_model": {"fixture": {"requests": 1}}}
        for code, kind in classes.items():
            for supplied in (None, paid_zero):
                with self.subTest(code=code, recorded=supplied is not None):
                    error = {"code": code, "message": "unchanged message"}
                    if supplied is not None:
                        error["usage"] = supplied
                    with self.assertRaises(kind) as raised:
                        _result(json.dumps({"ok": False, "error": error}))
                    self.assertIs(type(raised.exception), kind)
                    self.assertEqual(str(raised.exception), "unchanged message")
                    if supplied is None:
                        self.assertIsNone(raised.exception.usage)
                    else:
                        self.assertIsInstance(raised.exception.usage, markitai.ConversionUsage)
                        self.assertEqual(raised.exception.usage.requests, 1)
                        self.assertEqual(raised.exception.usage.input_tokens, 0)
                        self.assertEqual(raised.exception.usage.by_model, supplied["by_model"])

    def test_paid_native_failures_are_isolated_between_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name in ("TERMALPHA", "TERMBETA", "TERMZERO"):
                files[name] = root / f"{name}.md"
                files[name].write_text(f"# {name}\n\nComplete independent source document {name}.\n")
            requests, errors = [], []
            lock = threading.Lock()
            barrier = threading.Barrier(2, timeout=10)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_POST(self):
                    self.connection.settimeout(15)
                    try:
                        length = int(self.headers["Content-Length"])
                        assert 0 < length < 1024 * 1024
                        request = json.loads(self.rfile.read(length))
                        text = json.dumps(request["messages"], ensure_ascii=False)
                        name = next(name for name in files if name in text)
                        with lock:
                            requests.append(name)
                        if name != "TERMZERO":
                            barrier.wait()
                        input_tokens, output_tokens = {"TERMALPHA": (11, 3), "TERMBETA": (29, 7), "TERMZERO": (0, 0)}[name]
                        # The refusal echoes the key; its message is shown without it.
                        payload = {"error": {"message": "Incorrect API key provided: synthetic-key"}, "model": name,
                                   "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}
                        raw = json.dumps(payload).encode()
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                    except Exception as error:
                        with lock:
                            errors.append(repr(error))
                        self.send_error(500)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            config = {"cache": {"enabled": False, "global_dir": str(root / "cache")},
                      "prompts": {"dir": str(root / "prompts")}, "history": {"record": False},
                      "ocr": {"enabled": False}, "image": {"alt_enabled": False, "desc_enabled": False},
                      "llm": {"enabled": True, "on_failure": "fail", "max_requests_per_document": 1,
                              "router_settings": {"num_retries": 0, "timeout": 15},
                              "model_list": [{"model_name": "fixture", "litellm_params": {
                                  "model": "openai/fixture", "api_key": "synthetic-key",
                                  "api_base": f"http://127.0.0.1:{server.server_port}/v1"}}]}}
            previous = Path.cwd()
            environment = {key: os.environ[key] for key in ("HOME", "PATH", "SYSTEMROOT", "WINDIR", "USERPROFILE") if key in os.environ}
            environment.update(MARKITAI_HOME=str(root / "state"), NO_PROXY="127.0.0.1,localhost", PYTHON_DOTENV_DISABLED="1")
            try:
                os.chdir(root)
                with patch.dict(os.environ, environment, clear=True):
                    def failure(name):
                        try:
                            markitai.convert(files[name], config=config)
                        except markitai.ConversionError as error:
                            return error
                        self.fail("paid native failure unexpectedly succeeded")

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        alpha, beta = list(pool.map(failure, ("TERMALPHA", "TERMBETA")))
                    zero = failure("TERMZERO")
                    for name, error, expected in (("TERMALPHA", alpha, (11, 3)), ("TERMBETA", beta, (29, 7)), ("TERMZERO", zero, (0, 0))):
                        self.assertEqual(error.code, "conversion_error")
                        self.assertIn("HTTP 401", str(error))
                        self.assertNotIn("synthetic-key", str(error))
                        self.assertIsInstance(error.usage, markitai.ConversionUsage)
                        self.assertEqual((error.usage.requests, error.usage.input_tokens, error.usage.output_tokens), (1, *expected))
                        self.assertEqual(set(error.usage.by_model), {name})
                        coverage = error.usage.by_model[name]
                        self.assertEqual((coverage["priced_requests"], coverage["unpriced_requests"], coverage["cost_status"]), (0, 1, "unknown"))
                        self.assertNotIn("pricing_snapshot", coverage)
                        self.assertEqual(error.usage.cost_usd, 0)
                    with self.assertRaises(FileNotFoundError) as missing:
                        markitai.convert(root / "missing.md", config={}, llm=False)
                    self.assertIsNone(missing.exception.usage)
            finally:
                os.chdir(previous)
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertCountEqual(requests, list(files))


if __name__ == "__main__":
    unittest.main()
