"""desktop_web_install.py against a child of this test and a stand-in server on loopback; no app and no markitai."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import threading
import unittest
from unittest import mock

import desktop_web_install
from desktop_web_install import (EXPECTED, PAGE, PAGE_NAME, TOKEN_VARIABLE, Failure, convert, main, multipart,
                                 process_environment, serve_token)

# The systems Markitai Web ships for, where an installed app's server is read.
SHIPPED_HOSTS = {("Darwin", "arm64"), ("Linux", "x86_64"), ("Windows", "AMD64")}
TOKEN = "stand-in-token"


@contextlib.contextmanager
def child(**variables):
    """A process of this user that waits on its standard input, started with `variables` added."""
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                               env=dict(os.environ, **variables))
    try:
        yield process.pid
    finally:
        process.stdin.close()
        process.wait(timeout=30)


class Server:
    """What `markitai serve` answers a conversion with: the job, the history once it finished, the result."""

    def __init__(self, *, created=201, item_status="done", markdown="# Installed app\n\nhello installed\n"):
        self.created, self.item_status, self.markdown = created, item_status, markdown
        self.requests, self.uploads, self.history_calls = [], [], 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def answer(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def allowed(self):
                outer.requests.append((self.command, self.path))
                if self.headers.get("Authorization") == f"Bearer {TOKEN}":
                    return True
                self.answer(401, {"error": {"code": "unauthorized"}})
                return False

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.allowed():
                    outer.uploads.append((self.headers["Content-Type"], body))
                    self.answer(outer.created, {"job_id": "job-1", "items": [{"item_id": "item-1"}]})

            def do_GET(self):
                if not self.allowed():
                    return
                if self.path == "/api/history?include=items":
                    outer.history_calls += 1
                    other = {"job_id": "job-0", "items": [{"item_id": "item-0", "name": PAGE_NAME, "status": "error"}]}
                    mine = {"job_id": "job-1", "items": [{"item_id": "item-1", "name": PAGE_NAME,
                                                          "status": outer.item_status}]}
                    # Running at the first look: the history does not list the job yet.
                    self.answer(200, [other] if outer.history_calls == 1 else [other, mine])
                elif self.path == "/api/jobs/job-1/items/item-1/result":
                    self.answer(200, {"name": PAGE_NAME, "markdown": outer.markdown})
                else:
                    self.answer(404, {"error": {"code": "not_found"}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=30)


class EnvironmentTests(unittest.TestCase):
    def test_strings_end_at_the_empty_one_and_skip_drive_directories(self):
        block = "=C:=C:\\work\0PATH=C:\\bin\0EMPTY=\0NAME=a=b\0\0LEFTOVER=1\0"
        for encoding in ("utf-8", "utf-16-le"):
            self.assertEqual(desktop_web_install._strings(block.encode(encoding), encoding),
                             {"PATH": "C:\\bin", "EMPTY": "", "NAME": "a=b"})

    @unittest.skipUnless((platform.system(), platform.machine()) in SHIPPED_HOSTS,
                         "an installed app's server is read on the systems Markitai Web ships for")
    def test_a_child_s_environment_is_what_it_was_started_with(self):
        with child(**{TOKEN_VARIABLE: TOKEN, "MARKITAI_WEB_INSTALL_TEST": "two words, 中文"}) as pid:
            environment = process_environment(pid)
            self.assertEqual(environment.get(TOKEN_VARIABLE), TOKEN)
            self.assertEqual(environment.get("MARKITAI_WEB_INSTALL_TEST"), "two words, 中文")
            self.assertEqual(serve_token(pid), TOKEN)

    @unittest.skipUnless((platform.system(), platform.machine()) in SHIPPED_HOSTS,
                         "an installed app's server is read on the systems Markitai Web ships for")
    def test_a_process_without_the_token_is_not_the_app_s_server(self):
        environment = {name: value for name, value in os.environ.items() if name != TOKEN_VARIABLE}
        process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                                   env=environment)
        try:
            with self.assertRaisesRegex(Failure, f"has no {TOKEN_VARIABLE}"):
                serve_token(process.pid)
        finally:
            process.stdin.close()
            process.wait(timeout=30)

    def test_a_process_that_cannot_be_read_fails_with_its_number(self):
        with mock.patch.object(desktop_web_install, "process_environment",
                                        side_effect=PermissionError(13, "denied")):
            with self.assertRaisesRegex(Failure, "process 4242's environment cannot be read"):
                serve_token(4242)


class ConvertTests(unittest.TestCase):
    def test_multipart_carries_each_file_under_files(self):
        content_type, body = multipart([("a.html", b"<p>a</p>"), ("b.txt", b"b\r\n")])
        boundary = content_type.removeprefix("multipart/form-data; boundary=")
        self.assertNotEqual(boundary, content_type)
        parts = body.split(f"--{boundary}".encode())
        self.assertEqual(parts[0], b"")
        self.assertEqual(parts[-1], b"--\r\n")
        self.assertEqual(parts[1], b'\r\nContent-Disposition: form-data; name="files"; filename="a.html"\r\n'
                                   b"Content-Type: application/octet-stream\r\n\r\n<p>a</p>\r\n")
        self.assertTrue(parts[2].endswith(b"\r\n\r\nb\r\n\r\n"))

    def test_the_page_goes_up_with_the_token_and_its_markdown_comes_back(self):
        with Server() as server:
            markdown = convert(server.origin, TOKEN, timeout=30)
        self.assertTrue(all(text in markdown for text in EXPECTED))
        (content_type, body), = server.uploads
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(f'filename="{PAGE_NAME}"'.encode(), body)
        self.assertIn(PAGE, body)
        self.assertGreaterEqual(server.history_calls, 2)
        self.assertEqual(server.requests[0], ("POST", "/api/jobs"))
        self.assertEqual(server.requests[-1], ("GET", "/api/jobs/job-1/items/item-1/result"))

    def test_another_token_is_refused(self):
        with Server() as server:
            with self.assertRaisesRegex(Failure, "POST /api/jobs answered 401"):
                convert(server.origin, "another", timeout=30)

    def test_an_item_that_failed_fails(self):
        with Server(item_status="error") as server:
            with self.assertRaisesRegex(Failure, f"{PAGE_NAME} failed to convert"):
                convert(server.origin, TOKEN, timeout=30)

    def test_markdown_without_the_page_s_text_fails(self):
        with Server(markdown="# Something else\n") as server:
            with self.assertRaisesRegex(Failure, "Markdown is not the page's"):
                convert(server.origin, TOKEN, timeout=30)

    def test_a_job_that_never_finishes_fails_at_the_limit(self):
        with Server(item_status="running") as server:
            with self.assertRaisesRegex(Failure, "job job-1 did not finish in 1 s"):
                convert(server.origin, TOKEN, timeout=1)

    def test_no_server_fails_with_the_address(self):
        with Server() as server:
            origin = server.origin
        with self.assertRaisesRegex(Failure, origin):
            convert(origin, TOKEN, timeout=5)


class MainTests(unittest.TestCase):
    def run_main(self, pid, origin):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["convert", "--pid", str(pid), "--origin", origin + "/"])
        return code, output.getvalue()

    def test_convert_says_what_the_server_made_of_the_page(self):
        with Server() as server, mock.patch.object(desktop_web_install, "process_environment",
                                                            return_value={TOKEN_VARIABLE: TOKEN}):
            code, output = self.run_main(77, server.origin)
        self.assertEqual(code, 0)
        self.assertEqual(output, f"process 77 at {server.origin}/ converted {PAGE_NAME}: "
                                 "# Installed app / hello installed\n")

    def test_a_failure_is_one_line_on_standard_output_and_exit_1(self):
        with Server() as server, mock.patch.object(desktop_web_install, "process_environment",
                                                            return_value={}):
            code, output = self.run_main(77, server.origin)
        self.assertEqual(code, 1)
        self.assertEqual(output, f"FAIL process 77 has no {TOKEN_VARIABLE}: it is not a server Markitai Web started\n")
        self.assertEqual(server.requests, [])


if __name__ == "__main__":
    unittest.main()
