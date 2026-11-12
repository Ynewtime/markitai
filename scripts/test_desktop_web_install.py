"""desktop_web_install.py against a child of this test, a stand-in server on loopback and a stand-in app; no
Markitai Web and no markitai."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import contextlib
import io
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zlib
from unittest import mock

import desktop_web_install
from desktop_web_install import (APP_ID, EXPECTED, PAGE, PAGE_NAME, SLUG, TOKEN_VARIABLE, Failure, Installed, Report,
                                 alive, command, convert, exercise, listening, main, multipart, png,
                                 process_command, process_environment, run_archive, run_darwin, run_debian,
                                 serve_token, started_by, user_environment, wait_for)

# The systems Markitai Web ships for, where an installed app's server is read.
SHIPPED_HOSTS = {("Darwin", "arm64"), ("Linux", "x86_64"), ("Windows", "AMD64")}
TOKEN = "stand-in-token"
# Where `run` installs, and so where a process's command and windows are read.
INSTALLS = sys.platform in ("darwin", "linux")
# What an installed app does that the check sees, as a script: its server is a process named
# markitai that holds the token and waits on standard input, and serve.log names the stand-in
# server of this test. `mode` leaves one of them out.
STAND_IN_APP = r"""
import os, signal, subprocess, sys, time
from pathlib import Path

bundled, waits, origin, token, mode = sys.argv[1:6]
home = Path(os.environ["HOME"])
if mode == "ends early":
    sys.exit(7)
(home / "logs").mkdir()
(home / "data").mkdir()
server = subprocess.Popen([bundled, "-c", waits, "serve", "--supervised"],
                          stdin=subprocess.PIPE, env=dict(os.environ, MARKITAI_SERVE_TOKEN=token))
(home / "logs/serve.log").write_text(f"Markitai server listening on {origin}\n", encoding="utf-8")
if mode != "keeps no port":
    (home / "data/web.json").write_text("{}", encoding="utf-8")


def quit(*_):
    if mode != "quits badly":
        with open(home / "logs/serve.log", "a", encoding="utf-8") as lines:
            lines.write("Server stopping: draining active conversions and saving history.\n")
        server.stdin.close()
        server.wait()
    sys.exit(3 if mode == "quits badly" else 0)


signal.signal(signal.SIGTERM, quit)
while True:
    time.sleep(0.05)
"""


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


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(INSTALLS, "a process's command is read where `run` installs")
    def test_a_child_s_command_is_what_it_was_started_with(self):
        with child() as pid:
            program, arguments = process_command(pid)
            self.assertTrue(os.path.isabs(program) and os.access(program, os.X_OK), program)
            self.assertEqual(arguments[1:], ["-c", "import sys; sys.stdin.read()"])
            # The arguments alone: a framework's Python starts as a launcher that then becomes the interpreter.
            self.assertEqual([found[2][1:] for found in started_by(os.getpid()) if found[0] == pid], [arguments[1:]])
            self.assertTrue(alive(pid))
        self.assertFalse(alive(pid))
        self.assertEqual([found for found in started_by(os.getpid()) if found[0] == pid], [])

    @unittest.skipUnless(INSTALLS, "a process's command is read where `run` installs")
    def test_a_child_is_read_while_a_launcher_becomes_its_program(self):
        # A framework's python3 is a launcher: for a moment as it becomes the interpreter, a few
        # milliseconds after its start, the system refuses to read the process.
        for _ in range(5):
            with child() as pid:
                until = time.monotonic() + 0.05
                while time.monotonic() < until:
                    self.assertEqual(process_command(pid)[1][1:], ["-c", "import sys; sys.stdin.read()"])

    @unittest.skipUnless(INSTALLS, "a process's windows are read where `run` installs")
    def test_a_process_without_windows_has_none_on_the_screen(self):
        windows = (desktop_web_install._darwin_windows if sys.platform == "darwin"
                   else desktop_web_install._x11_windows)
        try:
            sizes = windows(os.getpid())
        except OSError as error:
            self.skipTest(f"no screen here: {error}")
        self.assertEqual(sizes, [])

    def test_wait_for_ends_at_the_first_true_value_the_limit_or_the_process_s_end(self):
        self.assertEqual(wait_for(lambda: "found", 30), "found")
        started = time.monotonic()
        self.assertIsNone(wait_for(lambda: "", 0.3))
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=30)
        self.assertIsNone(wait_for(lambda: None, 30, process))
        self.assertLess(time.monotonic() - started, 10)

    def test_a_command_that_fails_is_a_failure_with_what_it_printed(self):
        self.assertEqual(command([sys.executable, "-c", "print('out')"]), "out\n")
        with self.assertRaisesRegex(Failure, "exited 3: why"):
            command([sys.executable, "-c", "import sys; print('why', file=sys.stderr); sys.exit(3)"])
        with self.assertRaisesRegex(Failure, "^no-such-program-of-the-install-check: "):
            command(["no-such-program-of-the-install-check"])
        with self.assertRaisesRegex(Failure, "timed out after 0.2 seconds"):
            command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.2)


class RunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # Its real path, as the check's own state has.
        self.work = Path(directory.name).resolve()

    def test_serve_log_names_the_origin_once_the_server_listens(self):
        log = self.work / "serve.log"
        self.assertIsNone(listening(log))
        log.write_text("  • Jobs and history: /state/serve/jobs\n", encoding="utf-8")
        self.assertIsNone(listening(log))
        log.write_text("◆ Markitai server listening on http://127.0.0.1:23600\n\n  • Jobs\n", encoding="utf-8")
        self.assertEqual(listening(log), "http://127.0.0.1:23600")

    def test_the_app_gets_a_desktop_s_environment_with_the_check_s_home(self):
        ours = {"MARKITAI_CONFIG": "/developer/config.toml", "MARKITAI_HOME": "/developer/state",
                "MYGO_AUTOMATION": "a.sock", "TERM": "xterm", "TERM_PROGRAM": "a terminal", "ZDOTDIR": "/developer/zsh",
                "SHELL": "/bin/fish", "HOME": "/developer", "PATH": "/usr/bin:/bin",
                # A runner's, which would take the app's settings out of the check's HOME, and a session's.
                "XDG_CONFIG_HOME": "/developer/.config", "XDG_DATA_HOME": "/developer/.local/share",
                "XDG_CACHE_HOME": "/developer/.cache", "XDG_STATE_HOME": "/developer/.local/state",
                "WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1001"}
        with mock.patch.dict(os.environ, ours, clear=True):
            environment = user_environment(self.work / "home", self.work / "state", "/bin/zsh")
        self.assertEqual(environment, {"HOME": str(self.work / "home"), "MARKITAI_HOME": str(self.work / "state"),
                                       "MARKITAI_LANG": "en", "SHELL": "/bin/zsh", "PATH": "/usr/bin:/bin",
                                       "DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1001"})

    def test_a_report_counts_what_failed_and_keeps_its_lines(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = Report(self.work / "evidence")
            self.assertTrue(report.check("http://127.0.0.1:1", "one holds"))
            self.assertFalse(report.check(None, "one does not"))
            report.log("a note")
            report.close()
        self.assertEqual(report.failures, 1)
        self.assertEqual(output.getvalue(), "ok   one holds\nFAIL one does not\na note\n")
        self.assertEqual((self.work / "evidence/install.log").read_text(encoding="utf-8"), output.getvalue())

    def test_an_app_that_is_already_there_is_left_alone(self):
        (self.work / "Markitai Web.app").mkdir()
        (self.work / "Markitai Web.app/kept").write_text("a user's", encoding="utf-8")
        with mock.patch.object(desktop_web_install, "command") as ran:
            with self.assertRaisesRegex(Failure, "Markitai Web.app is already there"):
                run_darwin(self.work / "image.dmg", Report(), applications=self.work)
        ran.assert_not_called()
        self.assertEqual((self.work / "Markitai Web.app/kept").read_text(encoding="utf-8"), "a user's")

    def test_an_app_that_already_runs_stops_the_check_before_anything_is_installed(self):
        listed = "    1 /sbin/launchd\n  359 /Volumes/Elsewhere/Markitai Web.app/Contents/MacOS/Markitai Web\n"
        with mock.patch.object(desktop_web_install, "command", return_value=listed) as ran:
            with self.assertRaisesRegex(Failure, r"Markitai Web already runs \(process 359\)"):
                run_darwin(self.work / "image.dmg", Report(), applications=self.work)
        ran.assert_called_once_with(["ps", "-A", "-o", "pid=,comm="])
        self.assertEqual(list(self.work.iterdir()), [])

    def stand_in(self, origin, mode="", windows=((800, 600),)):
        """An installed app that is STAND_IN_APP, and its report after exercise() ran it."""
        folder = self.work / "app"
        (folder / "bin").mkdir(parents=True)
        bundled = folder / "bin/markitai"
        if sys.platform == "darwin":
            # This test's own Python under the bundled server's name: a link, which a process keeps as its
            # program's path. macOS shows no one the environment of a program of its own, such as a shell.
            bundled.symlink_to(process_command(os.getpid())[0])
            waits = "import sys; sys.stdin.read()"
        else:
            # Linux names a process's program by its file, whatever link started it: a copy of the shell.
            shutil.copy("/bin/sh", bundled)
            waits = "read line"
        (folder / "app.py").write_text(STAND_IN_APP, encoding="utf-8")
        program = folder / "Markitai Web"
        program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{folder / "app.py"}" "{bundled}" "{waits}" '
                           f'"{origin}" "{TOKEN}" "{mode}"\n', encoding="utf-8")
        program.chmod(0o755)

        def screenshot(path):
            path.write_bytes(b"a screen")

        def quit(process):
            process.send_signal(signal.SIGTERM)
            return "a signal"

        app = Installed(program, bundled if mode != "starts another markitai" else folder / "markitai",
                        Path("logs"), Path("data"), "/bin/sh", lambda pid: list(windows), screenshot, quit)
        state = self.work / "state"
        state.mkdir()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.multiple(
                desktop_web_install, START_SECONDS=30, WINDOW_SECONDS=0.5, QUIT_SECONDS=20, PAINT_SECONDS=0):
            report = Report(self.work / "evidence")
            exercise(app, report, state, self.work / "evidence")
            report.close()
        lines = output.getvalue().splitlines()
        return report, lines, [line for line in lines if line.startswith("FAIL ")]

    @unittest.skipUnless(INSTALLS, "`run` installs on macOS and Linux")
    def test_an_app_that_serves_shows_a_window_converts_and_quits_passes_every_check(self):
        with Server() as server:
            report, lines, failed = self.stand_in(server.origin)
        self.assertEqual((report.failures, failed), (0, []), lines)
        self.assertEqual([line.split(" (")[0] for line in lines if line.startswith("ok   ")], [
            "ok   the installed app's serve listens", "ok   its child is the bundled markitai serve --supervised",
            "ok   the app shows a window", "ok   the port kept in data/web.json in HOME",
            "ok   the installed app's serve converts a file", "ok   the app quits with 0", "ok   its serve has ended",
            "ok   serve drained"])
        self.assertIn("ok   the app shows a window (800x600)", lines)
        self.assertIn("screen: installed-ready.png", lines)
        self.assertEqual(server.requests[0], ("POST", "/api/jobs"))
        evidence = self.work / "evidence"
        self.assertEqual(sorted(path.name for path in evidence.iterdir()),
                         ["install.log", "installed-app.log", "installed-ready.png", "installed-serve.log"])
        self.assertEqual(evidence.joinpath("install.log").read_text(encoding="utf-8").splitlines(), lines)
        self.assertIn("draining active conversions", evidence.joinpath("installed-serve.log").read_text("utf-8"))
        # The app's HOME is the check's own, and nothing of it still runs.
        self.assertTrue((self.work / "state" / "home" / "logs" / "serve.log").is_file())
        serve = next(line for line in lines if line.startswith("serve: process "))
        self.assertFalse(alive(int(serve.split()[2].rstrip(","))))

    @unittest.skipUnless(INSTALLS, "`run` installs on macOS and Linux")
    def test_an_app_that_ends_before_its_server_listens_fails_with_its_status(self):
        report, lines, failed = self.stand_in("http://127.0.0.1:9", "ends early")
        self.assertEqual(failed, ["FAIL the installed app's serve listens (logs/serve.log in HOME)"], lines)
        self.assertEqual(lines[-1], "the app exited 7")
        self.assertEqual(report.failures, 1)

    @unittest.skipUnless(INSTALLS, "`run` installs on macOS and Linux")
    def test_an_app_without_a_window_or_its_port_that_quits_badly_fails_each_of_them(self):
        with Server() as server:
            report, lines, failed = self.stand_in(server.origin, "quits badly", windows=())
        self.assertEqual(failed, ["FAIL the app shows a window (none)", "FAIL the app quits with 0 (3)",
                                  "FAIL serve drained"], lines)
        with Server() as server:
            shutil.rmtree(self.work / "app")
            shutil.rmtree(self.work / "state")
            report, lines, failed = self.stand_in(server.origin, "keeps no port")
        self.assertEqual(failed, ["FAIL the port kept in data/web.json in HOME"], lines)

    @unittest.skipUnless(INSTALLS, "`run` installs on macOS and Linux")
    def test_a_server_that_is_not_the_bundled_one_is_not_asked_to_convert(self):
        with Server() as server:
            report, lines, failed = self.stand_in(server.origin, "starts another markitai")
        self.assertEqual(failed, ["FAIL its child is the bundled markitai serve --supervised"], lines)
        self.assertEqual(server.requests, [])
        self.assertFalse(any("converts a file" in line for line in lines), lines)


class Reported:
    """A report of what a run printed, for a test to read."""

    def __init__(self):
        self.output = io.StringIO()
        with contextlib.redirect_stdout(self.output):
            self.report = Report()

    def __enter__(self):
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        return self.report

    def __exit__(self, *raised):
        self.redirect.__exit__(*raised)

    @property
    def lines(self):
        return self.output.getvalue().splitlines()


class DebianTests(unittest.TestCase):
    """run_debian against a root of this test's own, where a stand-in apt-get installs and removes."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.package = self.root / f"{SLUG}_9.8.7_amd64.deb"
        self.package.write_bytes(b"!<arch>\n")
        self.folder, self.link = self.root / "opt" / SLUG, self.root / "usr/bin" / SLUG
        self.entry = self.root / "usr/share/applications" / f"{APP_ID}.desktop"
        self.status, self.rooted, self.exercised = "unknown", [], []

    def system(self, install="whole"):
        """dpkg, ps and apt-get of a system where the package installs as `install` says."""
        def command(arguments, *_, **__):
            arguments = [str(argument) for argument in arguments]
            if arguments[:2] == ["dpkg-query", "-W"] and arguments[-1] == SLUG:
                if self.status == "unknown":
                    raise Failure(f"dpkg-query: no packages found matching {SLUG}")
                return self.status
            if arguments[:2] == ["dpkg-query", "-W"]:
                return "2.52.6-0ubuntu0.24.04.1"
            if arguments[0] == "ps":
                return "    1 systemd\n  812 Xvfb\n"
            if arguments[:2] == ["dpkg-deb", "-f"]:
                return "libgtk-3-0 | libgtk-3-0t64, libwebkit2gtk-4.1-0\n"
            raise AssertionError(arguments)

        def as_root(arguments, timeout=900):
            arguments = [str(argument) for argument in arguments]
            self.assertEqual(arguments[:3], ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get"])
            self.rooted.append(arguments[3:])
            if arguments[3] == "remove":
                shutil.rmtree(self.folder, ignore_errors=True)
                for path in (self.link, self.entry):
                    if os.path.lexists(path):
                        path.unlink()
                self.status = "not-installed"
                return ""
            if install == "fails":
                self.status = "half-installed"
                raise Failure("apt-get install exited 100: dpkg: error processing archive")
            for folder in (self.folder / "bin", self.folder / "licenses", self.link.parent, self.entry.parent):
                folder.mkdir(parents=True)
            for program in (self.folder / SLUG, self.folder / "bin/markitai"):
                program.write_text("#!/bin/sh\n", encoding="utf-8")
                program.chmod(0o755)
            if install != "without its command":
                self.link.symlink_to(f"../../opt/{SLUG}/{SLUG}")
            self.entry.write_text(f"[Desktop Entry]\nExec={self.folder / SLUG}\n", encoding="utf-8")
            self.status = "installed"
            return ""

        def exercised(app, report, work, evidence=None):
            self.exercised.append((app, work.is_dir()))

        return mock.patch.multiple(desktop_web_install, command=command, as_root=as_root, exercise=exercised)

    def test_apt_get_installs_the_package_the_app_is_exercised_and_apt_get_removes_it(self):
        reported = Reported()
        with self.system(), reported as report:
            run_debian(self.package, report, root=self.root)
        self.assertEqual(reported.lines, [
            "depends: libgtk-3-0 | libgtk-3-0t64, libwebkit2gtk-4.1-0", "WebKitGTK: 2.52.6-0ubuntu0.24.04.1",
            f"ok   apt-get installs the app in {self.folder} with its markitai and licences, the command "
            f"{self.link} and the entry {APP_ID}.desktop",
            "ok   apt-get remove removes the app, its command and its entry"])
        self.assertEqual(self.rooted, [["install", "-y", "--no-install-recommends", str(self.package)],
                                       ["remove", "-y", SLUG]])
        (app, work_was_there), = self.exercised
        self.assertTrue(work_was_there)
        self.assertEqual((app.program, app.markitai, app.shell), (self.link, self.folder / "bin/markitai", "/bin/bash"))
        self.assertEqual((app.logs, app.data), (Path(".config") / APP_ID / "logs", Path(".config") / APP_ID))
        self.assertEqual(report.failures, 0)

    def test_a_package_that_is_installed_already_is_left_alone(self):
        self.status = "installed"
        with self.system(), self.assertRaisesRegex(Failure, f"{SLUG} is installed already"):
            run_debian(self.package, Report(), root=self.root)
        self.assertEqual((self.rooted, self.exercised), ([], []))

    def test_a_package_apt_get_cannot_install_is_removed_as_far_as_it_went(self):
        reported = Reported()
        with self.system("fails"), reported as report:
            with self.assertRaisesRegex(Failure, "apt-get install exited 100"):
                run_debian(self.package, report, root=self.root)
        self.assertEqual([arguments[0] for arguments in self.rooted], ["install", "remove"])
        self.assertEqual(self.exercised, [])
        self.assertEqual(reported.lines[-1], "ok   apt-get remove removes the app, its command and its entry")

    def test_a_package_without_its_command_fails_and_the_app_is_not_started(self):
        reported = Reported()
        with self.system("without its command"), reported as report:
            run_debian(self.package, report, root=self.root)
        self.assertEqual([line for line in reported.lines if line.startswith("FAIL ")], [
            f"FAIL apt-get installs the app in {self.folder} with its markitai and licences, the command "
            f"{self.link} and the entry {APP_ID}.desktop"])
        self.assertEqual((self.exercised, [arguments[0] for arguments in self.rooted]), ([], ["install", "remove"]))
        self.assertFalse(self.folder.exists())


# install.sh in small: the archive beside it into ~/.local, its command and its entry; --uninstall
# removes them, and what %s names besides.
INSTALL_SH = r"""#!/bin/sh
set -eu
app="$HOME/.local/markitai-web.app"
entry="$HOME/.local/share/applications/dev.markitai.web.desktop"
if [ "${1:-}" = --uninstall ]; then
	rm -rf "$app" "$HOME/.local/bin/markitai-web" "$entry" %s
	echo "Uninstalled Markitai Web. Its settings and data are still in place."
	exit 0
fi
echo "Installing Markitai Web in $app"
mkdir -p "$app" "$HOME/.local/bin" "$HOME/.local/share/applications"
tar -xzf "$(dirname -- "$0")/markitai-web-9.8.7-linux-amd64.tar.gz" -C "$app"
ln -sf "$app/markitai-web" "$HOME/.local/bin/markitai-web"
printf 'Exec="%%s"\n' "$app/markitai-web" >"$entry"
"""


@unittest.skipIf(os.name == "nt" or not shutil.which("sh"), "install.sh runs on Linux; this needs a POSIX sh")
class ArchiveTests(unittest.TestCase):
    """run_archive against an install.sh in small, with the app's run stood in for."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.build = Path(directory.name).resolve()
        self.package = self.build / "markitai-web-9.8.7-linux-amd64.tar.gz"
        with tarfile.open(self.package, "w:gz") as archive:
            for name, mode in (("markitai-web", 0o755), ("bin/markitai", 0o755), ("licenses/markitai/LICENSE", 0o644)):
                member = tarfile.TarInfo(name)
                member.mode, member.size = mode, 3
                archive.addfile(member, io.BytesIO(b"MIT"))
        self.works = []

    def run_archive(self, also_removes=""):
        (self.build / "install.sh").write_text(INSTALL_SH % also_removes, encoding="utf-8")

        def exercised(app, report, work, evidence=None):
            # What the app leaves in HOME once it ran.
            home = work / "home"
            (home / app.logs).mkdir(parents=True)
            (home / app.logs / "serve.log").write_text("Markitai server listening\n", encoding="utf-8")
            (home / app.data / "web.json").write_text("{}", encoding="utf-8")
            self.works.append((work, app, os.path.realpath(app.program), (home / ".local/bin" / SLUG).is_symlink()))

        reported = Reported()
        with mock.patch.multiple(desktop_web_install, exercise=exercised, processes_named=lambda name: []), \
                reported as report:
            run_archive(self.package, report)
        return report, reported.lines

    def test_install_sh_installs_in_the_check_s_home_and_uninstall_leaves_the_settings(self):
        report, lines = self.run_archive()
        self.assertEqual(lines, [
            f"install.sh: Installing Markitai Web in ~/.local/{SLUG}.app",
            f"ok   install.sh installs the app in ~/.local/{SLUG}.app with its markitai and licences, the command "
            f"~/.local/bin/{SLUG} and the entry {APP_ID}.desktop",
            "install.sh --uninstall: Uninstalled Markitai Web. Its settings and data are still in place.",
            "ok   install.sh --uninstall removes the app, its command and its entry",
            f"ok   and leaves the app's settings and its server's log (.config/{APP_ID} in HOME)"])
        (work, app, program, linked), = self.works
        self.assertEqual((program, linked), (str(work / "home/.local" / f"{SLUG}.app" / SLUG), True))
        self.assertEqual(app.markitai, work / "home/.local" / f"{SLUG}.app" / "bin/markitai")
        # The HOME was the check's own, and went with it.
        self.assertNotEqual(work / "home", Path.home())
        self.assertFalse(work.exists())
        self.assertEqual(report.failures, 0)

    def test_an_uninstall_that_takes_the_settings_fails(self):
        report, lines = self.run_archive('"$HOME/.config"')
        self.assertEqual([line for line in lines if line.startswith("FAIL ")],
                         [f"FAIL and leaves the app's settings and its server's log (.config/{APP_ID} in HOME)"])

    def test_an_archive_without_install_sh_beside_it_is_not_installed(self):
        with self.assertRaisesRegex(Failure, "no install.sh beside markitai-web-9.8.7-linux-amd64.tar.gz"):
            run_archive(self.package, Report())


class ScreenTests(unittest.TestCase):
    def test_a_png_holds_each_row_of_the_screen(self):
        rows = [bytes([255, 0, 0, 0, 255, 0]), bytes([0, 0, 255, 255, 255, 255])]
        data = png(2, 2, rows)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        chunks, at = {}, 8
        while at < len(data):
            size, kind = struct.unpack(">I4s", data[at:at + 8])
            body = data[at + 8:at + 8 + size]
            self.assertEqual(struct.unpack(">I", data[at + 8 + size:at + 12 + size])[0], zlib.crc32(kind + body))
            chunks[kind] = body
            at += 12 + size
        self.assertEqual(list(chunks), [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(struct.unpack(">IIBBBBB", chunks[b"IHDR"]), (2, 2, 8, 2, 0, 0, 0))
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), b"\0" + rows[0] + b"\0" + rows[1])


class MainTests(unittest.TestCase):
    def run_main(self, pid, origin):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["convert", "--pid", str(pid), "--origin", origin + "/"])
        return code, output.getvalue()

    def test_run_is_for_the_systems_it_installs_on(self):
        with mock.patch.object(sys, "platform", "win32"):
            with self.assertRaisesRegex(SystemExit, "run installs the disk image on macOS"):
                main(["run", "--package", "markitai-web-setup.exe"])

    def test_a_missing_package_fails_the_run(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as work, contextlib.redirect_stdout(output), \
                mock.patch.object(sys, "platform", "darwin"):
            missing = Path(work).resolve() / "none.dmg"
            code = main(["run", "--package", str(missing)])
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), f"FAIL no package at {missing}\n1 check(s) failed\n")

    def test_on_linux_the_app_needs_a_display_and_a_package_of_a_kind_it_installs(self):
        with tempfile.TemporaryDirectory() as work, mock.patch.object(sys, "platform", "linux"):
            package = Path(work).resolve() / "markitai-web.rpm"
            package.write_bytes(b"rpm")
            for display, said in ((None, "FAIL the app needs an X display and DISPLAY names none"),
                                  (":99", "FAIL markitai-web.rpm is neither the Debian package nor the .tar.gz")):
                environment = {name: value for name, value in os.environ.items() if name != "DISPLAY"}
                if display:
                    environment["DISPLAY"] = display
                output = io.StringIO()
                with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(output):
                    code = main(["run", "--package", str(package)])
                lines = output.getvalue().splitlines()
                self.assertEqual(code, 1)
                self.assertTrue(lines[0].startswith("package: markitai-web.rpm, 3 bytes, sha256 "), lines)
                self.assertRegex(lines[1], rf"^system: .+, X display {display or 'none'}$")
                self.assertTrue(lines[2].startswith(said), lines)
                self.assertEqual(lines[3:], ["1 check(s) failed"])

    def test_convert_says_what_the_server_made_of_the_page(self):
        with Server() as server, mock.patch.object(desktop_web_install, "process_environment",
                                                            return_value={TOKEN_VARIABLE: TOKEN}):
            code, output = self.run_main(77, server.origin)
        self.assertEqual(code, 0)
        self.assertEqual(output, f"process 77 at {server.origin}/ converted {PAGE_NAME}: "
                                 "# Installed app / hello installed\n")

    def test_an_answer_that_is_not_the_api_s_is_a_failure(self):
        with Server(created=201) as server, mock.patch.object(desktop_web_install, "process_environment",
                                                                       return_value={TOKEN_VARIABLE: TOKEN}), \
                mock.patch.object(desktop_web_install, "call", return_value=(201, b"<html>not the API</html>")):
            code, output = self.run_main(77, server.origin)
        self.assertEqual(code, 1)
        self.assertRegex(output, rf"^FAIL {server.origin}/ did not answer as the workbench's API does: JSONDecodeError")

    def test_a_failure_is_one_line_on_standard_output_and_exit_1(self):
        with Server() as server, mock.patch.object(desktop_web_install, "process_environment",
                                                            return_value={}):
            code, output = self.run_main(77, server.origin)
        self.assertEqual(code, 1)
        self.assertEqual(output, f"FAIL process 77 has no {TOKEN_VARIABLE}: it is not a server Markitai Web started\n")
        self.assertEqual(server.requests, [])


if __name__ == "__main__":
    unittest.main()
