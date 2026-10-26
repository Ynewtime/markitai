"""Child processes for the helper tests that cannot hold a test past its timeout.

`run` starts the child in a session of its own (a new process group on POSIX),
writes its output to temporary files instead of pipes, so a grandchild that
keeps them open cannot block the test, and on a timeout stops the child with
everything it started (ci_process's tree stop). On POSIX whatever the child
left running in its session is stopped once it exits, too.
"""
import faulthandler
import locale
import os
import signal
import subprocess
import tempfile

from ci_process import _stop_tree

# A child of the helper tests takes seconds; a hang fails the test within a minute.
TIMEOUT = 60
# A whole test, for a hang outside any child (a blocking close, a lock).
TEST_TIMEOUT = 180


class Timeout(subprocess.TimeoutExpired):
    """TimeoutExpired that shows the end of what the child wrote."""

    def __str__(self):
        lines = [super().__str__()]
        for name, text in [("stdout", self.output), ("stderr", self.stderr)]:
            if text:
                lines.append(f"{name}: ...{text[-2000:]}")
        return "\n".join(lines)


def watch(case, seconds=TEST_TIMEOUT):
    """Dump every thread's stack and end the run if `case` lasts over `seconds`."""
    faulthandler.dump_traceback_later(seconds, exit=True)
    case.addCleanup(faulthandler.cancel_dump_traceback_later)


def run(command, *, input=None, stdin=subprocess.DEVNULL, timeout=TIMEOUT, check=False, text=False, encoding=None,
        errors=None, **options):
    """subprocess.run with stdout and stderr captured, bounded as described above."""
    decode = text or encoding or errors
    encoding = encoding or (locale.getpreferredencoding(False) if decode else None)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as feed:
        if input is not None:
            feed.write(input.encode(encoding) if isinstance(input, str) else input)
            feed.seek(0)
            stdin = feed
        process = subprocess.Popen(command, stdin=stdin, stdout=out, stderr=err, start_new_session=os.name != "nt",
                                   **options)

        def captured():
            values = []
            for stream in [out, err]:
                stream.seek(0)
                data = stream.read()
                if decode:
                    data = data.decode(encoding, errors or "strict").replace("\r\n", "\n").replace("\r", "\n")
                values.append(data)
            return values

        try:
            code = process.wait(timeout=timeout)
        except BaseException as error:
            _stop_tree(process)
            if isinstance(error, subprocess.TimeoutExpired):
                stdout, stderr = captured()
                raise Timeout(command, timeout, stdout, stderr) from None
            raise
        finally:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        stdout, stderr = captured()
    if check and code:
        raise subprocess.CalledProcessError(code, command, stdout, stderr)
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def output(command, **options):
    """subprocess.check_output, bounded as `run`."""
    return run(command, check=True, **options).stdout
