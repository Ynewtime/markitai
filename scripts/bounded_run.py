"""Child processes for the helper tests that cannot hold a test past its timeout.

`run` starts the child in a session of its own on POSIX, or in a Job Object
on Windows (ci_process's, killing its members when its handle closes), and
writes its output to temporary files instead of pipes, so a grandchild that
keeps them open cannot block the test. On a timeout it stops the child with
everything it started, and whatever the child leaves running in its session or
job is stopped once it exits.
"""
import faulthandler
import locale
import os
import signal
import subprocess
import tempfile

from ci_process import _WindowsJob, _stop_tree, _windows_job

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

    def captured():
        values = []
        for stream in [out, err]:
            stream.seek(0)
            data = stream.read()
            if decode:
                data = data.decode(encoding, errors or "strict").replace("\r\n", "\n").replace("\r", "\n")
            values.append(data)
        return values

    job = _windows_job(kill_on_close=True) if os.name == "nt" else None
    try:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as feed:
            if input is not None:
                feed.write(input.encode(encoding) if isinstance(input, str) else input)
                feed.seek(0)
                stdin = feed
            process = subprocess.Popen(command, stdin=stdin, stdout=out, stderr=err,
                                       start_new_session=os.name != "nt",
                                       creationflags=_WindowsJob.CREATE_SUSPENDED if job else 0, **options)
            if job is not None:
                try:
                    joined = job.adopt(process.pid)  # it runs either way
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                if not joined:  # then taskkill /T is the fallback
                    job.close()
                    job = None
            try:
                code = process.wait(timeout=timeout)
            except BaseException as error:
                _stop_tree(process, job)
                if isinstance(error, subprocess.TimeoutExpired):
                    stdout, stderr = captured()
                    raise Timeout(command, timeout, stdout, stderr) from None
                raise
            finally:
                _stop_leftovers(process, job)
            stdout, stderr = captured()
    finally:
        if job is not None:
            job.close()
    if check and code:
        raise subprocess.CalledProcessError(code, command, stdout, stderr)
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def _stop_leftovers(process, job):
    """Stop what the child left running in its job (Windows) or session (POSIX)."""
    if job is not None:
        job.stop(timeout=15)
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def output(command, **options):
    """subprocess.check_output, bounded as `run`."""
    return run(command, check=True, **options).stdout
