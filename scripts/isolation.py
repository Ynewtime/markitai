"""Keep helper-run tests and audits away from the developer's real user state."""
import os
from pathlib import Path
import sys
import uuid


def isolated_env(home, markitai_home, environ=None):
    """Copy `environ` (default: this process) with private HOME and Markitai state.

    HOME and USERPROFILE name `home`, MARKITAI_HOME names `markitai_home`; both
    directories are created. CARGO_HOME and RUSTUP_HOME default to the original
    home so rustup and Cargo still find the installed toolchain.
    """
    environment = dict(os.environ if environ is None else environ)
    original = Path.home()
    environment.setdefault("CARGO_HOME", str(original / ".cargo"))
    environment.setdefault("RUSTUP_HOME", str(original / ".rustup"))
    home, markitai_home = Path(home).absolute(), Path(markitai_home).absolute()
    home.mkdir(parents=True, exist_ok=True)
    markitai_home.mkdir(parents=True, exist_ok=True)
    environment.update(HOME=str(home), USERPROFILE=str(home), MARKITAI_HOME=str(markitai_home))
    return environment


class ProtectedStateAccess(PermissionError):
    """A Python operation attempted to access real Markitai user state."""


NETWORK_EVENTS = {"socket.connect", "socket.getaddrinfo", "socket.sendto"}
PATH_EVENTS = {
    "open": (0,), "os.listdir": (0,), "os.scandir": (0,), "os.chdir": (0,),
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,),
    "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,),
    "os.rename": (0, 1), "os.link": (0, 1), "os.symlink": (0, 1),
}


def install_state_guard(scope, allow_network=lambda event, args: False):
    """Block Python access to the real ~/.markitai and networking `allow_network` refuses.

    Returns the live record of blocked events. The guard is a Python audit hook
    and cannot be removed. It proves itself on a missing probe path before
    returning; only its own ProtectedStateAccess counts, so an ordinary OS
    permission error cannot pass the self-test.
    """
    # Capture both spellings before registering the hook; do not change HOME.
    protected = Path.home() / ".markitai"
    roots = {os.path.abspath(protected), os.path.realpath(protected)}
    state = {"protected_paths": sorted(roots), "scope": scope, "self_tests": {},
             "blocked_state_events": [], "blocked_network_events": []}
    testing = True

    def guard(event, args):
        if event in NETWORK_EVENTS and not allow_network(event, args):
            state["blocked_network_events"].append(event)
            raise PermissionError(f"Network access is disabled here: {event}")
        for index in PATH_EVENTS.get(event, ()):
            path = args[index]
            if isinstance(path, int):
                continue  # Descriptor-only operations are outside this path guard.
            absolute = os.path.abspath(os.fsdecode(path) if path is not None else ".")
            if any(candidate == root or candidate.startswith(root + os.sep)
                   for candidate in (absolute, os.path.realpath(absolute)) for root in roots):
                if not testing:
                    state["blocked_state_events"].append({"event": event, "path": absolute})
                raise ProtectedStateAccess(f"Blocked {event} on protected user state: {absolute}")

    sys.addaudithook(guard)
    # No directory is created. The random missing parent also makes accidental
    # fall-through fail without writing a probe into the user's existing state.
    probe = protected / ("__markitai_guard_probe_" + uuid.uuid4().hex) / "blocked"
    operations = {
        "read": lambda: open(probe, "rb"), "write": lambda: open(probe, "wb"),
        "listdir": lambda: os.listdir(probe), "scandir": lambda: os.scandir(probe),
    }
    for name, operation in operations.items():
        try:
            result = operation()
        except ProtectedStateAccess:
            state["self_tests"][name] = "blocked"
        else:
            if hasattr(result, "close"):
                result.close()
            raise RuntimeError(f"Protected-state guard did not block {name}")
    testing = False
    return state
