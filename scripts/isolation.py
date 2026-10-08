"""Keep helper-run tests and audits away from the developer's real user state."""
import os
from pathlib import Path


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
