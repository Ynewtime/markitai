"""Keep the per-provider tables answerable from one place.

The rot this module exists to prevent: the model each provider defaults to
was hand-copied into four tables (credential detection, the `init` wizard,
`serve`'s startup candidates, the interactive wizard) plus the sample config
and the published docs. By 2026-08 they had drifted — a refresh commit
updated the wizard to the current generation and left the other three on the
previous one, and the docs still advertised a DeepSeek model that had been
retired.

The same happened to provider -> API-key env var: four copies (credential
detection, the setup wizard, serve's key check, discovery's card list), and
the wizard's had lost OpenRouter.

Three guards, matching the directions the copies drifted:

* **No second copy in the source.** A ``provider/model`` string literal
  outside ``constants.PROVIDER_DEFAULT_MODELS`` is either a routing prefix
  (allowlisted below, with the reason) or a new hand copy.
* **The docs name what the code picks.** Every default must be written down
  somewhere a reader can find it; a default the docs never mention is one
  nobody can verify went stale.
* **No second copy of the credential env vars.** A ``*_API_KEY`` literal
  outside the table is a new hand copy.

What this cannot catch: a default that is simply a bad choice, or docs that
name the right id while describing it wrongly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_DEFAULT_MODELS

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = Path(__file__).resolve().parents[2] / "src" / "markitai"
_DOC_ROOTS = (_REPO_ROOT / "website", _REPO_ROOT / "skills")
_README = _REPO_ROOT / "README.md"

# Literals that name a provider but are not a default pick. Each one is a
# prefix or path fragment the code matches against, not a model to send.
_NOT_A_DEFAULT_PICK = frozenset(
    {
        # llm/router.py: Copilot deployment families, for capability routing
        "copilot/claude-",
        "copilot/gemini-",
        "copilot/gpt-4.1",
        "copilot/gpt-4o",
        "copilot/gpt-5",
        "copilot/raptor-",
        # cli/processors/validators.py: models Copilot refuses to serve
        "copilot/o1",
        "copilot/o3",
        # providers/discovery.py: an HTTP path under the OpenAI base url
        "openai/models",
    }
)


def _module_docstrings(tree: ast.Module) -> set[str]:
    """Docstrings are prose; only executable literals can be a hidden table."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                found.add(doc)
    return found


def _provider_model_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _module_docstrings(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if value in docstrings or value in _NOT_A_DEFAULT_PICK:
            continue
        for provider in PROVIDER_DEFAULT_MODELS:
            prefix = f"{provider}/"
            if value.startswith(prefix) and len(value) > len(prefix):
                hits.append((node.lineno, value))
                break
    return hits


def test_provider_defaults_have_no_second_copy_in_the_source() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "constants.py":
            continue
        for lineno, value in _provider_model_literals(path):
            offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {value!r}")

    assert not offenders, (
        "provider/model literals outside constants.PROVIDER_DEFAULT_MODELS "
        "(read it from there, or add the string to _NOT_A_DEFAULT_PICK with "
        "the reason it is not a default):\n" + "\n".join(offenders)
    )


def test_api_key_env_vars_have_no_second_copy_in_the_source() -> None:
    """A provider's key name belongs to PROVIDER_API_KEY_ENV, nowhere else."""
    known = set(PROVIDER_API_KEY_ENV.values())
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "constants.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _module_docstrings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in known
                and node.value not in docstrings
            ):
                offenders.append(
                    f"{path.relative_to(_SRC)}:{node.lineno}: {node.value}"
                )

    assert not offenders, (
        "API-key env names outside constants.PROVIDER_API_KEY_ENV (read them "
        "from there so a provider is added in one place):\n" + "\n".join(offenders)
    )


@pytest.mark.skipif(
    not (_REPO_ROOT / "website").is_dir(),
    reason="website/ is not part of this checkout",
)
def test_docs_name_every_model_the_code_defaults_to() -> None:
    corpus = _README.read_text(encoding="utf-8") if _README.is_file() else ""
    for root in _DOC_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            corpus += path.read_text(encoding="utf-8")

    missing = sorted(
        f"{provider} -> {model}"
        for provider, model in PROVIDER_DEFAULT_MODELS.items()
        if model not in corpus
    )
    assert not missing, (
        "defaults the docs never name (a reader cannot tell what markitai "
        "will pick, and nobody notices when the pick goes stale):\n"
        + "\n".join(missing)
    )
