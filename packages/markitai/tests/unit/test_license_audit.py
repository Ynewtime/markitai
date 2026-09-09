"""License compliance guards: no non-commercial deps, AGPL only where declared.

`pymupdf-layout` 1.28.0 shipped under "Polyform Noncommercial or Artifex
Commercial" — a licence that forbids commercial use of an otherwise MIT tool.
1.28.2 moved back to AGPL-3.0 dual licensing. These tests keep the floor and
the audit script that stops the next such regression at CI time.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PKG_PYPROJECT = _REPO_ROOT / "packages" / "markitai" / "pyproject.toml"
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "check_licenses.py"


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_licenses", _AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load_audit_module()


def _core_dependencies() -> list[str]:
    data = tomllib.loads(_PKG_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


class TestPyMuPDFFloor:
    """The dependency floor is the primary defence against the NC licence."""

    def test_pymupdf4llm_floor_excludes_noncommercial_layout(self) -> None:
        pins = [d for d in _core_dependencies() if d.startswith("pymupdf4llm")]
        assert pins == ["pymupdf4llm>=1.28.2"], (
            "pymupdf4llm<1.28.2 resolves pymupdf-layout 1.28.0, which is "
            "Polyform Noncommercial"
        )

    def test_floor_is_documented_in_pyproject_comment(self) -> None:
        text = _PKG_PYPROJECT.read_text(encoding="utf-8")
        line = next(
            ln for ln in text.splitlines() if ln.strip().startswith('"pymupdf4llm')
        )
        lowered = line.lower()
        assert "polyform" in lowered and "1.28.0" in lowered, (
            "the version floor needs an inline reason, like the litellm pin"
        )


@pytest.fixture(scope="module")
def notice() -> str:
    path = _REPO_ROOT / "NOTICE"
    assert path.exists(), "NOTICE is missing"
    return path.read_text(encoding="utf-8")


class TestNoFitzAlias:
    """PyMuPDF 1.28.2 prints the `fitz` deprecation notice on **stdout**.

    That lands inside piped markdown (`markitai doc.pdf | ...`), so importing
    the legacy alias is now a correctness bug, not just a style one. Caught by
    the version bump this file exists to enforce.
    """

    def test_no_source_file_imports_the_legacy_fitz_alias(self) -> None:
        src = _REPO_ROOT / "packages" / "markitai" / "src"
        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{number}"
            for path in src.rglob("*.py")
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            )
            if line.strip().startswith(("import fitz", "from fitz"))
        ]
        assert offenders == [], f"use `import pymupdf` instead: {offenders}"

    def test_importing_the_cli_keeps_stdout_clean(self) -> None:
        """End-to-end guard: nothing on the CLI import path may print."""
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import markitai.cli.logging_config as c; c.setup_logging(verbose=0)",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=_REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "", proc.stdout


class TestNoticeFile:
    """Attribution obligations we currently owe."""

    def test_declares_agpl_pymupdf_trio(self, notice: str) -> None:
        for pkg in ("pymupdf", "pymupdf-layout", "pymupdf4llm"):
            assert pkg in notice
        assert "AGPL-3.0" in notice
        assert "Artifex" in notice

    def test_credits_defuddle_for_webextract(self, notice: str) -> None:
        assert "defuddle" in notice.lower()
        assert "kepano" in notice
        assert "MIT" in notice
        assert "markitai/webextract" in notice

    def test_credits_marker_for_benchmark_scorer(self, notice: str) -> None:
        assert "marker" in notice.lower()
        assert "Apache" in notice
        assert "benchmarks/scorer.py" in notice

    def test_every_extra_is_accounted_for(self, notice: str) -> None:
        """A new extra must say which licence it brings in.

        NOTICE closes by naming each extra's package and its licence family.
        Two extras added in 1.0.0 (`legacy`, `mcp`) were never added to that
        list — the paragraph read as exhaustive while it was not.
        """
        import re
        import tomllib

        metadata = tomllib.loads(
            (_REPO_ROOT / "packages" / "markitai" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        extras = metadata["project"]["optional-dependencies"]
        missing = sorted(
            {
                re.split(r"[<>=!\[; ]", requirement, maxsplit=1)[0]
                for name, requirements in extras.items()
                if name != "all"  # aggregates the others, brings nothing new
                for requirement in requirements
            }
            - set(re.findall(r"[A-Za-z0-9_.-]+", notice))
        )
        assert not missing, (
            f"extras whose package NOTICE never names: {missing}. Add it to "
            "the licence list so the closing paragraph stays exhaustive."
        )


class TestReadmeLicenceSection:
    def test_readme_discloses_agpl_pdf_engine(self) -> None:
        readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("## License", 1)
        assert len(section) == 2, "README has no License section"
        body = section[1]
        assert "AGPL" in body
        assert "PyMuPDF" in body
        assert "NOTICE" in body


class TestAuditClassification:
    """Unit-level behaviour of scripts/check_licenses.py."""

    def test_polyform_noncommercial_is_forbidden(self, audit: ModuleType) -> None:
        assert (
            audit.classify(
                "Dual Licensed - Polyform Noncommercial or Artifex Commercial License"
            )
            == audit.FORBIDDEN
        )

    def test_proprietary_classifier_is_forbidden(self, audit: ModuleType) -> None:
        assert audit.classify("License :: Other/Proprietary License") == audit.FORBIDDEN

    def test_agpl_is_copyleft(self, audit: ModuleType) -> None:
        assert (
            audit.classify(
                "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License"
            )
            == audit.COPYLEFT
        )
        assert audit.classify("AGPL-3.0-or-later") == audit.COPYLEFT

    def test_plain_gpl_is_copyleft(self, audit: ModuleType) -> None:
        assert audit.classify("GPL-3.0") == audit.COPYLEFT
        assert audit.classify("GNU General Public License v2 (GPLv2)") == audit.COPYLEFT

    def test_lgpl_is_not_strong_copyleft(self, audit: ModuleType) -> None:
        # Weak copyleft: dynamic linking imposes no obligation on our sources,
        # so it is not COPYLEFT. It is still reported as RESTRICTED, so that
        # taking on an LGPL dependency stays a declared decision.
        verdict = audit.classify("LGPL-3.0-or-later")
        assert verdict != audit.COPYLEFT
        assert verdict == audit.RESTRICTED

    def test_permissive_licenses_pass(self, audit: ModuleType) -> None:
        for text in ("MIT", "Apache-2.0", "BSD-3-Clause", "MIT-CMU", "ISC"):
            assert audit.classify(text) == audit.PERMISSIVE, text

    def test_gpl_compatible_prose_is_not_copyleft(self, audit: ModuleType) -> None:
        # pandas ships its full licence text, which mentions "GPL-compatible".
        assert audit.classify("Python releases have been GPL-compatible") == (
            audit.PERMISSIVE
        )


class TestAuditPolicy:
    def test_noncommercial_package_fails(self, audit: ModuleType) -> None:
        violations = audit.audit(
            [
                audit.PackageLicense(
                    "pymupdf-layout",
                    "Dual Licensed - Polyform Noncommercial or Artifex Commercial License",
                )
            ]
        )
        assert len(violations) == 1
        assert "pymupdf-layout" in violations[0]
        assert "non-commercial" in violations[0].lower()

    def test_unlisted_agpl_package_fails(self, audit: ModuleType) -> None:
        violations = audit.audit([audit.PackageLicense("some-new-dep", "AGPL-3.0")])
        assert len(violations) == 1
        assert "some-new-dep" in violations[0]
        assert "allowlist" in violations[0].lower()

    def test_allowlisted_agpl_packages_pass(self, audit: ModuleType) -> None:
        packages = [
            audit.PackageLicense(name, "GNU AFFERO GPL 3.0 or Artifex Commercial")
            for name in ("PyMuPDF", "pymupdf_layout", "pymupdf4llm")
        ]
        assert audit.audit(packages) == []

    def test_allowlist_matching_is_pep503_normalized(self, audit: ModuleType) -> None:
        assert audit.audit([audit.PackageLicense("PyMuPDF_Layout", "AGPL-3.0")]) == []

    def test_permissive_packages_pass(self, audit: ModuleType) -> None:
        assert audit.audit([audit.PackageLicense("httpx", "BSD-3-Clause")]) == []

    def test_allowlist_is_exactly_the_pymupdf_trio(self, audit: ModuleType) -> None:
        assert set(audit.ALLOWED_COPYLEFT) == {
            "pymupdf",
            "pymupdf-layout",
            "pymupdf4llm",
        }

    def test_forbidden_wins_over_copyleft_allowlist(self, audit: ModuleType) -> None:
        # The exact 1.28.0 regression: an allowlisted name must not buy a pass
        # for a non-commercial licence.
        violations = audit.audit(
            [audit.PackageLicense("pymupdf-layout", "Polyform Noncommercial")]
        )
        assert len(violations) == 1


class TestMetadataExtraction:
    """Precedence rules that keep real-world metadata from misfiring."""

    def test_license_expression_wins_over_classifiers(self, audit: ModuleType) -> None:
        text = audit.license_text(
            license_expression="BSD-3-Clause",
            license_field=None,
            classifiers=["License :: OSI Approved :: GNU General Public License v2"],
        )
        assert audit.classify(text) == audit.PERMISSIVE

    def test_short_license_field_wins_over_classifiers(self, audit: ModuleType) -> None:
        # pillow-heif: License "BSD-3-Clause" + a stale GPLv2 classifier.
        text = audit.license_text(
            license_expression=None,
            license_field="BSD-3-Clause",
            classifiers=["License :: OSI Approved :: GNU General Public License v2"],
        )
        assert audit.classify(text) == audit.PERMISSIVE

    def test_full_license_text_falls_back_to_classifiers(
        self, audit: ModuleType
    ) -> None:
        # pandas: 50KB of licence text mentioning "GPL-compatible".
        text = audit.license_text(
            license_expression=None,
            license_field="BSD 3-Clause License\n\n" + ("GPL-compatible " * 500),
            classifiers=["License :: OSI Approved :: BSD License"],
        )
        assert audit.classify(text) == audit.PERMISSIVE

    def test_short_license_field_still_catches_polyform(
        self, audit: ModuleType
    ) -> None:
        text = audit.license_text(
            license_expression=None,
            license_field=(
                "Dual Licensed - Polyform Noncommercial or Artifex Commercial License"
            ),
            classifiers=["License :: Other/Proprietary License"],
        )
        assert audit.classify(text) == audit.FORBIDDEN

    def test_no_metadata_is_unknown_not_a_failure(self, audit: ModuleType) -> None:
        text = audit.license_text(
            license_expression=None, license_field=None, classifiers=[]
        )
        assert audit.classify(text) == audit.UNKNOWN
        assert audit.audit([audit.PackageLicense("mystery", text)]) == []


class TestAuditEntryPoint:
    def test_main_returns_nonzero_on_violation(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(
            audit,
            "iter_installed_licenses",
            lambda: [
                audit.PackageLicense("markitai", "MIT"),
                audit.PackageLicense("bad-dep", "Polyform Noncommercial"),
            ],
        )
        assert audit.main([]) == 1
        assert "bad-dep" in capsys.readouterr().out

    def test_main_returns_zero_on_clean_environment(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            audit,
            "iter_installed_licenses",
            lambda: [
                audit.PackageLicense("markitai", "MIT"),
                audit.PackageLicense("httpx", "BSD-3-Clause"),
                audit.PackageLicense("pymupdf", "GNU AFFERO GPL 3.0"),
            ],
        )
        assert audit.main([]) == 0

    def test_scans_the_real_environment_without_violations(
        self, audit: ModuleType
    ) -> None:
        # The actual gate: whatever this venv holds must pass.
        assert audit.audit(list(audit.iter_installed_licenses())) == []


class TestCiWiring:
    def test_lint_job_runs_the_license_audit(self) -> None:
        ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        lint_job = ci.split("  lint:", 1)[1].split("\n  webextract-guardrails:", 1)[0]
        assert "scripts/check_licenses.py" in lint_job


class TestSourceAvailableLicences:
    """Source-available licences must be declared, not classified as permissive.

    Elastic-2.0 and its relatives forbid offering the software to others as
    a hosted service — a live concern for a project that ships ``markitai
    serve``. They contain none of the forbidden markers and no GPL
    spelling, so they used to fall through to ``permissive`` by accident
    rather than by decision. (kreuzberg was the dependency that prompted
    this; it has since relicensed to MIT and left the allowlist.)
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Elastic-2.0",
            "Elastic License 2.0",
            "BUSL-1.1",
            "Business Source License 1.1",
            "SSPL-1.0",
            "Server Side Public License",
        ],
    )
    def test_source_available_is_restricted(self, audit: ModuleType, text: str) -> None:
        assert audit.classify(text) == audit.RESTRICTED

    def test_restricted_dependency_off_the_allowlist_fails(
        self, audit: ModuleType
    ) -> None:
        violations = audit.audit(
            [audit.PackageLicense(name="somepkg", license="BUSL-1.1")]
        )
        assert violations
        assert "somepkg" in violations[0]
        assert "ALLOWED_RESTRICTED" in violations[0]

    def test_allowlisted_restricted_dependency_passes(self, audit: ModuleType) -> None:
        assert (
            audit.audit(
                [audit.PackageLicense(name="cairosvg", license="LGPL-3.0-or-later")]
            )
            == []
        )

    def test_a_relicensed_package_must_be_reviewed_again(
        self, audit: ModuleType
    ) -> None:
        """Leaving the allowlist is what makes a relicense visible.

        kreuzberg shipped under Elastic-2.0, was allowlisted for it, and has
        since moved to MIT. Keeping the stale entry would have silently
        waved through a move back.
        """
        violations = audit.audit(
            [audit.PackageLicense(name="kreuzberg", license="Elastic-2.0")]
        )
        assert violations
        assert "ALLOWED_RESTRICTED" in violations[0]

    def test_lgpl_is_restricted_but_allowlisted(self, audit: ModuleType) -> None:
        assert audit.classify("LGPL-3.0-or-later") == audit.RESTRICTED
        assert (
            audit.audit(
                [audit.PackageLicense(name="cairosvg", license="LGPL-3.0-or-later")]
            )
            == []
        )

    def test_lgpl_is_still_not_read_as_gpl(self, audit: ModuleType) -> None:
        assert audit.classify("LGPL-3.0-or-later") != audit.COPYLEFT


class TestAuditsTheRightEnvironment:
    """A compliance gate that finds nothing must not report success.

    Run as ``./scripts/check_licenses.py`` the shebang picks the system
    interpreter, where markitai is not installed: the audit then inspected 3
    unrelated distributions and printed "License audit passed", which reads
    exactly like a real pass. The gate now refuses to grade an environment
    that does not contain the package it is auditing.
    """

    def test_missing_markitai_fails_with_the_correct_invocation(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(
            audit,
            "iter_installed_licenses",
            lambda: iter([audit.PackageLicense(name="pip", license="MIT")]),
        )
        assert audit.main([]) == 1
        out = capsys.readouterr().out
        assert "markitai" in out
        assert "uv run" in out

    def test_an_environment_containing_markitai_is_graded(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            audit,
            "iter_installed_licenses",
            lambda: iter(
                [
                    audit.PackageLicense(name="markitai", license="MIT"),
                    audit.PackageLicense(name="pymupdf", license="AGPL-3.0"),
                ]
            ),
        )
        assert audit.main([]) == 0
