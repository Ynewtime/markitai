"""The 2.x setup scripts against a local fake release.

setup.sh runs end to end (POSIX shells present on this host) with a stand-in
`markitai` shell script in a release-shaped archive; setup.ps1's functions
run under pwsh where it is installed. A real archive and a Windows host are
exercised separately (docs/ci.md#releases).
"""
from pathlib import Path
import functools
import hashlib
import http.server
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "scripts/install/setup.sh"
SETUP_PS1 = ROOT / "scripts/install/setup.ps1"

FAKE_MARKITAI = """#!/bin/sh
# Stand-in for the release executable.
case "${1:-}" in
    --version) echo "markitai @VERSION@" ;;
    init) echo init >> "$HOME/calls" ;;
    doctor) echo "doctor $*" >> "$HOME/calls" ;;
esac
"""


def shell_target():
    machine = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(
        platform.machine().lower())
    system = {"Darwin": "apple-darwin", "Linux": "unknown-linux-gnu"}.get(platform.system())
    if not machine or not system:
        return None
    if system == "apple-darwin" and machine == "x86_64":
        translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
        if translated.stdout.strip() == "1":
            machine = "aarch64"
    return f"{machine}-{system}"


def release_tar(version, target):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as bundle:
        script = FAKE_MARKITAI.replace("@VERSION@", version).encode()
        members = [("markitai", 0o755, script), ("LICENSE", 0o644, b"MIT\n"), ("NOTICE", 0o644, b"notice\n"),
                   ("licenses/example/LICENSE", 0o644, b"x\n"), ("llms.txt", 0o644, b"# guide\n")]
        for name, mode, content in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
        for name in ["mkai", "markitai-mcp"]:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = "markitai"
            bundle.addfile(info)
    return data.getvalue()


def release_zip(version):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as bundle:
        for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe"]:
            bundle.writestr(name, f"exe {version}")
        bundle.writestr("LICENSE", "MIT\n")
        bundle.writestr("licenses/example/LICENSE", "x\n")
    return data.getvalue()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class FakeReleases:
    """`<base>/download/v<version>/<asset>` and `<base>/latest/download/<asset>`, as GitHub serves them."""

    def __init__(self, root):
        self.root = root
        handler = functools.partial(QuietHandler, directory=str(root))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def publish(self, version, assets, latest=True, corrupt=None):
        folders = [self.root / "download" / f"v{version}"]
        if latest:
            shutil.rmtree(self.root / "latest", ignore_errors=True)
            folders.append(self.root / "latest" / "download")
        lines = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(assets.items()))
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "SHA256SUMS").write_text(lines, encoding="utf-8")
            for name, content in assets.items():
                (folder / name).write_bytes(content + (b"tampered" if name == corrupt else b""))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@unittest.skipUnless(shutil.which("sh") and shell_target() and os.name != "nt", "needs a POSIX shell on macOS/Linux")
class SetupShTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        self.home = self.work / "home with space"
        self.home.mkdir()
        self.releases = FakeReleases(self.work / "releases")
        self.addCleanup(self.releases.close)
        self.target = shell_target()
        for version in ["2.0.0", "2.0.1"]:
            self.releases.publish(version, {
                f"markitai-v{version}-{self.target}.tar.gz": release_tar(version, self.target),
                f"markitai-v{version}-x86_64-pc-windows-msvc.zip": release_zip(version),
            }, latest=version == "2.0.0")

    def run_setup(self, *arguments, shell="sh", **environment):
        env = {"HOME": str(self.home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "NO_COLOR": "1",
               "MARKITAI_DOWNLOAD_URL": self.releases.base, "TMPDIR": str(self.work), **environment}
        return subprocess.run([shell, str(SETUP_SH), *arguments], env=env, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=120)

    def test_install_update_and_uninstall(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bin_dir = self.home / ".local/bin"
        self.assertEqual(subprocess.check_output([bin_dir / "markitai", "--version"], text=True).strip(),
                         "markitai 2.0.0")
        for alias in ["mkai", "markitai-mcp"]:
            self.assertEqual(os.readlink(bin_dir / alias), "markitai")
        current = self.home / ".local/share/markitai/current"
        self.assertEqual((current / "LICENSE").read_text(), "MIT\n")
        self.assertTrue((current / "licenses/example/LICENSE").is_file())
        self.assertFalse((current / "markitai").exists())
        self.assertIn("version=2.0.0", (current / "receipt").read_text())
        self.assertIn("SHA-256 verified", result.stdout)
        self.assertIn("PATH does not include", result.stdout)
        self.assertIn("MARKITAI_INSTALL_OPTIONAL=1", result.stdout)
        self.assertIn("--uninstall", result.stdout)
        self.assertEqual((self.home / "calls").read_text(), "init\n")
        self.assertEqual([path.name for path in self.work.glob("markitai-setup.*")], [])

        pinned = self.run_setup(MARKITAI_VERSION="v2.0.1", MARKITAI_INSTALL_OPTIONAL="1",
                                PATH=f"{bin_dir}:/usr/bin:/bin")
        self.assertEqual(pinned.returncode, 0, pinned.stdout + pinned.stderr)
        self.assertEqual(subprocess.check_output([bin_dir / "markitai", "--version"], text=True).strip(),
                         "markitai 2.0.1")
        self.assertNotIn("PATH does not include", pinned.stdout)
        self.assertIn("doctor doctor --fix", (self.home / "calls").read_text())
        self.assertIn("version=2.0.1", (current / "receipt").read_text())

        (self.home / ".markitai").mkdir()
        removed = self.run_setup("--uninstall")
        self.assertEqual(removed.returncode, 0, removed.stdout + removed.stderr)
        self.assertEqual(sorted(path.name for path in bin_dir.iterdir()), [])
        self.assertFalse((self.home / ".local/share/markitai").exists())
        self.assertIn("~/.markitai", removed.stdout)

    def test_other_shells_and_chinese(self):
        for shell in ["dash", "bash", "zsh"]:
            if not shutil.which(shell):
                continue
            with self.subTest(shell=shell):
                result = self.run_setup(shell=shell, LANG="zh_CN.UTF-8", MARKITAI_INSTALL_OPTIONAL="0")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("已校验 SHA-256", result.stdout)
                self.assertIn("markitai doctor --fix", result.stdout)

    def test_checksum_mismatch_installs_nothing(self):
        name = f"markitai-v2.0.0-{self.target}.tar.gz"
        self.releases.publish("2.0.0", {name: release_tar("2.0.0", self.target)}, corrupt=name)
        result = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("SHA-256 mismatch", result.stdout)
        self.assertFalse((self.home / ".local/bin/markitai").exists())

    def test_refusals(self):
        for environment, message in [({"MARKITAI_VERSION": "1.2.0"}, "Python package"),
                                     ({"MARKITAI_VERSION": "latest"}, "X.Y.Z"),
                                     ({"MARKITAI_VERSION": "2.9.9"}, "Download failed")]:
            with self.subTest(environment=environment):
                result = self.run_setup(**environment)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stdout)
        self.assertEqual(self.run_setup("--bogus").returncode, 2)
        self.assertIn("--uninstall", self.run_setup("--help").stdout)

    def test_piped_into_sh_replaces_python_markitai_from_uv(self):
        tools = self.work / "tools"
        tools.mkdir()
        (tools / "uv").write_text('#!/bin/sh\n'
                                  'case "$1 $2" in\n'
                                  '  "tool list") echo "markitai v1.2.0"; echo "- markitai" ;;\n'
                                  '  "tool uninstall") echo "uv $*" >> "$HOME/calls" ;;\n'
                                  'esac\n')
        (tools / "uv").chmod(0o755)
        env = {"HOME": str(self.home), "PATH": f"{tools}:/usr/bin:/bin", "LANG": "C", "TERM": "dumb",
               "MARKITAI_DOWNLOAD_URL": self.releases.base, "TMPDIR": str(self.work)}
        result = subprocess.run(["sh"], input=SETUP_SH.read_text(encoding="utf-8"), env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("uv tool uninstall markitai", (self.home / "calls").read_text())
        self.assertIn("+ Removed the Python markitai that uv installed", result.stdout)
        self.assertTrue(result.stdout.isascii(), "C locale and TERM=dumb print ASCII marks")

    def test_foreign_alias_is_kept_and_python_scripts_replaced(self):
        bin_dir = self.home / ".local/bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "mkai").write_bytes(b"\x7fELF someone else's tool")
        (bin_dir / "markitai-mcp").write_text("#!/usr/bin/python3\nfrom markitai.mcp_launcher import main\n")
        (bin_dir / "markitai").symlink_to(self.work / "old-uv-tool-markitai")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("left alone", result.stdout)
        self.assertEqual((bin_dir / "mkai").read_bytes(), b"\x7fELF someone else's tool")
        self.assertEqual(os.readlink(bin_dir / "markitai-mcp"), "markitai")
        self.assertFalse((bin_dir / "markitai").is_symlink())


PS_HARNESS = r"""
param([string]$Script, [string]$Base, [string]$Work)
$ErrorActionPreference = "Stop"
# Load every definition, not the two statements that run setup.
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$tokens, [ref]$errors)
if ($errors) { throw "parse errors: $errors" }
$body = @($ast.EndBlock.Statements | Where-Object { $_.Extent.Text -notmatch '^if \((-not \$script:OnWindows|\$Uninstall)' })
if ($body.Count -ne $ast.EndBlock.Statements.Count - 2) { throw "setup.ps1 ends differently than expected" }
. ([scriptblock]::Create(($body | ForEach-Object { $_.Extent.Text }) -join "`n"))
function Assert($Condition, $Message) { if (-not $Condition) { throw "FAILED: $Message" } }
function Write-Host { }
function Send-EnvironmentChange { }
$script:FakeUserPath = '%USERPROFILE%\bin;C:\Tools'
function Get-UserPath { $script:FakeUserPath }
function Set-UserPath { param($Value) $script:FakeUserPath = $Value }

Assert ((Get-ReleaseTarget "AMD64") -eq "x86_64-pc-windows-msvc") "x64 target"
Assert ((Get-ReleaseTarget "ARM64") -eq "aarch64-pc-windows-msvc") "arm64 target"
$failed = $false; try { Get-ReleaseTarget "X86" } catch { $failed = $true }; Assert $failed "x86 refused"

$latest = Resolve-Release $Base "aarch64-pc-windows-msvc" "" $Work
Assert ($latest.Version -eq "2.0.0" -and $latest.Archive -eq "markitai-v2.0.0-aarch64-pc-windows-msvc.zip") "latest"
$pinned = Resolve-Release $Base "x86_64-pc-windows-msvc" "v2.0.1" $Work
Assert ($pinned.Version -eq "2.0.1") "pinned"
foreach ($wanted in @("1.2.0", "latest", "2.9.9")) {
    $failed = $false; try { Resolve-Release $Base "x86_64-pc-windows-msvc" $wanted $Work } catch { $failed = $true }
    Assert $failed "refuses $wanted"
}

$zip = Join-Path $Work $pinned.Archive
Save-Download "$Base/download/v2.0.1/$($pinned.Archive)" $zip
Test-Checksum $zip $pinned.Sha256
$failed = $false; try { Test-Checksum $zip ("0" * 64) } catch { $failed = $true }
Assert ($failed -and -not (Test-Path $zip)) "a mismatch deletes the download"
Save-Download "$Base/download/v2.0.1/$($pinned.Archive)" $zip
$unpacked = Join-Path $Work "unpacked"
Expand-Release $zip $unpacked
Assert (Test-Path (Join-Path $unpacked "licenses/example/LICENSE")) "expanded"
$evil = Join-Path $Work "evil.zip"
Save-Download "$Base/evil.zip" $evil
$failed = $false; try { Expand-Release $evil (Join-Path $Work "evil") } catch { $failed = $true }
Assert ($failed -and -not (Test-Path (Join-Path $Work "evil"))) "unsafe member refused before extraction"

$install = Join-Path $Work "Programs/markitai"
New-Item -ItemType Directory -Force -Path $install | Out-Null
Set-Content -LiteralPath (Join-Path $install "user-notes.txt") -Value "mine"
Set-Content -LiteralPath (Join-Path $install "markitai.exe") -Value "old exe"
Install-Release $unpacked $install $pinned "x86_64-pc-windows-msvc"
Assert ((Get-Content -Raw (Join-Path $install "user-notes.txt")).Trim() -eq "mine") "files not installed here stay"
Assert ((Get-Content -Raw (Join-Path $install "markitai.exe")) -eq "exe 2.0.1") "new exe"
Assert (@(Get-ChildItem $install -Filter "*.old-*").Count -eq 0) "old exe deleted"
Assert ((Get-Content -Raw (Join-Path $install "receipt")) -match "version=2.0.1") "receipt"
Assert ((Get-InstalledNames $install) -contains "licenses") "receipt lists the folder's names"
$second = Join-Path $Work "second"
New-Item -ItemType Directory -Path $second | Out-Null
foreach ($name in @("markitai.exe", "mkai.exe", "markitai-mcp.exe")) { Set-Content -LiteralPath (Join-Path $second $name) -Value "exe 2.0.2" }
Install-Release $second $install @{ Version = "2.0.2" } "x86_64-pc-windows-msvc"
Assert (-not (Test-Path (Join-Path $install "licenses"))) "the previous release's files go"
Assert (Test-Path (Join-Path $install "user-notes.txt")) "still kept"

Assert (Add-ToUserPath $install) "added"
Assert ($script:FakeUserPath -eq "$install;%USERPROFILE%\bin;C:\Tools") "prepended, others kept unexpanded"
Assert (-not (Add-ToUserPath ($install + "\"))) "not twice"
Assert (Remove-FromUserPath $install) "removed"
Assert ($script:FakeUserPath -eq '%USERPROFILE%\bin;C:\Tools') "user Path restored"

# The whole flow, with the pieces only Windows has replaced.
function Get-NativeArchitecture { "ARM64" }
function Invoke-MarkitaiVersion { param($Executable) Assert (Test-Path $Executable) "unpacked exe"; "markitai 2.0.0" }
function Remove-PythonMarkitai { }
$env:LOCALAPPDATA = Join-Path $Work "LocalAppData"
$env:MARKITAI_DOWNLOAD_URL = $Base
$env:MARKITAI_INSTALL_OPTIONAL = "0"
$env:MARKITAI_HOME = Join-Path $Work "state"
Invoke-Install
$programs = Join-Path $env:LOCALAPPDATA "Programs/markitai"
Assert ((Get-Content -Raw (Join-Path $programs "mkai.exe")) -eq "exe 2.0.0") "installed latest"
Assert ($script:FakeUserPath.StartsWith("$programs;")) "on the user Path"
Assert (@(Get-ChildItem ([IO.Path]::GetTempPath()) -Filter "markitai-setup-*").Count -eq 0) "work folder removed"
Invoke-Uninstall
Assert (-not (Test-Path $programs)) "uninstalled"
Assert ($script:FakeUserPath -eq '%USERPROFILE%\bin;C:\Tools') "off the user Path"

$script:LANG_CODE = "zh"
Assert ((Msg "verified") -eq ("$([char]0x5DF2)$([char]0x6821)$([char]0x9A8C) SHA-256")) "zh decoding"
$script:LANG_CODE = "en"
Assert ((Msg "verified") -eq "SHA-256 verified") "en"
"ok"
"""


@unittest.skipUnless(shutil.which("pwsh"), "needs PowerShell 7 (pwsh)")
class SetupPs1Tests(unittest.TestCase):
    def test_functions_against_a_fake_release(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            releases = FakeReleases(work / "releases")
            self.addCleanup(releases.close)
            for version in ["2.0.0", "2.0.1"]:
                releases.publish(version, {
                    f"markitai-v{version}-{target}.zip": release_zip(version)
                    for target in ["x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc"]
                }, latest=version == "2.0.0")
            with zipfile.ZipFile(work / "releases/evil.zip", "w") as bundle:
                for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe"]:
                    bundle.writestr(name, "x")
                bundle.writestr("../escape.txt", "x")
            harness = work / "harness.ps1"
            harness.write_text(PS_HARNESS, encoding="utf-8")
            (work / "scratch").mkdir()
            home = work / "home"
            home.mkdir()
            env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home), "TMPDIR": str(work / "scratch")}
            for name in ["MARKITAI_VERSION", "MARKITAI_INSTALL_DIR", "MARKITAI_UNINSTALL"]:
                env.pop(name, None)
            result = subprocess.run(["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(harness),
                                     "-Script", str(SETUP_PS1), "-Base", releases.base, "-Work", str(work / "scratch")],
                                    capture_output=True, text=True, timeout=180, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip().splitlines()[-1], "ok")


class SetupScriptTextTests(unittest.TestCase):
    def test_ps1_is_ascii(self):
        SETUP_PS1.read_bytes().decode("ascii")

    def test_chinese_tables_match(self):
        shell = SETUP_SH.read_text(encoding="utf-8")
        zh_block = shell.split('if [ "$LANG_CODE" = zh ]; then', 1)[1].split('*) echo "$1" ;;', 1)[0]
        shell_zh = dict(re.findall(r'^\s+([a-z_]+)\) echo "(.*)" ;;$', zh_block, re.M))
        ps = SETUP_PS1.read_text(encoding="ascii")
        ps_zh_block = ps.split("$text = switch ($Key) {", 1)[1].split("default { $null }", 1)[0]
        ps_zh = {key: re.sub(r"\\u([0-9a-f]{4})", lambda match: chr(int(match.group(1), 16)), value).replace("''", "'")
                 for key, value in re.findall(r'^\s+"([a-z_]+)" \{ \'(.*)\' \}$', ps_zh_block, re.M)}
        ps_en = set(re.findall(r'^\s+"([a-z_]+)" \{ "', ps.split("default { $null }", 1)[1], re.M))
        self.assertEqual(set(ps_zh), ps_en, "every English message of setup.ps1 has its Chinese")
        shared = set(ps_zh) & set(shell_zh)
        self.assertGreater(len(shared), 30)
        for key in shared:
            self.assertEqual(ps_zh[key], shell_zh[key], key)


if __name__ == "__main__":
    unittest.main()
