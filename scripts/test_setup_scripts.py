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

    def test_a_share_folder_setup_did_not_create_is_kept(self):
        current = self.home / ".local/share/markitai/current"
        current.mkdir(parents=True)
        (current / "notes.txt").write_text("mine")
        result = self.run_setup()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Setup did not create this folder", result.stdout)
        self.assertFalse((self.home / ".local/bin/markitai").exists())
        self.assertEqual(self.run_setup("--uninstall").returncode, 0)
        self.assertEqual((current / "notes.txt").read_text(), "mine")

    def test_no_hash_tool_says_so(self):
        tools = self.work / "no-hash-tools"
        tools.mkdir()
        for name in ["cat", "chmod", "cp", "curl", "cut", "env", "getconf", "grep", "gzip", "head", "id", "ln", "ls",
                     "mkdir", "mktemp", "mv", "readlink", "rm", "sed", "sh", "tar", "tr", "uname", "wget"]:
            if shutil.which(name):
                (tools / name).symlink_to(shutil.which(name))
        result = self.run_setup(PATH=str(tools))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Needs sha256sum, shasum or openssl", result.stdout)
        self.assertIn("Setup failed", result.stdout)

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
        self.assertIn('markitai@v1.2.0#subdirectory=packages/markitai" --with "pymupdf4llm<2"',
                      self.run_setup(MARKITAI_VERSION="1.2.0").stdout)
        self.assertEqual(self.run_setup("--bogus").returncode, 2)
        self.assertIn("--uninstall", self.run_setup("--help").stdout)

    def fake_tool(self, name, script):
        tools = self.work / "tools"
        tools.mkdir(exist_ok=True)
        (tools / name).write_text("#!/bin/sh\n" + script)
        (tools / name).chmod(0o755)
        return tools

    def fake_uv(self):
        return self.fake_tool("uv", 'case "$1 $2" in\n'
                                    '  "tool list") echo "markitai v1.2.0"; echo "- markitai" ;;\n'
                                    '  "tool uninstall") echo "uv $*" >> "$HOME/calls" ;;\n'
                                    'esac\n')

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root writes into a read-only folder")
    def test_python_markitai_from_uv_stays_when_the_install_fails(self):
        tools = self.fake_uv()
        target = self.work / "read-only"
        target.mkdir()
        target.chmod(0o555)
        self.addCleanup(target.chmod, 0o755)
        result = self.run_setup(PATH=f"{tools}:/usr/bin:/bin", MARKITAI_INSTALL_DIR=str(target))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Could not install into this folder", result.stdout)
        self.assertIn("Setup failed", result.stdout)
        self.assertNotIn("uv tool uninstall", (self.home / "calls").read_text() if (self.home / "calls").exists() else "")

    def test_a_failed_copy_leaves_no_staged_file(self):
        tools = self.fake_tool("chmod", "exit 1\n")
        result = self.run_setup(PATH=f"{tools}:/usr/bin:/bin")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Could not install into this folder", result.stdout)
        self.assertEqual([path.name for path in (self.home / ".local/bin").iterdir()], [])

    def test_piped_into_sh_replaces_python_markitai_from_uv(self):
        tools = self.fake_uv()
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
# Load every definition of setup's script block, not the two statements that run setup.
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$tokens, [ref]$errors)
if ($errors) { throw "parse errors: $errors" }
$block = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.ScriptBlockExpressionAst] }, $false)
if (-not $block -or $ast.EndBlock.Statements.Count -ne 1) { throw "setup.ps1 is not one script block" }
$statements = $block.ScriptBlock.EndBlock.Statements
$body = @($statements | Where-Object { $_.Extent.Text -notmatch '^if \((-not \$OnWindows|\$Uninstall)' })
if ($body.Count -ne $statements.Count - 2) { throw "setup.ps1 ends differently than expected" }
. ([scriptblock]::Create(($body | ForEach-Object { $_.Extent.Text }) -join "`n"))
function Assert($Condition, $Message) { if (-not $Condition) { throw "FAILED: $Message" } }
$script:Shown = New-Object System.Collections.ArrayList
function Write-Host { param([Parameter(Position = 0)]$Object, $ForegroundColor, [switch]$NoNewline) [void]$script:Shown.Add("$Object") }
function Send-EnvironmentChange { }
# uv's Python markitai goes only once the new release is staged in the folder.
$script:UvRemoved = 0
function Remove-PythonMarkitai {
    Assert (@(Get-ChildItem -LiteralPath $script:InstallTo -Force | Where-Object { $_.Name -like ".markitai-new-*" }).Count -eq 1) "staged first"
    $script:UvRemoved++
}
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
$script:Shown.Clear()
foreach ($wanted in @("1.2.0", "latest", "2.9.9")) {
    $failed = $false; try { Resolve-Release $Base "x86_64-pc-windows-msvc" $wanted $Work } catch { $failed = $true }
    Assert $failed "refuses $wanted"
}
Assert (($script:Shown -join "`n") -match [regex]::Escape('markitai@v1.2.0#subdirectory=packages/markitai" --with "pymupdf4llm<2"')) "1.x hint"

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
$script:InstallTo = $install
if (-not $IsWindows -and (& id -u) -ne "0") {
    $readOnly = Join-Path $Work "read-only"
    New-Item -ItemType Directory -Path $readOnly | Out-Null
    & chmod 555 $readOnly
    $failed = $false; try { Install-Release $unpacked $readOnly $pinned "x86_64-pc-windows-msvc" } catch { $failed = $true }
    Assert ($failed -and $script:UvRemoved -eq 0) "an unwritable folder fails before uv's markitai goes"
    & chmod 755 $readOnly
}
New-Item -ItemType Directory -Force -Path $install | Out-Null
Set-Content -LiteralPath (Join-Path $install "user-notes.txt") -Value "mine"
Set-Content -LiteralPath (Join-Path $install "LICENSE") -Value "my own"
$failed = $false; try { Install-Release $unpacked $install $pinned "x86_64-pc-windows-msvc" } catch { $failed = $true }
Assert $failed "a first install refuses a folder holding one of its names"
Assert ((Get-Content -Raw (Join-Path $install "LICENSE")).Trim() -eq "my own") "the user's file is kept"
Assert (@(Get-ChildItem -LiteralPath $install -Force).Count -eq 2) "nothing installed beside it"
Remove-Item -LiteralPath (Join-Path $install "LICENSE")
Set-Content -LiteralPath (Join-Path $install "settings.old-2024.json") -Value "keep"
Set-Content -LiteralPath (Join-Path $install ("markitai.exe.old-" + ("0" * 32))) -Value "left by a running exe"
Install-Release $unpacked $install $pinned "x86_64-pc-windows-msvc" | Out-Null
Assert ((Get-Content -Raw (Join-Path $install "user-notes.txt")).Trim() -eq "mine") "files not installed here stay"
Assert (Test-Path (Join-Path $install "settings.old-2024.json")) "only this script's old copies are swept"
Assert ((Get-Content -Raw (Join-Path $install "markitai.exe")) -eq "exe 2.0.1") "new exe"
Assert (@(Get-ChildItem $install -Force | Where-Object { $_.Name -match '\.old-[0-9a-f]{32}$|^\.markitai-' }).Count -eq 0) "old copies and staging gone"
Assert ((Get-Content -Raw (Join-Path $install "receipt")) -match "version=2.0.1") "receipt"
Assert ((Read-Receipt $install).Names -contains "licenses") "receipt lists the folder's names"
Assert ($script:UvRemoved -eq 1) "uv's markitai removed once installing"
$script:FakeUserPath = "Z:\old-usb-tools;$install"
Assert ((Find-FirstMarkitai) -eq [IO.Path]::Combine($install, "markitai.exe")) "a Path entry on a missing drive is skipped"
$script:FakeUserPath = '%USERPROFILE%\bin;C:\Tools'

$second = Join-Path $Work "second"
New-Item -ItemType Directory -Path $second | Out-Null
foreach ($name in @("markitai.exe", "mkai.exe", "markitai-mcp.exe", "NOTICE")) { Set-Content -LiteralPath (Join-Path $second $name) -Value "exe 2.0.2" }
Set-Content -LiteralPath (Join-Path $install "NOTICE") -Value "mine too"
$failed = $false; try { Install-Release $second $install @{ Version = "2.0.2" } "x86_64-pc-windows-msvc" } catch { $failed = $true }
Assert ($failed -and (Get-Content -Raw (Join-Path $install "NOTICE")).Trim() -eq "mine too") "an update refuses a name the receipt does not list"
Assert ((Get-Content -Raw (Join-Path $install "markitai.exe")) -eq "exe 2.0.1") "and keeps the installed release"
Remove-Item -LiteralPath (Join-Path $install "NOTICE")

# A failure halfway through the swap puts the previous release back.
$script:FailMove = "mkai.exe"
function Move-Item {
    [CmdletBinding()] param([string]$LiteralPath, [string]$Destination)
    if ($script:FailMove -and (Split-Path -Leaf $Destination) -eq $script:FailMove) { $script:FailMove = $null; throw "disk full" }
    Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination
}
$failed = $false; try { Install-Release $second $install @{ Version = "2.0.2" } "x86_64-pc-windows-msvc" } catch { $failed = $true }
Assert $failed "the swap failed"
foreach ($name in @("markitai.exe", "mkai.exe")) { Assert ((Get-Content -Raw (Join-Path $install $name)) -eq "exe 2.0.1") "$name restored" }
Assert (Test-Path (Join-Path $install "licenses/example/LICENSE")) "licenses restored"
Assert ((Get-Content -Raw (Join-Path $install "receipt")) -match "version=2.0.1") "receipt restored"
Assert (@(Get-ChildItem $install -Force | Where-Object { $_.Name -match '\.old-[0-9a-f]{32}$|^\.markitai-' }).Count -eq 0) "nothing left aside"
Remove-Item function:Move-Item

Install-Release $second $install @{ Version = "2.0.2" } "x86_64-pc-windows-msvc" | Out-Null
Assert (-not (Test-Path (Join-Path $install "licenses"))) "the previous release's files go"
Assert (Test-Path (Join-Path $install "user-notes.txt")) "still kept"
Assert ((Get-Content -Raw (Join-Path $install "NOTICE")).Trim() -eq "exe 2.0.2") "the new release's names come"

Assert (Add-ToUserPath $install) "added"
Assert ($script:FakeUserPath -eq "$install;%USERPROFILE%\bin;C:\Tools") "prepended, others kept unexpanded"
Assert (-not (Add-ToUserPath ($install + "\"))) "not twice"
Assert (Remove-FromUserPath $install) "removed"
Assert ($script:FakeUserPath -eq '%USERPROFILE%\bin;C:\Tools') "user Path restored"

# The whole flow, with the pieces only Windows has replaced.
function Get-NativeArchitecture { "ARM64" }
function Invoke-MarkitaiVersion { param($Executable) Assert (Test-Path $Executable) "unpacked exe"; "markitai 2.0.0" }
$script:InstallTo = Join-Path $Work "LocalAppData/Programs/markitai"
$env:LOCALAPPDATA = Join-Path $Work "LocalAppData"
$env:MARKITAI_DOWNLOAD_URL = $Base
$env:MARKITAI_INSTALL_OPTIONAL = "0"
$env:MARKITAI_HOME = Join-Path $Work "state"
# An existing configuration: init would start the stand-in "exe", which is no program here.
New-Item -ItemType Directory -Path $env:MARKITAI_HOME | Out-Null
Set-Content -LiteralPath (Join-Path $env:MARKITAI_HOME "config.json") -Value "{}"
Invoke-Install
$programs = Join-Path $env:LOCALAPPDATA "Programs/markitai"
Assert ((Get-Content -Raw (Join-Path $programs "mkai.exe")) -eq "exe 2.0.0") "installed latest"
Assert ($script:FakeUserPath.StartsWith("$programs;")) "on the user Path"
Assert (@(Get-ChildItem ([IO.Path]::GetTempPath()) -Filter "markitai-setup-*").Count -eq 0) "work folder removed"
Assert ((Get-Content -Raw (Join-Path $programs "receipt")) -match "path_added=1") "the receipt says setup added it"
Invoke-Uninstall
Assert (-not (Test-Path $programs)) "uninstalled"
Assert ($script:FakeUserPath -eq '%USERPROFILE%\bin;C:\Tools') "off the user Path"

# A folder already on the user Path, holding the user's files too.
$tools = Join-Path $Work "tools"
New-Item -ItemType Directory -Path $tools | Out-Null
Set-Content -LiteralPath (Join-Path $tools "jq.exe") -Value "jq"
$script:FakeUserPath = "$tools;C:\Tools"
$env:MARKITAI_INSTALL_DIR = $tools
$script:InstallTo = $tools
Invoke-Install
Assert ($script:FakeUserPath -eq "$tools;C:\Tools") "the Path is left as it was"
Assert ((Get-Content -Raw (Join-Path $tools "receipt")) -match "path_added=0") "the receipt says setup did not add it"
$script:Shown.Clear()
Invoke-Uninstall
Assert ($script:FakeUserPath -eq "$tools;C:\Tools") "uninstall leaves an entry setup did not add"
Assert (@(Get-ChildItem -LiteralPath $tools -Force | ForEach-Object { $_.Name }) -join "," -eq "jq.exe") "only markitai's files go"
$shown = $script:Shown -join "`n"
Assert ($shown -match "the folder keeps its other files" -and $shown -match "setup did not add it" -and $shown -notmatch "Removed: ") "uninstall says what it did: $shown"
Remove-Item env:MARKITAI_INSTALL_DIR

$script:LANG_CODE = "zh"
Assert ((Msg "verified") -eq ("$([char]0x5DF2)$([char]0x6821)$([char]0x9A8C) SHA-256")) "zh decoding"
$script:LANG_CODE = "en"
Assert ((Msg "verified") -eq "SHA-256 verified") "en"
"ok"
"""


PS_SESSION = r"""
param([string]$Script)
# setup.ps1 piped into iex, then as a script block with -Uninstall, in this
# scope as in a user's window; nothing is installed, so both only uninstall.
$ErrorActionPreference = "Continue"; $ProgressPreference = "Continue"
$code = [IO.File]::ReadAllText($Script)
$check = '$OnWindows = ($PSVersionTable.PSEdition -ne "Core") -or $IsWindows'
if (-not $code.Contains($check)) { throw "setup.ps1 has no OS check" }
$code = $code.Replace($check, '$OnWindows = $true')
$piped = $null; $block = $null; $functions = $null; $variables = $null
$functions = @(Get-ChildItem function: | ForEach-Object { $_.Name })
$variables = @(Get-Variable | ForEach-Object { $_.Name })
$env:MARKITAI_UNINSTALL = "1"
$piped = $code | Invoke-Expression 6>&1 | Out-String
Remove-Item env:MARKITAI_UNINSTALL
$block = & ([scriptblock]::Create($code)) -Uninstall 6>&1 | Out-String
function Assert($Condition, $Message) { if (-not $Condition) { throw "FAILED: $Message" } }
Assert ($ErrorActionPreference -eq "Continue" -and $ProgressPreference -eq "Continue") "preferences kept"
$left = @(Get-ChildItem function: | Where-Object { $functions -notcontains $_.Name -and $_.Name -ne "Assert" })
Assert ($left.Count -eq 0) "functions left: $($left.Name -join ', ')"
$left = @(Get-Variable | Where-Object { $variables -notcontains $_.Name -and $_.Name -notmatch '^(_|PSItem|left|LASTEXITCODE)$' })
Assert ($left.Count -eq 0) "variables left: $($left.Name -join ', ')"
Assert (-not (Test-Path env:MARKITAI_UNINSTALL)) "MARKITAI_UNINSTALL not set"
foreach ($output in @($piped, $block)) {
    Assert ($output -match "Markitai uninstall" -and $output -match "No markitai installed") "uninstall ran: $output"
}
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

    def test_irm_iex_leaves_the_session_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            harness = work / "session.ps1"
            harness.write_text(PS_SESSION, encoding="utf-8")
            (work / "home").mkdir()
            env = {**os.environ, "HOME": str(work / "home"), "USERPROFILE": str(work / "home"), "TMPDIR": str(work),
                   "LOCALAPPDATA": str(work / "LocalAppData"), "MARKITAI_HOME": str(work / "state")}
            for name in ["MARKITAI_VERSION", "MARKITAI_INSTALL_DIR", "MARKITAI_UNINSTALL"]:
                env.pop(name, None)
            result = subprocess.run(["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(harness),
                                     "-Script", str(SETUP_PS1)], capture_output=True, text=True, timeout=180, env=env)
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
