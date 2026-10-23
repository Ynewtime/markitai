# Markitai setup: installs the markitai executable (2.x) from a GitHub Release.
# Windows PowerShell 5.1 and PowerShell 7, x64 and ARM64; English or Chinese
# after the culture.
#
# Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"   # install or update
#   & ([scriptblock]::Create((irm https://markitai.dev/setup.ps1))) -Uninstall         # remove it
#   .\setup.ps1 [-Uninstall]
#
# It downloads markitai-v<version>-<target>.zip and the release's SHA256SUMS,
# checks the archive against it, and installs markitai.exe, mkai.exe and
# markitai-mcp.exe with the license notices and offline guides into
# %LOCALAPPDATA%\Programs\markitai, which it adds to the user Path. No
# administrator rights needed.
#
# Environment:
#   MARKITAI_VERSION=X.Y.Z       install this release (default: the latest)
#   MARKITAI_INSTALL_DIR=DIR     install folder (default: %LOCALAPPDATA%\Programs\markitai)
#   MARKITAI_INSTALL_OPTIONAL=1  also prepare the optional components
#                                (markitai doctor --fix: local OCR models and,
#                                when no browser is found, a headless Chromium)
#                                without a terminal to ask; 0 never prepares them
#   MARKITAI_DOWNLOAD_URL=URL    releases base, for a mirror
#                                (default: https://github.com/Ynewtime/markitai/releases)
#   MARKITAI_UNINSTALL=1         the same as -Uninstall
#   NO_COLOR=1                   no colours; the marks stay
#
# Keep this file ASCII-only (a test enforces it): Windows PowerShell 5.1 reads
# a script without a BOM in the ANSI code page, and irm decodes the download
# as ISO-8859-1, so any other byte arrives as mojibake.

param([switch]$Uninstall)

# Everything below runs in this script block's own scope, so `irm | iex` in
# the user's session leaves no preference, function or variable behind
# (hence no $script: variables: under iex that scope is the user's).
& {
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
# Invoke-WebRequest's progress bar slows Windows PowerShell 5.1 downloads tenfold.
$ProgressPreference = "SilentlyContinue"

$RepositoryReleases = "https://github.com/Ynewtime/markitai/releases"
$ScriptUrl = "https://markitai.dev/setup.ps1"
$Targets = @{ "AMD64" = "x86_64-pc-windows-msvc"; "ARM64" = "aarch64-pc-windows-msvc" }
$Executables = @("markitai.exe", "mkai.exe", "markitai-mcp.exe")

# Windows PowerShell 5.1 has no $IsWindows; it only runs on Windows.
$OnWindows = ($PSVersionTable.PSEdition -ne "Core") -or $IsWindows

# ============================================================
# Terminal capabilities (the rules the 1.x script and markitai follow)
# ============================================================

# The output code page of the console this script runs in. When this
# script's output is redirected, its reader decodes it with that code page,
# so the marks follow it. 0: no console attached.
function Get-ConsoleOutputCodePage {
    try {
        if (-not ("MarkitaiSetup.Kernel32" -as [type])) {
            Add-Type -Namespace MarkitaiSetup -Name Kernel32 -MemberDefinition (
                '[System.Runtime.InteropServices.DllImport("kernel32.dll")] ' +
                'public static extern uint GetConsoleOutputCP();')
        }
        return [int][MarkitaiSetup.Kernel32]::GetConsoleOutputCP()
    } catch {
        try { return [Console]::OutputEncoding.CodePage } catch { return 0 }
    }
}

function Test-OutputRedirected {
    try { return [Console]::IsOutputRedirected } catch { return $false }
}

$ConsoleOutputCodePage = 0
if ($OnWindows -and (Test-OutputRedirected)) { $ConsoleOutputCodePage = Get-ConsoleOutputCodePage }

# Windows Terminal, VS Code, ConEmu and SSH clients draw Unicode and CJK with
# font fallback; the legacy console host does not.
$ModernTerminal = [bool](
    $env:WT_SESSION -or $env:TERM_PROGRAM -or ($env:ConEmuANSI -eq "ON") -or
    $env:SSH_CONNECTION -or $env:SSH_CLIENT
)

function Test-UnicodeMarks {
    if ($env:TERM -eq "dumb") { return $false }
    if (-not $OnWindows) { return $true }
    if (Test-OutputRedirected) {
        return (@(0, 65001) -contains $ConsoleOutputCodePage)
    }
    if ($ModernTerminal) { return $true }
    try { return [bool]$Host.UI.SupportsVirtualTerminal } catch { return $false }
}

function Test-UseColor {
    if ($env:NO_COLOR) { return $false }
    if ($env:TERM -eq "dumb") { return $false }
    if ($env:FORCE_COLOR -and $env:FORCE_COLOR -notmatch '^(0|false)$') { return $true }
    return -not (Test-OutputRedirected)
}

# The legacy console host draws Chinese only with a CJK or UTF-8 code page.
function Test-CjkCapableOutput {
    if (-not $OnWindows -or $ModernTerminal) { return $true }
    if (Test-OutputRedirected) { return $true }
    $codePage = 0
    try { $codePage = [Console]::OutputEncoding.CodePage } catch {}
    return (@(932, 936, 949, 950, 54936, 65001) -contains $codePage)
}

function Get-Lang {
    $culture = ""
    try { $culture = (Get-Culture).Name } catch {}
    if ($culture -match "^zh" -and (Test-CjkCapableOutput)) { return "zh" }
    return "en"
}

# Decode the \uXXXX escapes of the Chinese table (a plain loop: identical on
# PowerShell 5.1 and 7).
function ConvertFrom-CodePointEscape {
    param([string]$Text)
    $builder = New-Object System.Text.StringBuilder
    $position = 0
    foreach ($match in [regex]::Matches($Text, '\\u([0-9a-fA-F]{4})')) {
        [void]$builder.Append($Text.Substring($position, $match.Index - $position))
        [void]$builder.Append([char][Convert]::ToInt32($match.Groups[1].Value, 16))
        $position = $match.Index + $match.Length
    }
    [void]$builder.Append($Text.Substring($position))
    return $builder.ToString()
}

$UseColor = Test-UseColor
if (Test-UnicodeMarks) {
    $S = @{ Bar = [string][char]0x2502; Top = [string][char]0x256D + [char]0x2500; Branch = [string][char]0x251C + [char]0x2500
        Bottom = [string][char]0x2570 + [char]0x2500; Check = [string][char]0x2713; Cross = [string][char]0x2717
        Arrow = [string][char]0x2192; Dot = [string][char]0x2022; Circle = [string][char]0x25CB }
} else {
    $S = @{ Bar = "|"; Top = "+-"; Branch = "+-"; Bottom = "+-"; Check = "+"; Cross = "x"; Arrow = ">"; Dot = "-"; Circle = "o" }
}
$LANG_CODE = Get-Lang

# Messages; setup.sh carries the same (a test compares the Chinese tables).
function Msg {
    param([string]$Key)
    if ($LANG_CODE -eq "zh") {
        $text = switch ($Key) {
            "configure_llm" { '\u914d\u7f6e LLM' }
            "confirm_optional" { '\u73b0\u5728\u51c6\u5907\u53ef\u9009\u7ec4\u4ef6? (\u672c\u5730 OCR \u6a21\u578b\u7ea6 45 MB\uff1b\u627e\u4e0d\u5230\u6d4f\u89c8\u5668\u65f6\u4e0b\u8f7d\u65e0\u5934 Chromium\uff0c\u7ea6 280 MB)' }
            "convert_file" { '\u8f6c\u6362\u6587\u4ef6' }
            "docs" { '\u6587\u6863' }
            "done_uninstall" { 'markitai \u5df2\u5378\u8f7d' }
            "done_warnings" { 'markitai \u5df2\u5c31\u7eea\uff0c\u4f46\u6709\u90e8\u5206\u7ec4\u4ef6\u5931\u8d25' }
            "downloading" { '\u6b63\u5728\u4e0b\u8f7d' }
            "error_arch" { '\u4e0d\u652f\u6301\u6b64\u5904\u7406\u5668\u67b6\u6784' }
            "error_checksum" { 'SHA-256 \u4e0d\u5339\u914d\uff0c\u4e0b\u8f7d\u7684\u6587\u4ef6\u5df2\u5220\u9664' }
            "error_download" { '\u4e0b\u8f7d\u5931\u8d25' }
            "error_install" { '\u65e0\u6cd5\u5b89\u88c5\u5230\u6b64\u76ee\u5f55' }
            "error_no_asset" { '\u8be5\u7248\u672c\u6ca1\u6709\u6b64\u5e73\u53f0\u7684\u5f52\u6863' }
            "error_os_windows" { '\u6b64\u811a\u672c\u7528\u4e8e Windows\uff1bmacOS \u548c Linux \u8bf7\u4f7f\u7528 setup.sh' }
            "error_python_version" { 'markitai 1.x \u662f Python \u5305\uff1b\u8bf7\u7528 uv tool install \u6216 pip \u5b89\u88c5\u8be5\u7248\u672c' }
            "error_run" { '\u4e0b\u8f7d\u7684 markitai \u65e0\u6cd5\u5728\u6b64\u7cfb\u7edf\u8fd0\u884c' }
            "error_shared_folder" { '\u6b64\u76ee\u5f55\u5df2\u6709\u4e0e markitai \u5b89\u88c5\u6587\u4ef6\u540c\u540d\u7684\u6587\u4ef6\uff1b\u8bf7\u4f7f\u7528\u7a7a\u76ee\u5f55\u6216\u4e13\u7528\u76ee\u5f55' }
            "error_version" { 'MARKITAI_VERSION \u5e94\u4e3a X.Y.Z \u5f62\u5f0f' }
            "failed" { '\u5b89\u88c5\u5931\u8d25' }
            "files_removed" { '\u5df2\u5220\u9664 markitai \u7684\u6587\u4ef6\uff1b\u76ee\u5f55\u4e2d\u7684\u5176\u4ed6\u6587\u4ef6\u4fdd\u7559' }
            "installed" { '\u5df2\u5b89\u88c5' }
            "interactive" { '\u4ea4\u4e92\u6a21\u5f0f' }
            "nothing_installed" { '\u672a\u627e\u5230\u5b89\u88c5\u811a\u672c\u5b89\u88c5\u7684 markitai' }
            "old_in_use" { '\u65e7\u7684 markitai.exe \u4ecd\u5728\u8fd0\u884c\uff1b\u5b83\u7684\u6587\u4ef6\u5c06\u5728\u4e0b\u6b21\u66f4\u65b0\u65f6\u5220\u9664' }
            "optional_failed" { '\u53ef\u9009\u7ec4\u4ef6\u672a\u80fd\u5168\u90e8\u51c6\u5907\u597d\uff1b\u7a0d\u540e\u53ef\u518d\u8fd0\u884c' }
            "optional_headless" { '\u6ca1\u6709\u7ec8\u7aef\uff0c\u53ef\u9009\u7ec4\u4ef6\u5df2\u8df3\u8fc7\uff1b\u8981\u4e00\u5e76\u51c6\u5907\uff0c\u8bbe\u7f6e\u6b64\u53d8\u91cf\u540e\u91cd\u65b0\u8fd0\u884c\u5b89\u88c5\u811a\u672c' }
            "optional_later" { '\u7a0d\u540e\u51c6\u5907\u53ef\u9009\u7ec4\u4ef6' }
            "optional_ready" { '\u53ef\u9009\u7ec4\u4ef6\u5df2\u5c31\u7eea' }
            "path_added" { '\u5df2\u52a0\u5165\u7528\u6237 Path\uff1b\u65b0\u7ec8\u7aef\u53ef\u76f4\u63a5\u8fd0\u884c markitai' }
            "path_kept" { '\u8be5\u76ee\u5f55\u4ecd\u5728\u7528\u6237 Path \u4e2d\uff1b\u5b83\u4e0d\u662f\u5b89\u88c5\u811a\u672c\u6dfb\u52a0\u7684' }
            "path_removed" { '\u5df2\u4ece\u7528\u6237 Path \u4e2d\u79fb\u9664' }
            "platform" { '\u5e73\u53f0' }
            "ready" { 'markitai \u5df2\u5c31\u7eea' }
            "release" { '\u7248\u672c' }
            "removed" { '\u5df2\u5220\u9664' }
            "replaced_python" { '\u5df2\u79fb\u9664 uv \u5b89\u88c5\u7684 Python \u7248 markitai' }
            "section_install" { '\u5b89\u88c5' }
            "section_optional" { '\u53ef\u9009\u7ec4\u4ef6' }
            "section_start" { '\u5f00\u59cb\u4f7f\u7528' }
            "shadowed" { 'PATH \u4e2d\u66f4\u9760\u524d\u7684\u53e6\u4e00\u4e2a markitai \u4f1a\u88ab\u4f18\u5148\u4f7f\u7528' }
            "show_help" { '\u67e5\u770b\u5e2e\u52a9' }
            "state_kept" { '\u914d\u7f6e\u3001\u7f13\u5b58\u548c\u6a21\u578b\u4ecd\u5728\u6b64\u76ee\u5f55\uff1b\u4e0d\u518d\u9700\u8981\u65f6\u53ef\u5220\u9664' }
            "uninstall" { '\u5378\u8f7d' }
            "update" { '\u66f4\u65b0' }
            "verified" { '\u5df2\u6821\u9a8c SHA-256' }
            "warn_http" { '\u4e0b\u8f7d\u5730\u5740\u4e0d\u662f HTTPS' }
            "web_ui" { '\u542f\u52a8 Web UI' }
            "welcome" { 'Markitai \u5b89\u88c5' }
            "welcome_uninstall" { 'Markitai \u5378\u8f7d' }
            default { $null }
        }
        if ($text) { return (ConvertFrom-CodePointEscape $text) }
    }
    switch ($Key) {
        "welcome" { "Markitai setup" }
        "welcome_uninstall" { "Markitai uninstall" }
        "ready" { "markitai is ready" }
        "done_warnings" { "markitai is ready, but some components failed" }
        "done_uninstall" { "markitai is uninstalled" }
        "failed" { "Setup failed" }
        "section_install" { "Install" }
        "section_optional" { "Optional components" }
        "section_start" { "Getting started" }
        "platform" { "Platform" }
        "release" { "Release" }
        "downloading" { "Downloading" }
        "verified" { "SHA-256 verified" }
        "installed" { "Installed" }
        "replaced_python" { "Removed the Python markitai that uv installed" }
        "confirm_optional" { "Prepare the optional components now? (local OCR models, about 45 MB; a headless Chromium, about 280 MB, when no browser is found)" }
        "optional_ready" { "Optional components are ready" }
        "optional_failed" { "Not every optional component could be prepared; run this again later" }
        "optional_headless" { "No terminal to ask, so optional components were skipped; to prepare them too, set this and run setup again" }
        "optional_later" { "Prepare optional components later" }
        "path_added" { "Added to the user Path; new terminals find markitai" }
        "path_removed" { "Removed from the user Path" }
        "path_kept" { "The folder stays on the user Path; setup did not add it" }
        "files_removed" { "Removed markitai's files; the folder keeps its other files" }
        "shadowed" { "Another markitai comes first on PATH and runs instead" }
        "convert_file" { "Convert a file" }
        "interactive" { "Interactive mode" }
        "web_ui" { "Start Web UI" }
        "configure_llm" { "Configure LLM" }
        "show_help" { "Show help" }
        "update" { "Update" }
        "uninstall" { "Uninstall" }
        "removed" { "Removed" }
        "nothing_installed" { "No markitai installed by this script was found" }
        "state_kept" { "Configuration, cache and models stay in this folder; delete it when no longer needed" }
        "old_in_use" { "An older markitai.exe is still running; its file goes at the next update" }
        "docs" { "Docs" }
        "error_os_windows" { "This script is for Windows; on macOS and Linux use setup.sh" }
        "error_arch" { "This processor architecture is not supported" }
        "error_download" { "Download failed" }
        "error_no_asset" { "This release has no archive for this platform" }
        "error_checksum" { "SHA-256 mismatch; the download was deleted" }
        "error_run" { "The downloaded markitai does not run on this system" }
        "error_python_version" { "markitai 1.x is the Python package; install that release with uv tool install or pip" }
        "error_version" { "MARKITAI_VERSION must look like X.Y.Z" }
        "error_shared_folder" { "This folder already holds files of the names markitai installs; use an empty or dedicated folder" }
        "error_install" { "Could not install into this folder" }
        "warn_http" { "The download address is not HTTPS" }
        default { $Key }
    }
}

# ============================================================
# Output: the tree-style rows of the 1.x setup scripts
# ============================================================

function Write-Part {
    param([string]$Text, [string]$Color, [switch]$NoNewline)
    if ($UseColor -and $Color) {
        Write-Host $Text -ForegroundColor $Color -NoNewline:$NoNewline
    } else {
        Write-Host $Text -NoNewline:$NoNewline
    }
}

function Write-Item {
    param([string]$Mark, [string]$Color, [string]$Text)
    Write-Part $S.Bar "DarkGray" -NoNewline
    Write-Part "  $Mark" $Color -NoNewline
    Write-Host " $Text"
}

function Write-Intro { param([string]$Text) Write-Host ""; Write-Part $S.Top "DarkGray" -NoNewline; Write-Host " $Text" }
function Write-Section { param([string]$Text) Write-Part $S.Bar "DarkGray"; Write-Part $S.Branch "DarkGray" -NoNewline; Write-Host " $Text" }
function Write-Ok { param([string]$Text) Write-Item $S.Check "DarkGreen" $Text }
function Write-Warn { param([string]$Text) Write-Item "!" "DarkYellow" $Text }
function Write-Info { param([string]$Text) Write-Item $S.Dot "DarkGray" $Text }
function Write-Step { param([string]$Text) Write-Item $S.Arrow "DarkCyan" $Text }
function Write-Skip { param([string]$Text) Write-Item $S.Circle "DarkGray" $Text }
function Write-Command { param([string]$Text) Write-Part $S.Bar "DarkGray" -NoNewline; Write-Part "    $Text" "DarkCyan" }
function Write-Hint { param([string]$Label, [string]$Command) Write-Info "${Label}:"; Write-Command $Command }
function Write-Outro {
    param([string]$Color, [string]$Text)
    Write-Part $S.Bar "DarkGray"
    Write-Part $S.Bottom "DarkGray" -NoNewline
    Write-Part " $Text" $Color
    Write-Host ""
}

# A failure ends setup: a terminating error, so `irm | iex` does not close
# the user's own window as `exit` would.
function Stop-Setup {
    param([string]$Text, [string]$Detail)
    Write-Item $S.Cross "DarkRed" $Text
    if ($Detail) { Write-Part $S.Bar "DarkGray" -NoNewline; Write-Part "    $Detail" "DarkGray" }
    Write-Outro "DarkRed" (Msg "failed")
    throw "markitai setup: $Text"
}

function Test-Truthy { param([string]$Value) return $Value -match '^(1|true|yes|on)$' }
function Test-Falsy { param([string]$Value) return $Value -match '^(0|false|no|off)$' }

function Test-InteractiveInput {
    if ($env:CI) { return $false }
    try { return -not [Console]::IsInputRedirected } catch { return $false }
}

function Read-Confirm {
    param([string]$Question, [bool]$Default)
    $hint = if ($Default) { "[Y/n]" } else { "[y/N]" }
    Write-Part $S.Bar "DarkGray" -NoNewline
    Write-Part "  ?" "DarkCyan" -NoNewline
    Write-Host " $Question " -NoNewline
    Write-Part "$hint " "DarkGray" -NoNewline
    $answer = Read-Host
    if (-not $answer) { return $Default }
    return $answer -match '^[Yy]'
}

# ============================================================
# Platform, downloads, checksums
# ============================================================

# The machine's architecture, from the registry: an x64 PowerShell emulated on
# ARM64 sees AMD64 in its own PROCESSOR_ARCHITECTURE.
function Get-NativeArchitecture {
    $arch = $null
    try {
        $arch = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" -ErrorAction Stop).PROCESSOR_ARCHITECTURE
    } catch {}
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITEW6432 }
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    return "$arch".ToUpperInvariant()
}

function Get-ReleaseTarget {
    param([string]$Architecture)
    $target = $Targets[$Architecture]
    if (-not $target) { Stop-Setup (Msg "error_arch") $Architecture }
    return $target
}

function Save-Download {
    param([string]$Url, [string]$Path)
    Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
}

# The release named by MARKITAI_VERSION, else the latest, from its SHA256SUMS:
# @{ Version; Archive; Sha256 }.
function Resolve-Release {
    param([string]$Base, [string]$Target, [string]$Wanted, [string]$Work)
    if ($Wanted) {
        $Wanted = $Wanted -replace '^v', ''
        if ($Wanted -match '^[01]\.') { Stop-Setup (Msg "error_python_version") "uv tool install markitai==$Wanted" }
        if ($Wanted -notmatch '^\d+\.\d+\.\d+') { Stop-Setup (Msg "error_version") $Wanted }
        $sumsUrl = "$Base/download/v$Wanted/SHA256SUMS"
    } else {
        $sumsUrl = "$Base/latest/download/SHA256SUMS"
    }
    $sums = Join-Path $Work "SHA256SUMS"
    try { Save-Download $sumsUrl $sums } catch { Stop-Setup (Msg "error_download") "$sumsUrl ($($_.Exception.Message))" }
    $pattern = '^([0-9a-f]{64})  (markitai-v(\S+)-' + [regex]::Escape($Target) + '\.zip)$'
    foreach ($line in [IO.File]::ReadAllLines($sums)) {
        $match = [regex]::Match($line, $pattern)
        if ($match.Success) {
            $release = @{ Sha256 = $match.Groups[1].Value; Archive = $match.Groups[2].Value; Version = $match.Groups[3].Value }
            if ($Wanted -and $release.Version -ne $Wanted) { Stop-Setup (Msg "error_no_asset") $release.Archive }
            return $release
        }
    }
    Stop-Setup (Msg "error_no_asset") "$Target ($sumsUrl)"
}

function Test-Checksum {
    param([string]$Path, [string]$Expected)
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        Remove-Item -LiteralPath $Path -Force
        Stop-Setup (Msg "error_checksum") "${actual} != $Expected"
    }
}

# Members are checked before anything is written: plain relative names, the
# three executables at the root.
function Expand-Release {
    param([string]$Zip, [string]$Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $names = @($archive.Entries | ForEach-Object { $_.FullName })
        foreach ($name in $names) {
            if ($name -match '(^|[\\/])\.\.([\\/]|$)' -or $name -match '^[\\/]' -or $name -match ':') {
                Stop-Setup (Msg "error_run") "unsafe archive member $name"
            }
        }
        foreach ($name in $Executables) {
            if ($names -notcontains $name) { Stop-Setup (Msg "error_run") "$name is missing from the archive" }
        }
    } finally {
        $archive.Dispose()
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Destination)
}

# markitai.exe --version from the unpacked release; proves it runs here.
function Invoke-MarkitaiVersion {
    param([string]$Executable)
    # Under "Stop", Windows PowerShell 5.1 turns a native program's stderr into an error.
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Executable --version 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($code -ne 0) { throw "exit ${code}: $output" }
    return "$output".Trim()
}

# ============================================================
# Install, update, uninstall
# ============================================================

# Python markitai 1.x installed with uv put markitai.exe, mkai.exe and
# markitai-mcp.exe in uv's own folder, which can come first on Path.
function Remove-PythonMarkitai {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { return }
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $tools = & uv tool list 2>$null
        if (-not ($tools -match '^markitai ')) { return }
        & uv tool uninstall markitai 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Ok (Msg "replaced_python") } else { Write-Warn "uv tool uninstall markitai" }
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# The receipt names what this script installed in the folder: the release,
# whether setup added the folder to the user Path (path_added=1, so
# uninstall leaves an entry the user made alone), and the folder's top-level names (file=...). Only those are ever replaced or
# removed, so a MARKITAI_INSTALL_DIR shared with other files keeps them.
# $null when the folder holds no receipt of this script's.
function Read-Receipt {
    param([string]$Directory)
    $path = [IO.Path]::Combine($Directory, "receipt")
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    $lines = @([IO.File]::ReadAllLines($path))
    if (-not @($lines | Where-Object { $_ -like "version=*" })) { return $null }
    $names = @($lines | Where-Object { $_ -like "file=*" } | ForEach-Object { $_.Substring(5) } |
        Where-Object { $_ -and $_ -notmatch '[\\/:]' -and $_ -ne ".." -and $_ -ne "." -and $_ -ne "receipt" })
    return @{ Names = $names; PathAdded = ($lines -contains "path_added=1") }
}

function Write-Receipt {
    param([string]$Directory, [hashtable]$Receipt)
    $pathAdded = 0
    if ($Receipt.PathAdded) { $pathAdded = 1 }
    $lines = @("version=$($Receipt.Version)", "target=$($Receipt.Target)", "path_added=$pathAdded") +
        @($Receipt.Names | ForEach-Object { "file=$_" })
    [IO.File]::WriteAllText([IO.Path]::Combine($Directory, "receipt"), (($lines -join "`n") + "`n"))
}

# The release's names (and "receipt") that the folder already holds but the
# receipt does not list: the user's or another program's, never replaced.
function Get-ForeignNames {
    param([string]$Directory, [string[]]$Names, $Receipt)
    $owned = @()
    if ($Receipt) { $owned = @($Receipt.Names) + @("receipt") }
    return @(@($Names) + @("receipt") | Where-Object {
        $owned -notcontains $_ -and (Test-Path -LiteralPath ([IO.Path]::Combine($Directory, $_)))
    })
}

# Old copies an earlier run renamed aside (a running executable cannot be
# deleted, only renamed) and a staging folder an interrupted run left behind:
# exactly the names this script creates.
$LeftoverPattern = '^((markitai|mkai|markitai-mcp)\.exe\.old-|\.markitai-new-)[0-9a-f]{32}$'

function Remove-Leftovers {
    param([string]$Directory)
    $left = $false
    foreach ($item in @(Get-ChildItem -LiteralPath $Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $LeftoverPattern })) {
        try { Remove-Item -LiteralPath $item.FullName -Recurse -Force } catch { $left = $true }
    }
    return $left
}

# Renames each name aside, then deletes it. Returns $true when something
# (a running executable) could not be deleted yet.
function Remove-Names {
    param([string]$Directory, [string[]]$Names)
    $id = [Guid]::NewGuid().ToString("N")
    $inUse = $false
    foreach ($name in @($Names)) {
        $path = [IO.Path]::Combine($Directory, $name)
        if (-not (Test-Path -LiteralPath $path)) { continue }
        Move-Item -LiteralPath $path -Destination "$path.old-$id"
        try { Remove-Item -LiteralPath "$path.old-$id" -Recurse -Force } catch { $inUse = $true }
    }
    if (Remove-Leftovers $Directory) { $inUse = $true }
    return $inUse
}

# The release is first copied into a staging folder inside the install
# folder: that proves the folder is writable and has room before anything
# installed is touched, and keeps the swap to renames on one volume (a
# sibling folder could sit where the user may not write).
function New-Staging {
    param([string]$Unpacked, [string]$Directory, [hashtable]$Release, [string]$Target, $Previous)
    $names = @(Get-ChildItem -LiteralPath $Unpacked -Force | ForEach-Object { $_.Name })
    $foreign = @(Get-ForeignNames $Directory $names $Previous)
    if ($foreign.Count) { Stop-Setup (Msg "error_shared_folder") "${Directory}: $($foreign -join ', ')" }
    $created = -not (Test-Path -LiteralPath $Directory)
    $stage = [IO.Path]::Combine($Directory, ".markitai-new-" + [Guid]::NewGuid().ToString("N"))
    $receipt = @{ Version = $Release.Version; Target = $Target; Names = $names; PathAdded = [bool]($Previous -and $Previous.PathAdded) }
    try {
        New-Item -ItemType Directory -Force -Path $stage | Out-Null
        foreach ($name in $names) {
            Copy-Item -LiteralPath ([IO.Path]::Combine($Unpacked, $name)) -Destination ([IO.Path]::Combine($stage, $name)) -Recurse -Force
        }
        Write-Receipt $stage $receipt
    } catch {
        $problem = $_.Exception.Message
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        if ($created) { Remove-Item -LiteralPath $Directory -Recurse -Force -ErrorAction SilentlyContinue }
        Stop-Setup (Msg "error_install") $problem
    }
    return @{ Path = $stage; Receipt = $receipt }
}

# Swaps the staged release in: the previous release's names and the new
# ones are renamed aside, the staged ones renamed into place, and on any
# error everything goes back as it was, the previous receipt included.
function Complete-Install {
    param([hashtable]$Stage, [string]$Directory, $Previous)
    $id = [Guid]::NewGuid().ToString("N")
    $names = @($Stage.Receipt.Names) + @("receipt")
    $old = @()
    if ($Previous) { $old = @($Previous.Names) }
    $aside = New-Object System.Collections.ArrayList
    $placed = New-Object System.Collections.ArrayList
    try {
        foreach ($name in @(@($old) + $names | Select-Object -Unique)) {
            $path = [IO.Path]::Combine($Directory, $name)
            if (-not (Test-Path -LiteralPath $path)) { continue }
            Move-Item -LiteralPath $path -Destination "$path.old-$id"
            [void]$aside.Add($name)
        }
        foreach ($name in $names) {
            Move-Item -LiteralPath ([IO.Path]::Combine($Stage.Path, $name)) -Destination ([IO.Path]::Combine($Directory, $name))
            [void]$placed.Add($name)
        }
    } catch {
        $problem = $_.Exception.Message
        foreach ($name in $placed) {
            Remove-Item -LiteralPath ([IO.Path]::Combine($Directory, $name)) -Recurse -Force -ErrorAction SilentlyContinue
        }
        foreach ($name in $aside) {
            $path = [IO.Path]::Combine($Directory, $name)
            Move-Item -LiteralPath "$path.old-$id" -Destination $path -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $Stage.Path -Recurse -Force -ErrorAction SilentlyContinue
        Stop-Setup (Msg "error_install") $problem
    }
    Remove-Item -LiteralPath $Stage.Path -Recurse -Force -ErrorAction SilentlyContinue
    $inUse = $false
    foreach ($name in $aside) {
        try { Remove-Item -LiteralPath ([IO.Path]::Combine($Directory, "$name.old-$id")) -Recurse -Force } catch { $inUse = $true }
    }
    if (Remove-Leftovers $Directory) { $inUse = $true }
    if ($inUse) { Write-Info (Msg "old_in_use") }
    return $Stage.Receipt
}

# Returns the receipt of what is installed now.
function Install-Release {
    param([string]$Unpacked, [string]$Directory, [hashtable]$Release, [string]$Target)
    $previous = Read-Receipt $Directory
    $stage = New-Staging $Unpacked $Directory $Release $Target $previous
    # Only now, with the new release staged, does uv's Python markitai go.
    Remove-PythonMarkitai
    return (Complete-Install $stage $Directory $previous)
}

# The user Path lives in HKCU\Environment as REG_EXPAND_SZ: written back as
# that kind so entries such as %USERPROFILE%\bin keep expanding
# ([Environment]::SetEnvironmentVariable would store REG_SZ).
function Get-UserPath {
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment")
    try {
        if (-not $key) { return "" }
        return [string]$key.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    } finally {
        if ($key) { $key.Dispose() }
    }
}

function Set-UserPath {
    param([string]$Value)
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey("Environment")
    try {
        $key.SetValue("Path", $Value, [Microsoft.Win32.RegistryValueKind]::ExpandString)
    } finally {
        $key.Dispose()
    }
    Send-EnvironmentChange
}

# Tell Explorer (and terminals it starts) that the environment changed.
function Send-EnvironmentChange {
    try {
        if (-not ("MarkitaiSetup.User32" -as [type])) {
            Add-Type -Namespace MarkitaiSetup -Name User32 -MemberDefinition (
                '[System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] ' +
                'public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);')
        }
        $result = [UIntPtr]::Zero
        [void][MarkitaiSetup.User32]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, "Environment", 2, 5000, [ref]$result)
    } catch {}
}

function Split-PathList {
    param([string]$Value)
    return @($Value -split ';' | Where-Object { $_ -ne "" })
}

function Test-SamePath {
    param([string]$Left, [string]$Right)
    $expand = { param($p) [Environment]::ExpandEnvironmentVariables($p).TrimEnd('\', '/') }
    return [string]::Equals((& $expand $Left), (& $expand $Right), [StringComparison]::OrdinalIgnoreCase)
}

# Returns $true when the folder was added.
function Add-ToUserPath {
    param([string]$Directory)
    $entries = Split-PathList (Get-UserPath)
    $changed = $false
    if (-not ($entries | Where-Object { Test-SamePath $_ $Directory })) {
        Set-UserPath ((@($Directory) + $entries) -join ';')
        $changed = $true
    }
    if (-not (Split-PathList $env:Path | Where-Object { Test-SamePath $_ $Directory })) {
        $env:Path = "$Directory;$env:Path"
    }
    return $changed
}

function Remove-FromUserPath {
    param([string]$Directory)
    $entries = Split-PathList (Get-UserPath)
    $kept = @($entries | Where-Object { -not (Test-SamePath $_ $Directory) })
    if ($kept.Count -eq $entries.Count) { return $false }
    Set-UserPath ($kept -join ';')
    return $true
}

# The markitai a new terminal runs: the machine Path comes before the user's.
function Find-FirstMarkitai {
    $machine = ""
    try { $machine = [Environment]::GetEnvironmentVariable("Path", "Machine") } catch {}
    foreach ($entry in @((Split-PathList $machine) + (Split-PathList (Get-UserPath)))) {
        $folder = [Environment]::ExpandEnvironmentVariables($entry)
        foreach ($name in @("markitai.exe", "markitai.cmd", "markitai.bat")) {
            $candidate = Join-Path $folder $name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }
    return $null
}

function Get-DefaultInstallDirectory {
    if ($env:MARKITAI_INSTALL_DIR) { return $env:MARKITAI_INSTALL_DIR }
    return (Join-Path $env:LOCALAPPDATA "Programs\markitai")
}

function Get-StateDirectory {
    if ($env:MARKITAI_HOME) { return $env:MARKITAI_HOME }
    return (Join-Path $HOME ".markitai")
}

function Invoke-OptionalComponents {
    param([string]$Markitai)
    Write-Section (Msg "section_optional")
    $requested = "$env:MARKITAI_INSTALL_OPTIONAL"
    if (Test-Falsy $requested) {
        Write-Hint (Msg "optional_later") "markitai doctor --fix"
        return $true
    }
    if (-not (Test-Truthy $requested)) {
        if (-not (Test-InteractiveInput)) {
            Write-Skip (Msg "optional_headless")
            Write-Command '$env:MARKITAI_INSTALL_OPTIONAL = "1"'
            return $true
        }
        if (-not (Read-Confirm (Msg "confirm_optional") $true)) {
            Write-Hint (Msg "optional_later") "markitai doctor --fix"
            return $true
        }
    }
    Write-Step "markitai doctor --fix"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Markitai doctor --fix *> $null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($code -eq 0) {
        Write-Ok (Msg "optional_ready")
        return $true
    }
    Write-Warn (Msg "optional_failed")
    Write-Command "markitai doctor --fix"
    return $false
}

function Write-GettingStarted {
    param([string]$Directory)
    Write-Section (Msg "section_start")
    $first = Find-FirstMarkitai
    if ($first -and -not (Test-SamePath (Split-Path -Parent $first) $Directory)) {
        Write-Warn "$(Msg 'shadowed'): $first"
    }
    Write-Hint (Msg "convert_file") "markitai file.pdf"
    Write-Hint (Msg "interactive") "markitai -I"
    Write-Hint (Msg "web_ui") "markitai serve"
    Write-Hint (Msg "configure_llm") "markitai init"
    Write-Hint (Msg "show_help") "markitai --help"
    Write-Hint (Msg "update") "irm $ScriptUrl | iex"
    # A script block takes -Uninstall and, unlike setting MARKITAI_UNINSTALL,
    # leaves nothing set in this window for a later update to trip over.
    Write-Hint (Msg "uninstall") "& ([scriptblock]::Create((irm $ScriptUrl))) -Uninstall"
    Write-Info "$(Msg 'docs'): https://markitai.dev"
}

function Invoke-Uninstall {
    Write-Intro (Msg "welcome_uninstall")
    $directory = Get-DefaultInstallDirectory
    $receipt = Read-Receipt $directory
    if ($receipt) {
        if (Remove-Names $directory (@($receipt.Names) + @("receipt"))) { Write-Info (Msg "old_in_use") }
        if (-not @(Get-ChildItem -LiteralPath $directory -Force)) {
            Remove-Item -LiteralPath $directory -Force
            Write-Ok "$(Msg 'removed'): $directory"
        } else {
            Write-Ok "$(Msg 'files_removed'): $directory"
        }
        if ($receipt.PathAdded) {
            if (Remove-FromUserPath $directory) { Write-Ok (Msg "path_removed") }
        } elseif (@(Split-PathList (Get-UserPath) | Where-Object { Test-SamePath $_ $directory })) {
            Write-Info (Msg "path_kept")
        }
    } else {
        Write-Info (Msg "nothing_installed")
    }
    $state = Get-StateDirectory
    if (Test-Path -LiteralPath $state) { Write-Info "$(Msg 'state_kept'): $state" }
    Write-Outro "DarkGreen" (Msg "done_uninstall")
}

function Invoke-Install {
    Write-Intro (Msg "welcome")
    $base = if ($env:MARKITAI_DOWNLOAD_URL) { $env:MARKITAI_DOWNLOAD_URL.TrimEnd('/') } else { $RepositoryReleases }
    if ($base -notmatch '^https://') { Write-Warn "$(Msg 'warn_http'): $base" }
    # Windows PowerShell 5.1 on older .NET may not offer TLS 1.2 by default.
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {}

    Write-Section (Msg "section_install")
    $target = Get-ReleaseTarget (Get-NativeArchitecture)
    Write-Info "$(Msg 'platform'): $target"
    $work = Join-Path ([IO.Path]::GetTempPath()) ("markitai-setup-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        $release = Resolve-Release $base $target "$env:MARKITAI_VERSION" $work
        Write-Info "$(Msg 'release'): $($release.Version)"
        Write-Step "$(Msg 'downloading') $($release.Archive)"
        $url = "$base/download/v$($release.Version)/$($release.Archive)"
        $zip = Join-Path $work $release.Archive
        try { Save-Download $url $zip } catch { Stop-Setup (Msg "error_download") "$url ($($_.Exception.Message))" }
        Test-Checksum $zip $release.Sha256
        Write-Ok (Msg "verified")
        $unpacked = Join-Path $work "unpacked"
        Expand-Release $zip $unpacked
        try {
            $reported = Invoke-MarkitaiVersion (Join-Path $unpacked "markitai.exe")
        } catch {
            Stop-Setup (Msg "error_run") "$($_.Exception.Message)"
        }
        $directory = Get-DefaultInstallDirectory
        $receipt = Install-Release $unpacked $directory $release $target
        $markitai = Join-Path $directory "markitai.exe"
        Write-Ok "$(Msg 'installed'): $markitai ($reported)"
        if (Add-ToUserPath $directory) {
            $receipt.PathAdded = $true
            Write-Receipt $directory $receipt
            Write-Ok (Msg "path_added")
        }
        # First install only: `init --yes` writes a configuration with the
        # providers this machine already has.
        if (-not (Test-Path -LiteralPath (Join-Path (Get-StateDirectory) "config.json"))) {
            $oldPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try { & $markitai init --yes *> $null } catch {} finally { $ErrorActionPreference = $oldPreference }
        }
        $optionalOk = Invoke-OptionalComponents $markitai
        Write-GettingStarted $directory
        if ($optionalOk) { Write-Outro "DarkGreen" (Msg "ready") } else { Write-Outro "DarkYellow" (Msg "done_warnings") }
    } catch {
        # Any other error still ends with the "Setup failed" line.
        if ("$($_.Exception.Message)" -notlike "markitai setup: *") { Stop-Setup "$($_.Exception.Message)" }
        throw
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $OnWindows) {
    Write-Intro (Msg "welcome")
    Stop-Setup (Msg "error_os_windows") "curl -fsSL https://markitai.dev/setup.sh | sh"
}
if ($Uninstall -or (Test-Truthy "$env:MARKITAI_UNINSTALL")) { Invoke-Uninstall } else { Invoke-Install }
} -Uninstall:$Uninstall
