# Markitai Setup Script - Unified installer with i18n support
# PowerShell 5.1+
# Auto-detects the language (en/zh). Run from the root of a markitai checkout,
# it asks whether to install markitai or set up a development environment.
#
# Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"   # Install markitai
#   .\scripts\setup.ps1                                     # In a checkout: install or dev setup
#
# Environment:
#   MARKITAI_SETUP_MODE=install|dev  choose without the question (default: install)
#   MARKITAI_INSTALL_OPTIONAL=1      install optional components without a terminal
#   MARKITAI_VERSION=X.Y.Z           pin the markitai release
#   MARKITAI_USE_MIRROR=1|0          use (or never offer) a PyPI mirror
#   MARKITAI_SETUP_LOG=PATH          full failure log (default: a private temp file)
#   NO_COLOR=1                       no colours; the marks stay
#
# Keep this file ASCII-only (a test enforces it): Windows PowerShell 5.1 reads
# a script without a BOM in the ANSI code page, and irm decodes the download
# as ISO-8859-1, so any other byte arrives as mojibake.

$ErrorActionPreference = "Stop"

# Native tools (uv, markitai) print UTF-8. Remember the console's encoding so
# it can be restored at the end: under irm | iex this is the user's own shell.
$script:OriginalOutputEncoding = $null
try {
    $script:OriginalOutputEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
} catch {}

# ============================================================
# Terminal capabilities
# ============================================================

# Windows PowerShell 5.1 has no $IsWindows; it only runs on Windows.
$script:OnWindows = ($PSVersionTable.PSEdition -ne "Core") -or $IsWindows

# Windows Terminal, VS Code, ConEmu and SSH clients draw Unicode and CJK with
# font fallback. The legacy console host does not: its font (Consolas, or
# NSimSun on Chinese systems) lacks the check and rounded-corner glyphs.
$script:ModernTerminal = [bool](
    $env:WT_SESSION -or $env:TERM_PROGRAM -or ($env:ConEmuANSI -eq "ON") -or
    $env:SSH_CONNECTION -or $env:SSH_CLIENT
)

function Test-OutputRedirected {
    try { return [Console]::IsOutputRedirected } catch { return $false }
}

# Unicode marks on any modern terminal, off Windows, and in redirected UTF-8
# output; ASCII in the legacy console host and for TERM=dumb.
function Test-UnicodeMarks {
    if ($env:TERM -eq "dumb") { return $false }
    if (-not $script:OnWindows -or $script:ModernTerminal) { return $true }
    if (Test-OutputRedirected) {
        try { return ([Console]::OutputEncoding.CodePage -eq 65001) } catch { return $false }
    }
    return $false
}

# Colour is off for NO_COLOR, TERM=dumb and redirected output (unless
# FORCE_COLOR is set); the marks carry the meaning either way.
function Test-UseColor {
    if ($env:NO_COLOR) { return $false }
    if ($env:TERM -eq "dumb") { return $false }
    if ($env:FORCE_COLOR -and $env:FORCE_COLOR -notmatch '^(0|false)$') { return $true }
    return -not (Test-OutputRedirected)
}

# The legacy console host can only draw Chinese with a CJK or UTF-8 code page
# (its font follows the code page); everywhere else it can.
function Test-CjkCapableOutput {
    if (-not $script:OnWindows -or $script:ModernTerminal) { return $true }
    if (Test-OutputRedirected) { return $true }
    $codePage = 0
    try { $codePage = $script:OriginalOutputEncoding.CodePage } catch {}
    return (@(932, 936, 949, 950, 54936, 65001) -contains $codePage)
}

# The classic "Windows PowerShell" shortcut repaints two console slots: the
# DarkMagenta background is navy and DarkYellow, its default text colour, is
# near-white. A DarkYellow warning would look like plain text there, so that
# palette alone uses the bright slots, which read well on its dark navy.
function Test-ClassicPowerShellPalette {
    if (-not $script:OnWindows -or $script:ModernTerminal) { return $false }
    try {
        $ui = $Host.UI.RawUI
        return ($ui.BackgroundColor -eq "DarkMagenta" -and $ui.ForegroundColor -eq "DarkYellow")
    } catch {
        return $false
    }
}

# Wrap width of the console, or 0 for redirected output (one line per
# message). COLUMNS overrides the measured width.
function Get-SetupWrapWidth {
    if ($env:TERM -eq "dumb" -or (Test-OutputRedirected)) { return 0 }
    if ($env:COLUMNS -match '^\d+$') { return [int]$env:COLUMNS }
    try { return [int][Console]::WindowWidth } catch {}
    try { return [int]$Host.UI.RawUI.WindowSize.Width } catch {}
    return 0
}

$script:UseColor = Test-UseColor
$script:WrapWidth = Get-SetupWrapWidth

# One colour per role (console slots matching ANSI 90/36/32/33/31, which
# terminal themes tune for light and dark backgrounds alike).
$script:C_GUIDE = "DarkGray"
$script:C_DIM = "DarkGray"
$script:C_ACCENT = "DarkCyan"
$script:C_OK = "DarkGreen"
$script:C_WARN = "DarkYellow"
$script:C_ERR = "DarkRed"
if (Test-ClassicPowerShellPalette) {
    $script:C_ACCENT = "Cyan"
    $script:C_OK = "Green"
    $script:C_WARN = "Yellow"
    $script:C_ERR = "Red"
}

# ============================================================
# Internationalization (i18n) System
# ============================================================

# Detect language from the culture. Chinese also needs output that can draw
# it (see Test-CjkCapableOutput); otherwise English is used.
# Returns: "zh" for Chinese, "en" for English (default)
function Get-Lang {
    $culture = ""
    try { $culture = (Get-Culture).Name } catch {}
    if ($culture -match "^zh" -and (Test-CjkCapableOutput)) {
        return "zh"
    }
    return "en"
}

# Decode the \uXXXX escapes of the Chinese table. A plain loop (no script
# block delegate) keeps it identical on PowerShell 5.1 and 7.
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

$script:LANG_CODE = Get-Lang

# Internationalization function
# Usage: i18n "key"
# Returns localized string for the given key. The script must stay ASCII:
# Windows PowerShell 5.1 reads a file without a BOM in the ANSI code page, so
# literal Chinese text turned into mojibake and parse errors under -File, and
# into garbled prompts under irm | iex. The Chinese table is written as \uXXXX
# escapes and decoded at run time; it matches setup.sh's strings (a test
# compares the two).
function i18n {
    param([string]$Key)

    if ($script:LANG_CODE -eq "zh") {
        $text = switch ($Key) {
            # Intro/Outro
            "welcome"                   { 'Markitai \u5b89\u88c5' }
            "setup_complete"            { 'markitai \u5df2\u5c31\u7eea' }
            "setup_complete_with_warnings" { 'markitai \u5df2\u5c31\u7eea\uff0c\u4f46\u6709\u90e8\u5206\u7ec4\u4ef6\u5931\u8d25' }
            "dev_setup_complete"        { '\u5f00\u53d1\u73af\u5883\u5df2\u5c31\u7eea' }
            "dev_setup_complete_with_warnings" { '\u5f00\u53d1\u73af\u5883\u5df2\u5c31\u7eea\uff0c\u4f46\u6709\u90e8\u5206\u7ec4\u4ef6\u5931\u8d25' }
            "setup_cancelled"           { '\u5df2\u53d6\u6d88\u5b89\u88c5' }

            # Mode
            "mode_user"                 { '\u6a21\u5f0f\uff1a\u5b89\u88c5 markitai' }
            "mode_dev"                  { '\u6a21\u5f0f\uff1a\u5f00\u53d1\u73af\u5883' }
            "repo_root_detected"        { '\u5f53\u524d\u76ee\u5f55\u662f markitai \u6e90\u7801\u4ed3\u5e93' }
            "confirm_dev_mode"          { '\u5728\u8fd9\u91cc\u8bbe\u7f6e\u5f00\u53d1\u73af\u5883\uff0c\u800c\u4e0d\u662f\u5b89\u88c5 markitai?' }
            "info_mode_default"         { '\u5c06\u5b89\u88c5 markitai\uff1b\u8bbe\u7f6e MARKITAI_SETUP_MODE=dev \u53ef\u6539\u4e3a\u8bbe\u7f6e\u5f00\u53d1\u73af\u5883' }
            "warn_unknown_mode"         { '\u5ffd\u7565 MARKITAI_SETUP_MODE (\u5e94\u4e3a install \u6216 dev)' }
            "error_dev_needs_repo"      { '\u5f00\u53d1\u73af\u5883\u9700\u8981 markitai \u6e90\u7801\u4ed3\u5e93\uff1b\u8bf7\u5728\u4ed3\u5e93\u6839\u76ee\u5f55\u8fd0\u884c\u5b89\u88c5\u811a\u672c' }

            # Sections
            "section_prerequisites"     { '\u524d\u7f6e\u6761\u4ef6' }
            "section_core"              { '\u6838\u5fc3\u7ec4\u4ef6' }
            "section_optional"          { '\u53ef\u9009\u7ec4\u4ef6' }
            "section_dev_env"           { '\u5f00\u53d1\u73af\u5883' }
            "section_llm_cli"           { 'LLM CLI \u5de5\u5177' }
            "section_summary"           { '\u5b89\u88c5\u6458\u8981' }

            # Status
            "installed"                 { '\u5df2\u5b89\u88c5' }
            "installing"                { '\u6b63\u5728\u5b89\u88c5' }
            "skipped"                   { '\u5df2\u8df3\u8fc7' }
            "failed"                    { '\u5931\u8d25' }
            "already_installed"         { '\u5df2\u7ecf\u5b89\u88c5' }
            "not_found"                 { '\u672a\u627e\u5230' }

            # Components
            "uv"                        { 'uv \u5305\u7ba1\u7406\u5668' }
            "python"                    { 'Python' }
            "markitai"                  { 'markitai' }
            "serve"                     { 'Web UI (markitai serve)' }
            "ocr"                       { 'OCR (\u626b\u63cf\u4ef6\u6587\u5b57\u8bc6\u522b)' }
            "playwright"                { 'Playwright \u6d4f\u89c8\u5668' }
            "claude_cli"                { 'Claude Code CLI' }
            "copilot_cli"               { 'Copilot CLI' }
            "precommit"                 { 'pre-commit hooks' }
            "python_deps"               { 'Python \u4f9d\u8d56' }

            # Confirmations
            "confirm_serve"             { '\u5b89\u88c5 Web UI \u4f9d\u8d56? (\u542f\u7528 markitai serve)' }
            "confirm_ocr"               { '\u5b89\u88c5 OCR \u652f\u6301? (\u8bc6\u522b\u626b\u63cf\u4ef6\u548c\u56fe\u7247\u4e2d\u7684\u6587\u5b57, \u7ea6 150MB)' }
            "confirm_playwright"        { '\u5b89\u88c5 Playwright \u6d4f\u89c8\u5668? (\u7528\u4e8e JS \u6e32\u67d3\u9875\u9762)' }
            "confirm_claude_cli"        { '\u5b89\u88c5 Claude Code CLI? (\u4f7f\u7528 Claude \u8ba2\u9605)' }
            "confirm_copilot_cli"       { '\u5b89\u88c5 Copilot CLI? (\u4f7f\u7528 GitHub Copilot \u8ba2\u9605)' }
            "confirm_uv"                { '\u5b89\u88c5 uv \u5305\u7ba1\u7406\u5668?' }

            # Info messages
            "info_chatgpt_route"        { 'ChatGPT \u8ba2\u9605\u65e0\u9700 CLI\uff0c\u4e5f\u65e0\u9700 extra\uff08\u9996\u6b21\u4f7f\u7528\u65f6\u767b\u5f55\uff09' }
            "info_playwright_purpose"   { 'Playwright \u7528\u4e8e\u83b7\u53d6 JavaScript \u6e32\u67d3\u7684\u7f51\u9875\u5185\u5bb9' }
            "info_project_dir"          { '\u9879\u76ee\u76ee\u5f55' }
            "info_docs"                 { '\u6587\u6863' }
            "info_issues"               { '\u95ee\u9898\u53cd\u9988' }
            "info_syncing_deps"         { '\u6b63\u5728\u540c\u6b65\u4f9d\u8d56...' }
            "info_deps_synced"          { '\u4f9d\u8d56\u540c\u6b65\u5b8c\u6210' }
            "info_precommit_installed"  { 'pre-commit hooks \u5df2\u5b89\u88c5' }
            "info_error_log"            { '\u5b8c\u6574\u9519\u8bef\u65e5\u5fd7' }

            # Error messages
            "error_uv_required"         { '\u9700\u8981\u5b89\u88c5 uv \u5305\u7ba1\u7406\u5668' }
            "error_python_required"     { '\u9700\u8981\u5b89\u88c5 Python 3.11-3.13' }
            "error_unexpected"          { '\u53d1\u751f\u610f\u5916\u9519\u8bef' }
            "error_setup_failed"        { '\u5b89\u88c5\u5931\u8d25' }

            # Install source (repo detection)
            "repo_detected"             { '\u68c0\u6d4b\u5230 markitai \u6e90\u7801\u4ed3\u5e93' }
            "confirm_local_install"     { '\u4ece\u8fd9\u4e2a\u6e90\u7801\u4ed3\u5e93\u5b89\u88c5 markitai\uff0c\u800c\u4e0d\u662f PyPI \u53d1\u5e03\u7248?' }
            "source_local"              { '\u5b89\u88c5\u6765\u6e90: \u672c\u5730\u6e90\u7801' }
            "source_pypi"               { '\u5b89\u88c5\u6765\u6e90: PyPI' }
            "info_repo_noninteractive"  { '\u975e\u4ea4\u4e92\u6a21\u5f0f: \u5df2\u68c0\u6d4b\u5230\u6e90\u7801\u4ed3\u5e93, \u5c06\u4f7f\u7528 PyPI \u53d1\u5e03\u7248' }

            # Network / Mirrors
            "mirror_slow_index"         { 'PyPI \u9ed8\u8ba4\u6e90\u54cd\u5e94\u7f13\u6162\u6216\u4e0d\u53ef\u8fbe' }
            "mirror_confirm"            { '\u4f7f\u7528\u955c\u50cf\u6e90\u52a0\u901f\u5b89\u88c5?' }
            "mirror_select"             { '\u9009\u62e9\u955c\u50cf\u6e90' }
            "mirror_tuna"               { '\u6e05\u534e TUNA (\u63a8\u8350)' }
            "mirror_aliyun"             { '\u963f\u91cc\u4e91' }
            "mirror_tencent"            { '\u817e\u8baf\u4e91' }
            "mirror_huawei"             { '\u534e\u4e3a\u4e91' }
            "mirror_enabled"            { '\u5df2\u542f\u7528\u955c\u50cf\u52a0\u901f' }
            "mirror_skipped"            { '\u7ee7\u7eed\u4f7f\u7528\u9ed8\u8ba4\u6e90' }
            "mirror_pypi"               { 'PyPI \u955c\u50cf' }
            "mirror_playwright"         { 'Playwright \u955c\u50cf' }
            "mirror_npm"                { 'npm \u955c\u50cf' }

            # Warnings
            "warn_admin"                { '\u6b63\u5728\u4ee5\u7ba1\u7406\u5458\u8eab\u4efd\u8fd0\u884c' }
            "warn_admin_risk"           { 'markitai \u6309\u7528\u6237\u5b89\u88c5\uff0c\u4ece\u4e0d\u9700\u8981\u7ba1\u7406\u5458\u6743\u9650\u3002\u4ee5\u7ba1\u7406\u5458\u8fd0\u884c\u65f6\uff0c\u4e0b\u8f7d\u7684\u5b89\u88c5\u7a0b\u5e8f\u90fd\u62e5\u6709\u8fd9\u53f0\u7535\u8111\u7684\u5b8c\u5168\u63a7\u5236\u6743\uff1b\u5982\u679c\u7ba1\u7406\u5458\u662f\u53e6\u4e00\u4e2a\u8d26\u6237\uff0cmarkitai \u4f1a\u88c5\u5230\u90a3\u4e2a\u8d26\u6237\u4e0b\uff0c\u800c\u4e0d\u662f\u4f60\u7684\u8d26\u6237\u3002' }
            "confirm_continue_as_admin" { '\u4ecd\u4ee5\u7ba1\u7406\u5458\u8eab\u4efd\u7ee7\u7eed?' }
            "info_admin_noninteractive" { '\u6ca1\u6709\u53ef\u4f9b\u786e\u8ba4\u7684\u7ec8\u7aef\uff0c\u7ee7\u7eed\u4ee5\u7ba1\u7406\u5458\u8eab\u4efd\u8fd0\u884c' }
            "warn_wsl"                  { '\u68c0\u6d4b\u5230 WSL \u73af\u5883' }
            "warn_wsl_tip"              { '\u5728 WSL \u91cc\u8bf7\u6539\u7528 shell \u5b89\u88c5\u811a\u672c:' }
            "confirm_continue"          { '\u4ecd\u7136\u7ee7\u7eed?' }
            "warn_exec_policy"          { '\u4f60\u7684 PowerShell \u6267\u884c\u7b56\u7565 ({0}) \u53ef\u80fd\u4f1a\u963b\u6b62\u811a\u672c\u8fd0\u884c\uff1b\u5982\u9700\u4e3a\u4f60\u7684\u8d26\u6237\u5141\u8bb8\u811a\u672c\uff0c\u8bf7\u8fd0\u884c:' }
            "warn_uv_restart"           { 'uv \u5df2\u5b89\u88c5\uff0c\u4f46\u5f53\u524d PowerShell \u8fd8\u627e\u4e0d\u5230\u5b83\uff1b\u8bf7\u6253\u5f00\u65b0\u7684 PowerShell \u7a97\u53e3\u540e\u91cd\u65b0\u8fd0\u884c\u5b89\u88c5\u811a\u672c' }
            "warn_bin_off_path"         { '\u5f53\u524d shell \u7684 PATH \u4e0d\u5305\u542b markitai \u7684\u547d\u4ee4\u76ee\u5f55' }
            "info_bin_off_path_fix"     { '\u8bf7\u6253\u5f00\u65b0\u7684\u7ec8\u7aef\uff1b\u5982\u679c\u4ecd\u627e\u4e0d\u5230 markitai\uff0c\u8bf7\u8fd0\u884c:' }

            # Getting started
            "getting_started"           { '\u5f00\u59cb\u4f7f\u7528' }
            "quick_start"               { '\u5feb\u901f\u5f00\u59cb' }
            "run_tests"                 { '\u8fd0\u884c\u6d4b\u8bd5' }
            "run_cli"                   { '\u8fd0\u884c CLI' }
            "interactive_mode"          { '\u4ea4\u4e92\u6a21\u5f0f' }
            "start_web_ui"              { '\u542f\u52a8 Web UI' }
            "configure_llm"             { '\u914d\u7f6e LLM' }
            "configure_env"             { '\u914d\u7f6e\u73af\u5883\u53d8\u91cf' }
            "convert_file"              { '\u8f6c\u6362\u6587\u4ef6' }
            "show_help"                 { '\u663e\u793a\u5e2e\u52a9' }
            "short_alias"               { '\u63d0\u793a\uff1amkai \u662f markitai \u7684\u7b80\u5199\uff0c\u4e24\u4e2a\u547d\u4ee4\u7b49\u4ef7' }
            "mkai_conflict"             { '\u7cfb\u7edf\u5df2\u6709 mkai \u547d\u4ee4\uff0c\u4f1a\u906e\u853d markitai \u7684\u522b\u540d\uff1b\u8bf7\u4f7f\u7528\u5b8c\u6574\u547d\u4ee4 markitai' }

            # Default fallback
            default                     { $Key }
        }
        return (ConvertFrom-CodePointEscape $text)
    }

    # English (default)
    switch ($Key) {
        # Intro/Outro
        "welcome"                   { return 'Markitai Setup' }
        "setup_complete"            { return 'Markitai is ready' }
        "setup_complete_with_warnings" { return 'Markitai is ready, but some components failed' }
        "dev_setup_complete"        { return 'Development environment ready' }
        "dev_setup_complete_with_warnings" { return 'Development environment ready, but some components failed' }
        "setup_cancelled"           { return 'Setup cancelled' }

        # Mode
        "mode_user"                 { return 'Mode: install markitai' }
        "mode_dev"                  { return 'Mode: development environment' }
        "repo_root_detected"        { return 'This folder is a markitai source checkout' }
        "confirm_dev_mode"          { return 'Set up a development environment here instead of installing markitai?' }
        "info_mode_default"         { return 'Installing markitai; set MARKITAI_SETUP_MODE=dev for a development environment instead' }
        "warn_unknown_mode"         { return 'Ignoring MARKITAI_SETUP_MODE (expected install or dev)' }
        "error_dev_needs_repo"      { return 'A development environment needs a markitai checkout; run setup from its root folder' }

        # Sections
        "section_prerequisites"     { return 'Prerequisites' }
        "section_core"              { return 'Core components' }
        "section_optional"          { return 'Optional components' }
        "section_dev_env"           { return 'Development environment' }
        "section_llm_cli"           { return 'LLM CLI tools' }
        "section_summary"           { return 'Summary' }

        # Status
        "installed"                 { return 'installed' }
        "installing"                { return 'Installing' }
        "skipped"                   { return 'skipped' }
        "failed"                    { return 'failed' }
        "already_installed"         { return 'already installed' }
        "not_found"                 { return 'not found' }

        # Components
        "uv"                        { return 'uv package manager' }
        "python"                    { return 'Python' }
        "markitai"                  { return 'markitai' }
        "serve"                     { return 'Web UI (markitai serve)' }
        "ocr"                       { return 'OCR (scanned-document text recognition)' }
        "playwright"                { return 'Playwright browser' }
        "claude_cli"                { return 'Claude Code CLI' }
        "copilot_cli"               { return 'Copilot CLI' }
        "precommit"                 { return 'pre-commit hooks' }
        "python_deps"               { return 'Python dependencies' }

        # Confirmations
        "confirm_serve"             { return 'Install Web UI dependencies? (enables markitai serve)' }
        "confirm_ocr"               { return 'Install OCR support? (text recognition in scanned PDFs and images, ~150MB)' }
        "confirm_playwright"        { return 'Install Playwright browser? (for JS-rendered pages)' }
        "confirm_claude_cli"        { return 'Install Claude Code CLI? (use your Claude subscription)' }
        "confirm_copilot_cli"       { return 'Install Copilot CLI? (use your GitHub Copilot subscription)' }
        "confirm_uv"                { return 'Install uv package manager?' }

        # Info messages
        "info_chatgpt_route"        { return 'ChatGPT subscription needs no CLI and no extra (it signs in on first use)' }
        "info_playwright_purpose"   { return 'Playwright fetches JavaScript-rendered web pages' }
        "info_project_dir"          { return 'Project directory' }
        "info_docs"                 { return 'Documentation' }
        "info_issues"               { return 'Issues' }
        "info_syncing_deps"         { return 'Syncing dependencies...' }
        "info_deps_synced"          { return 'Dependencies synced' }
        "info_precommit_installed"  { return 'pre-commit hooks installed' }
        "info_error_log"            { return 'Full error log' }

        # Error messages
        "error_uv_required"         { return 'uv package manager is required' }
        "error_python_required"     { return 'Python 3.11-3.13 is required' }
        "error_unexpected"          { return 'Unexpected error' }
        "error_setup_failed"        { return 'Setup failed' }

        # Install source (repo detection)
        "repo_detected"             { return 'markitai source repo detected' }
        "confirm_local_install"     { return 'Install markitai from this checkout instead of the PyPI release?' }
        "source_local"              { return 'Install source: local repo' }
        "source_pypi"               { return 'Install source: PyPI' }
        "info_repo_noninteractive"  { return 'Non-interactive mode: source repo detected, using PyPI release' }

        # Network / Mirrors
        "mirror_slow_index"         { return 'The default PyPI index is slow or unreachable from here' }
        "mirror_confirm"            { return 'Use a mirror to speed up the install?' }
        "mirror_select"             { return 'Select mirror source' }
        "mirror_tuna"               { return 'Tsinghua TUNA (Recommended)' }
        "mirror_aliyun"             { return 'Alibaba Cloud' }
        "mirror_tencent"            { return 'Tencent Cloud' }
        "mirror_huawei"             { return 'Huawei Cloud' }
        "mirror_enabled"            { return 'Mirror acceleration enabled' }
        "mirror_skipped"            { return 'Keeping the default index' }
        "mirror_pypi"               { return 'PyPI mirror' }
        "mirror_playwright"         { return 'Playwright mirror' }
        "mirror_npm"                { return 'npm mirror' }

        # Warnings
        "warn_admin"                { return 'Running as administrator' }
        "warn_admin_risk"           { return 'markitai installs per user and never needs administrator rights. Elevated, every downloaded installer gets full control of this computer, and if the administrator is a different account, markitai is installed for that account instead of yours.' }
        "confirm_continue_as_admin" { return 'Continue as administrator anyway?' }
        "info_admin_noninteractive" { return 'No terminal to ask; continuing as administrator' }
        "warn_wsl"                  { return 'WSL detected' }
        "warn_wsl_tip"              { return 'Inside WSL, use the shell installer instead:' }
        "confirm_continue"          { return 'Continue anyway?' }
        "warn_exec_policy"          { return 'Your PowerShell execution policy ({0}) may block scripts; to allow them for your account, run:' }
        "warn_uv_restart"           { return 'uv package manager installed, but this PowerShell cannot find it yet; open a new PowerShell window and run setup again' }
        "warn_bin_off_path"         { return 'This shell''s PATH does not include markitai''s command folder' }
        "info_bin_off_path_fix"     { return 'Open a new terminal; if markitai is still not found, run:' }

        # Getting started
        "getting_started"           { return 'Getting started' }
        "quick_start"               { return 'Quick start' }
        "run_tests"                 { return 'Run tests' }
        "run_cli"                   { return 'Run CLI' }
        "interactive_mode"          { return 'Interactive mode' }
        "start_web_ui"              { return 'Start Web UI' }
        "configure_llm"             { return 'Configure LLM' }
        "configure_env"             { return 'Configure environment' }
        "convert_file"              { return 'Convert a file' }
        "show_help"                 { return 'Show help' }
        "short_alias"               { return 'Tip: mkai is a short alias for markitai (both work)' }
        "mkai_conflict"             { return 'An existing ''mkai'' command shadows markitai''s alias; use the full ''markitai'' command' }

        # Default fallback
        default                     { return $Key }
    }
}

# ============================================================
# Compact tree-style visual components
# ============================================================
# The one-column guide follows the `withGuide` visual grammar popularized by
# @clack/prompts. Keeping this renderer inline preserves PowerShell 5.1 support
# and lets the bootstrap run before Node or Python is installed.
#
# One terminal style for everything markitai prints, the CLI included: the
# guide is gray, structure keeps the default colour, the accent marks prompts,
# actions and commands, green is only success, yellow only warnings, red only
# errors. A guide-only row precedes each section header and none sits inside a
# section; items are "<bar>  <mark> <text>", details and commands sit two
# columns further in. Marks: check success, ! warning, cross error, arrow
# action, bullet info, circle skipped, ? prompt (ASCII + ! x > - o ? where
# the console cannot draw them).

if (Test-UnicodeMarks) {
    $S_BAR = [string][char]0x2502
    $S_TOP = [string][char]0x256D + [char]0x2500
    $S_BRANCH = [string][char]0x251C + [char]0x2500
    $S_BOTTOM = [string][char]0x2570 + [char]0x2500
    $S_CHECK = [string][char]0x2713
    $S_CROSS = [string][char]0x2717
    $S_ARROW = [string][char]0x2192
    $S_DOT = [string][char]0x2022
    $S_CIRCLE = [string][char]0x25CB
} else {
    $S_BAR = "|"
    $S_TOP = "+-"
    $S_BRANCH = "+-"
    $S_BOTTOM = "+-"
    $S_CHECK = "+"
    $S_CROSS = "x"
    $S_ARROW = ">"
    $S_DOT = "-"
    $S_CIRCLE = "o"
}
$S_WARN = "!"
$S_ASK = "?"

# Session state lets the global catch close an interrupted tree exactly once.
$script:CLACK_SESSION_ACTIVE = $false
$script:CLACK_SESSION_CLOSED = $false
$script:SETUP_LOG_FILE = $env:MARKITAI_SETUP_LOG
$script:SETUP_LOG_AUTO = $false
$script:SETUP_LOG_SHOWN = $false

# Write text in a role colour (or plain when colour is off).
function Write-Styled {
    param(
        [string]$Text,
        [string]$Color = ""
    )
    if ([string]::IsNullOrEmpty($Text)) { return }
    if ($script:UseColor -and $Color) {
        Write-Host $Text -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Text -NoNewline
    }
}

# Display columns of a string: CJK and full-width characters take two.
function Get-DisplayWidth {
    param([string]$Text)

    $width = 0
    foreach ($ch in $Text.ToCharArray()) {
        $code = [int]$ch
        if (($code -ge 0x1100 -and $code -le 0x115F) -or
            ($code -ge 0x2E80 -and $code -le 0xA4CF) -or
            ($code -ge 0xAC00 -and $code -le 0xD7A3) -or
            ($code -ge 0xF900 -and $code -le 0xFAFF) -or
            ($code -ge 0xFE30 -and $code -le 0xFE4F) -or
            ($code -ge 0xFF00 -and $code -le 0xFF60) -or
            ($code -ge 0xFFE0 -and $code -le 0xFFE6)) {
            $width += 2
        } else {
            $width += 1
        }
    }
    return $width
}

# Word-wrap text to a number of display columns (0: no wrapping). A run of
# CJK characters longer than a line may break between characters; other long
# words (paths, URLs) are never split.
function Split-WrappedText {
    param(
        [string]$Text,
        [int]$Width
    )

    if ($Width -le 0) { return ,@($Text) }
    $lines = New-Object System.Collections.Generic.List[string]
    $line = ""
    $lineWidth = 0
    foreach ($word in ($Text -split ' ')) {
        if ($word -eq "") { continue }
        $wordWidth = Get-DisplayWidth $word
        if ($line -ne "" -and $lineWidth + 1 + $wordWidth -le $Width) {
            $line += " $word"
            $lineWidth += 1 + $wordWidth
            continue
        }
        if ($line -ne "") {
            $lines.Add($line)
            $line = ""
            $lineWidth = 0
        }
        if ($wordWidth -le $Width -or $wordWidth -eq $word.Length) {
            $line = $word
            $lineWidth = $wordWidth
            continue
        }
        foreach ($ch in $word.ToCharArray()) {
            $charWidth = Get-DisplayWidth ([string]$ch)
            if ($line -ne "" -and $lineWidth + $charWidth -gt $Width) {
                $lines.Add($line)
                $line = ""
                $lineWidth = 0
            }
            $line += $ch
            $lineWidth += $charWidth
        }
    }
    $lines.Add($line)
    return ,$lines.ToArray()
}

# One row on the guide: "<bar>  <indent><mark> <text>[ <suffix>]". On a
# console, long text wraps at word boundaries with a hanging indent under its
# first character; redirected output keeps one line per message.
function Write-GuideRow {
    param(
        [string]$Mark = "",
        [string]$MarkColor = "",
        [string]$Text = "",
        [string]$TextColor = "",
        [string]$Indent = "",
        [string]$Suffix = "",
        [string]$SuffixColor = "",
        [int]$Reserve = 0,
        [switch]$NoNewline
    )

    $pad = if ($Mark) { "$Indent  " } else { $Indent }
    $width = 0
    if ($script:WrapWidth -gt 0) {
        # Bar and two spaces, plus one column so the legacy console host does
        # not wrap a line that fills the window exactly.
        $width = $script:WrapWidth - 4 - $pad.Length - $Reserve
        if ($width -lt 16) { $width = 0 }
    }
    $lines = Split-WrappedText -Text $Text -Width $width
    for ($i = 0; $i -lt $lines.Count; $i++) {
        Write-Styled $S_BAR $script:C_GUIDE
        Write-Host "  " -NoNewline
        if ($i -eq 0) {
            Write-Styled $Indent
            if ($Mark) {
                Write-Styled $Mark $MarkColor
                Write-Host " " -NoNewline
            }
        } else {
            Write-Styled $pad
        }
        Write-Styled $lines[$i] $TextColor
        if ($i -lt $lines.Count - 1) { Write-Host "" }
    }
    if ($Suffix) {
        Write-Host " " -NoNewline
        Write-Styled $Suffix $SuffixColor
    }
    if (-not $NoNewline) { Write-Host "" }
}

# Print an empty row while preserving the session guide.
function Write-GuideLine {
    Write-Styled $S_BAR $script:C_GUIDE
    Write-Host ""
}

# Session intro - start of CLI flow. Items may follow directly; each section
# adds its own guide row above its header.
function Clack-Intro {
    param([string]$Title)
    $script:CLACK_SESSION_ACTIVE = $true
    $script:CLACK_SESSION_CLOSED = $false
    Write-Host ""
    Write-Styled $S_TOP $script:C_GUIDE
    Write-Host " $Title"
}

# Session outro - end of CLI flow
function Clack-Outro {
    param([string]$Message)
    $script:CLACK_SESSION_CLOSED = $true
    if ($script:SETUP_LOG_AUTO -and $script:SETUP_LOG_FILE) {
        Remove-Item $script:SETUP_LOG_FILE -Force -ErrorAction SilentlyContinue
        $script:SETUP_LOG_FILE = $null
    }
    Write-GuideLine
    Write-Styled $S_BOTTOM $script:C_GUIDE
    Write-Styled " $Message" $script:C_OK
    Write-Host ""
    Write-Host ""
}

# Non-fatal component failures close the tree in warning yellow, not success
# green, while still returning a successful core installation.
function Clack-OutroWarning {
    param([string]$Message)
    Show-SetupErrorLog
    $script:CLACK_SESSION_CLOSED = $true
    Write-GuideLine
    Write-Styled $S_BOTTOM $script:C_GUIDE
    Write-Styled " $Message" $script:C_WARN
    Write-Host ""
    Write-Host ""
}

# Section header connected to the session guide. Structure is never coloured:
# a coloured header reads as a status.
function Clack-Section {
    param([string]$Title)
    Write-GuideLine
    Write-Styled $S_BRANCH $script:C_GUIDE
    Write-Host " $Title"
}

# Log with guide line - success
function Clack-Success {
    param([string]$Message)
    Write-GuideRow -Mark $S_CHECK -MarkColor $script:C_OK -Text $Message
}

# Log with guide line - error
function Clack-Error {
    param([string]$Message)
    Write-GuideRow -Mark $S_CROSS -MarkColor $script:C_ERR -Text $Message
}

# Log with guide line - warning
function Clack-Warn {
    param([string]$Message)
    Write-GuideRow -Mark $S_WARN -MarkColor $script:C_WARN -Text $Message
}

# Log with guide line - a neutral fact (paths, choices, what a step is for)
function Clack-Info {
    param([string]$Message)
    Write-GuideRow -Mark $S_DOT -MarkColor $script:C_DIM -Text $Message
}

# Log with guide line - an action in progress
function Clack-Step {
    param([string]$Message)
    Write-GuideRow -Mark $S_ARROW -MarkColor $script:C_ACCENT -Text $Message
}

# Log with guide line - skipped on purpose
function Clack-Skip {
    param([string]$Message)
    Write-GuideRow -Mark $S_CIRCLE -MarkColor $script:C_DIM -Text $Message -TextColor $script:C_DIM
}

# Gray detail under the item above it, two columns further in.
function Clack-DetailRow {
    param([string]$Message)
    Write-GuideRow -Indent "  " -Text $Message -TextColor $script:C_DIM
}

# A command to copy: accent colour, its own row, never wrapped, so a console
# narrower than the command still copies it back as one line.
function Clack-Command {
    param([string]$Command)
    Write-Styled $S_BAR $script:C_GUIDE
    Write-Host "    " -NoNewline
    Write-Styled $Command $script:C_ACCENT
    Write-Host ""
}

# A label and the command it describes.
function Clack-Hint {
    param(
        [string]$Label,
        [string]$Command
    )
    Clack-Info "${Label}:"
    Clack-Command $Command
}

function Initialize-SetupLog {
    if ($script:SETUP_LOG_FILE) {
        try {
            $null = [IO.File]::Open(
                $script:SETUP_LOG_FILE,
                [IO.FileMode]::Append,
                [IO.FileAccess]::Write,
                [IO.FileShare]::Read
            ).Dispose()
            return $true
        } catch {
            $script:SETUP_LOG_FILE = $null
            return $false
        }
    }

    try {
        $script:SETUP_LOG_FILE = Join-Path (
            [IO.Path]::GetTempPath()
        ) "markitai-setup-$([Guid]::NewGuid()).log"
        $null = [IO.File]::Create($script:SETUP_LOG_FILE).Dispose()
        $script:SETUP_LOG_AUTO = $true
        return $true
    } catch {
        $script:SETUP_LOG_FILE = $null
        return $false
    }
}

function Write-SetupDiagnostic {
    param(
        [string]$Text,
        [string]$Context = "diagnostic"
    )

    if ([string]::IsNullOrWhiteSpace($Text)) { return }
    if (-not (Initialize-SetupLog)) { return }
    try {
        $entry = "`r`n== $Context ==`r`n$Text`r`n"
        [IO.File]::AppendAllText($script:SETUP_LOG_FILE, $entry)
    } catch {}
}

function Show-SetupErrorLog {
    if (-not $script:SETUP_LOG_FILE -or $script:SETUP_LOG_SHOWN) { return }
    $script:SETUP_LOG_SHOWN = $true
    Clack-Info "$(i18n 'info_error_log'): $($script:SETUP_LOG_FILE)"
}

# Print only the final few third-party diagnostic lines. ANSI sequences are
# removed so captured native errors cannot turn the whole block red; the full
# sanitized text is retained in a private setup log for troubleshooting.
function Clack-Detail {
    param(
        [AllowNull()]
        [object]$Detail,
        [int]$MaxLines = 6,
        [string]$Context = "diagnostic"
    )

    if ($null -eq $Detail) { return }
    $text = if ($Detail -is [string]) {
        $Detail
    } else {
        (@($Detail) | ForEach-Object { "$_" }) -join [Environment]::NewLine
    }
    if ([string]::IsNullOrWhiteSpace($text)) { return }

    $escape = [Regex]::Escape([string][char]27)
    $clean = [Regex]::Replace($text, "$escape\[[0-9;?]*[A-Za-z]", "")
    Write-SetupDiagnostic -Text $clean -Context $Context
    $lines = @(
        $clean -split "`r?`n" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Select-Object -Last $MaxLines
    )
    foreach ($line in $lines) {
        $bounded = if ($line.Length -gt 300) {
            $line.Substring(0, 297) + "..."
        } else {
            $line
        }
        Clack-DetailRow $bounded
    }
}

# Read one answer, or take the default without interactive input (the default
# is then printed so a log shows what was chosen).
function Read-ClackAnswer {
    param([string]$Default)

    Write-Host " " -NoNewline
    if (Test-InteractiveInput) {
        $answer = Read-Host
    } else {
        Write-Host $Default
        $answer = $Default
    }
    if ([string]::IsNullOrWhiteSpace($answer)) {
        $answer = $Default
    }
    return $answer
}

# Yes/no question on the guide: "<bar>  ? Question [y/N] ".
function Clack-Confirm {
    param(
        [string]$Prompt,
        [string]$Default = "n"
    )

    if ($Default -eq "y") {
        $hint = "[Y/n]"
    } else {
        $hint = "[y/N]"
    }

    Write-GuideRow -Mark $S_ASK -MarkColor $script:C_ACCENT -Text $Prompt `
        -Suffix $hint -SuffixColor $script:C_DIM -Reserve 7 -NoNewline
    $answer = Read-ClackAnswer -Default $Default
    return $answer -match "^[Yy]"
}

# PowerShell's redirected-input state is the equivalent of the POSIX setup
# script having no usable terminal for prompts.
function Test-InteractiveInput {
    try {
        return -not [Console]::IsInputRedirected
    } catch {
        return $false
    }
}

function Test-OptionalInstallRequested {
    return $env:MARKITAI_INSTALL_OPTIONAL -match "^(1|true|yes|on)$"
}

# Without interactive input, optional components are disabled unless the
# caller explicitly opts in through MARKITAI_INSTALL_OPTIONAL.
function Confirm-OptionalInstall {
    param(
        [string]$Prompt,
        [string]$Default = "n"
    )

    if (-not (Test-InteractiveInput)) {
        return (Test-OptionalInstallRequested)
    }
    return (Clack-Confirm $Prompt $Default)
}

# Cancel message
function Clack-Cancel {
    param([string]$Message)
    Show-SetupErrorLog
    $script:CLACK_SESSION_CLOSED = $true
    Write-GuideLine
    Write-Styled $S_BOTTOM $script:C_GUIDE
    Write-Styled " $Message" $script:C_ERR
    Write-Host ""
    Write-Host ""
}

# Last-resort guard for exceptions outside the expected per-component paths.
function Complete-UnexpectedFailure {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    if ($script:CLACK_SESSION_ACTIVE -and -not $script:CLACK_SESSION_CLOSED) {
        Clack-Error (i18n "error_unexpected")
        Clack-Detail -Detail $ErrorRecord.Exception.Message -MaxLines 3
        Clack-Cancel (i18n "error_setup_failed")
        return
    }

    Write-Styled (i18n "error_setup_failed") $script:C_ERR
    Write-Host ""
}

# ============================================================
# Installation Status Tracking
# ============================================================
$script:INSTALLED_COMPONENTS = @()
$script:SKIPPED_COMPONENTS = @()
$script:FAILED_COMPONENTS = @()

function Track-Install {
    param(
        [string]$Component,
        [ValidateSet("installed", "skipped", "failed")]
        [string]$Status
    )

    switch ($Status) {
        "installed" { $script:INSTALLED_COMPONENTS += $Component }
        "skipped" { $script:SKIPPED_COMPONENTS += $Component }
        "failed" { $script:FAILED_COMPONENTS += $Component }
    }
}

# Optional components are failure-isolated: an unexpected exception is shown
# as a bounded tree item, tracked, and does not prevent later choices/summary.
function Invoke-OptionalStep {
    param(
        [string]$Component,
        [scriptblock]$Action
    )

    try {
        & $Action | Out-Null
    } catch {
        Clack-Error "$(i18n $Component) $(i18n 'failed')"
        Clack-Detail -Detail $_.Exception.Message -MaxLines 3
        $alreadyTracked =
            ($script:INSTALLED_COMPONENTS -contains $Component) -or
            ($script:SKIPPED_COMPONENTS -contains $Component) -or
            ($script:FAILED_COMPONENTS -contains $Component)
        if (-not $alreadyTracked) {
            Track-Install -Component $Component -Status "failed"
        }
    }
}

# Best-effort maintenance steps have no standalone summary component, but must
# still not tear down an otherwise successful setup.
function Invoke-BestEffortStep {
    param(
        [string]$FailureMessage,
        [scriptblock]$Action
    )

    try {
        & $Action | Out-Null
    } catch {
        Clack-Warn $FailureMessage
        Clack-Detail -Detail $_.Exception.Message -MaxLines 3
    }
}

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
$script:MarkitaiVersion = $env:MARKITAI_VERSION
$script:UvVersion = $env:UV_VERSION
$script:PYTHON_CMD = $null
# PATH as setup found it, and the folder uv puts tool commands in.
$script:OriginalPath = $env:PATH
$script:UvToolBin = $null

# ============================================================
# Utility Functions
# ============================================================

# Check if a command exists
function Test-CommandExists {
    param([string]$CommandName)
    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    return ($null -ne $cmd)
}

# Run a native command without allowing PowerShell 5.1 to paint native stderr
# as unbounded red ErrorRecords. Callers decide whether to show the bounded
# Output through Clack-Detail.
function Invoke-NativeQuietly {
    param([scriptblock]$Command)

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @()
    $exitCode = 1
    try {
        $LASTEXITCODE = $null
        $output = @(& $Command 2>&1)
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
        $output = @($_.Exception.Message)
        $exitCode = 1
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }

    return [PSCustomObject]@{
        Success = ($exitCode -eq 0)
        ExitCode = $exitCode
        Output = $output
    }
}

# Download official PowerShell installers to a temporary file and execute them
# in a child process. An installer calling `exit` can then never terminate this
# parent script before its tree is closed.
function Invoke-PowerShellInstallerQuietly {
    param([string]$Uri)

    $tempPath = Join-Path ([IO.Path]::GetTempPath()) "markitai-installer-$([Guid]::NewGuid()).ps1"
    try {
        $content = Invoke-RestMethod -Uri $Uri -ErrorAction Stop
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($tempPath, [string]$content, $utf8NoBom)
        $engine = (Get-Process -Id $PID -ErrorAction Stop).Path
        return Invoke-NativeQuietly {
            & $engine -NoLogo -NoProfile -ExecutionPolicy Bypass -File $tempPath
        }
    } catch {
        return [PSCustomObject]@{
            Success = $false
            ExitCode = 1
            Output = @($_.Exception.Message)
        }
    } finally {
        Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
    }
}

# Return true when a folder is the root of a markitai checkout.
function Test-MarkitaiCheckout {
    param([string]$Path)

    if (-not $Path) { return $false }
    $pyproject = Join-Path $Path "pyproject.toml"
    if (-not (Test-Path $pyproject)) { return $false }
    if (-not (Test-Path (Join-Path $Path "packages/markitai"))) { return $false }
    $content = Get-Content $pyproject -Raw -ErrorAction SilentlyContinue
    return ($content -match "markitai")
}

# Get project root directory for a development setup: the checkout this script
# lives in, else the current directory when it is one; $null otherwise.
function Get-ProjectRoot {
    if ($PSScriptRoot) {
        $root = Split-Path -Parent $PSScriptRoot
        if (Test-MarkitaiCheckout $root) { return $root }
    }
    if (Test-MarkitaiCheckout $PWD.Path) { return $PWD.Path }
    return $null
}

# Return true when this session is elevated (root under PowerShell 7 on
# Linux or macOS).
function Test-IsAdministrator {
    if (-not $script:OnWindows) {
        try { return ((& id -u 2>$null) -eq "0") } catch { return $false }
    }
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

# Warn when running elevated and say why it matters. An interactive user is
# asked (default no); without interactive input (an SSH command, CI) there is
# nobody to ask, so setup continues after the warning.
function Test-AdminWarning {
    if (-not (Test-IsAdministrator)) { return }

    Clack-Warn (i18n "warn_admin")
    Clack-DetailRow (i18n "warn_admin_risk")
    if (-not (Test-InteractiveInput)) {
        Clack-Info (i18n "info_admin_noninteractive")
        return
    }
    if (-not (Clack-Confirm (i18n "confirm_continue_as_admin") "n")) {
        Clack-Cancel (i18n "setup_cancelled")
        exit 1
    }
}

# Check for WSL environment and warn
function Test-WSLWarning {
    if ($env:WSL_DISTRO_NAME) {
        Clack-Warn (i18n "warn_wsl")
        Clack-DetailRow (i18n "warn_wsl_tip")
        Clack-Command "curl -fsSL https://markitai.dev/setup.sh | sh"

        if (-not (Clack-Confirm (i18n "confirm_continue") "n")) {
            Clack-Cancel (i18n "setup_cancelled")
            exit 1
        }
    }
}

# Check for proxy environment variables
function Test-Proxy {
    if ($env:HTTPS_PROXY -or $env:HTTP_PROXY -or $env:ALL_PROXY -or
        $env:https_proxy -or $env:http_proxy -or $env:all_proxy) {
        return $true
    }
    return $false
}

# Time-boxed reachability probe of the default package index.
#
# This must stay an APPLICATION-layer check. A bare TCP/port probe
# (Test-NetConnection, TcpClient) is meaningless on a machine running a
# TUN-mode proxy: every port answers, so every probe "succeeds". An HTTP
# status alone is not enough either, because an intermittently empty body
# still returns 200. So: real GET, verify status *and* that bytes arrived.
#
# The timeout doubles as the "too slow to install through" threshold.
#
# Env: MARKITAI_INDEX_PROBE_URL / MARKITAI_INDEX_PROBE_TIMEOUT (test seams).
# Returns: $true when the default index answered in time.
function Test-DefaultIndexReachable {
    $url = if ($env:MARKITAI_INDEX_PROBE_URL) {
        $env:MARKITAI_INDEX_PROBE_URL
    } else {
        "https://pypi.org/simple/markitai/"
    }
    $timeout = 3
    if ($env:MARKITAI_INDEX_PROBE_TIMEOUT -match "^\d+$") {
        $timeout = [int]$env:MARKITAI_INDEX_PROBE_TIMEOUT
    }

    $previousProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        # Windows PowerShell 5.1 still negotiates TLS 1.0 by default on some
        # builds; pypi.org requires 1.2 or better.
        try {
            [Net.ServicePointManager]::SecurityProtocol =
                [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        } catch {}

        $response = Invoke-WebRequest -Uri $url -UseBasicParsing `
            -TimeoutSec $timeout -ErrorAction Stop
        if ([int]$response.StatusCode -ne 200) { return $false }

        $length = 0
        try { $length = [int]$response.RawContentLength } catch { $length = 0 }
        if ($length -le 0 -and $response.Content) { $length = $response.Content.Length }
        return ($length -gt 0)
    } catch {
        return $false
    } finally {
        $ProgressPreference = $previousProgress
    }
}

# Offer China mirrors only when this machine has an actual problem reaching
# the default index.
#
# The previous rule was "no proxy variable set", which is true for most of
# the planet, so the very first thing a user in Berlin or Sao Paulo saw was a
# warning about Chinese mirrors that had nothing to do with them. Locale and
# timezone are no better: they misfire on travellers, expats and CI runners.
# Measured reachability is the only evidence that actually predicts the
# problem mirrors solve.
function Configure-Mirrors {
    # Explicit override in either direction wins: no probe, no question.
    if ($env:MARKITAI_USE_MIRROR -match "^(1|true|yes|on)$") {
        Select-MirrorSource
        return
    }
    if ($env:MARKITAI_USE_MIRROR -match "^(0|false|no|off)$") { return }

    # A configured proxy is the user's own routing decision.
    if (Test-Proxy) { return }

    # Never ask without interactive input, and never spend the probe budget there.
    if (-not (Test-InteractiveInput)) { return }

    # Fast, healthy index: say nothing at all and use the default.
    if (Test-DefaultIndexReachable) { return }

    Clack-Warn (i18n "mirror_slow_index")

    if (-not (Clack-Confirm (i18n "mirror_confirm") "y")) {
        Clack-Info (i18n "mirror_skipped")
        return
    }

    Select-MirrorSource
}

# Pick a mirror and export it. Reached from an accepted prompt or from an
# explicit MARKITAI_USE_MIRROR=1.
function Select-MirrorSource {
    # The choices first, then the question, so the cursor waits after it.
    Clack-DetailRow "1. $(i18n 'mirror_tuna')"
    Clack-DetailRow "2. $(i18n 'mirror_aliyun')"
    Clack-DetailRow "3. $(i18n 'mirror_tencent')"
    Clack-DetailRow "4. $(i18n 'mirror_huawei')"
    Write-GuideRow -Mark $S_ASK -MarkColor $script:C_ACCENT -Text (i18n "mirror_select") `
        -Suffix "[1]" -SuffixColor $script:C_DIM -Reserve 5 -NoNewline
    $choice = Read-ClackAnswer -Default "1"

    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

    switch ($choice) {
        "2" {
            $env:UV_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
            $env:NPM_CONFIG_REGISTRY = "https://registry.npmmirror.com"
        }
        "3" {
            $env:UV_INDEX_URL = "https://mirrors.cloud.tencent.com/pypi/simple"
            $env:NPM_CONFIG_REGISTRY = "https://mirrors.cloud.tencent.com/npm/"
        }
        "4" {
            $env:UV_INDEX_URL = "https://repo.huaweicloud.com/repository/pypi/simple"
            $env:NPM_CONFIG_REGISTRY = "https://mirrors.huaweicloud.com/repository/npm/"
        }
        default {
            $env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
            $env:NPM_CONFIG_REGISTRY = "https://registry.npmmirror.com"
        }
    }

    # Playwright: only npmmirror CDN provides reliable browser binary mirrors
    $env:PLAYWRIGHT_DOWNLOAD_HOST = "https://cdn.npmmirror.com/binaries/playwright"

    Clack-Success (i18n "mirror_enabled")
    Clack-DetailRow "$(i18n 'mirror_pypi'): $env:UV_INDEX_URL"
    Clack-DetailRow "$(i18n 'mirror_npm'): $env:NPM_CONFIG_REGISTRY"
    Clack-DetailRow "$(i18n 'mirror_playwright'): $env:PLAYWRIGHT_DOWNLOAD_HOST"
}

# Check execution policy
function Test-ExecutionPolicy {
    if (-not $script:OnWindows) { return $true }
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Clack-Warn ((i18n "warn_exec_policy") -f $policy)
        Clack-Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        return $false
    }
    return $true
}

# ============================================================
# Mode Detection
# ============================================================

# Check whether the current directory is the root of a markitai checkout,
# where a development environment can be set up.
function Test-DevMode {
    # Note: .git can be a directory (normal repo) or file (worktree)
    if ((Test-Path "./pyproject.toml") -and (Test-Path "./.git") -and (Test-Path "./scripts")) {
        $content = Get-Content "./pyproject.toml" -Raw -ErrorAction SilentlyContinue
        if ($content -match "markitai") {
            return $true
        }
    }
    return $false
}

# What setup does: "install" (markitai for this user) or "dev" (sync the
# checkout, pre-commit hooks). Someone who cloned the repo and runs the
# installer from its root most likely wants markitai installed, so a checkout
# root no longer switches to dev mode silently: it asks, defaulting to install.
$script:SETUP_MODE = "install"
$script:CHECKOUT_ANNOUNCED = $false

# Sets: $script:SETUP_MODE (from MARKITAI_SETUP_MODE or the question)
function Select-SetupMode {
    $requested = ("" + $env:MARKITAI_SETUP_MODE).Trim().ToLowerInvariant()
    if ($requested -eq "install" -or $requested -eq "user") {
        $script:SETUP_MODE = "install"
    } elseif ($requested -eq "dev" -or $requested -eq "development") {
        $script:SETUP_MODE = "dev"
    } else {
        if ($requested) {
            Clack-Warn "$(i18n 'warn_unknown_mode'): $($env:MARKITAI_SETUP_MODE)"
        }
        if (Test-DevMode) {
            $script:CHECKOUT_ANNOUNCED = $true
            Clack-Info (i18n "repo_root_detected")
            if (-not (Test-InteractiveInput)) {
                Clack-Info (i18n "info_mode_default")
                return
            }
            if (Clack-Confirm (i18n "confirm_dev_mode") "n") {
                $script:SETUP_MODE = "dev"
            }
        }
    }

    if ($script:SETUP_MODE -eq "dev" -and -not (Get-ProjectRoot)) {
        Clack-Error (i18n "error_dev_needs_repo")
        Clack-Cancel (i18n "error_setup_failed")
        exit 1
    }

    if ($script:SETUP_MODE -eq "dev") {
        Clack-Info (i18n "mode_dev")
    } else {
        Clack-Info (i18n "mode_user")
    }
}

# ============================================================
# Install Source Selection (PyPI vs local repo)
# ============================================================

# Install source: "pypi" (default) or "local" (from source repo)
$script:MARKITAI_SOURCE = "pypi"
$script:MarkitaiLocalPath = $null

# Detect the markitai source repo (for the local install option)
# Checks the script location first, then cwd (irm | iex case)
# Returns: repo root path if detected, $null otherwise
function Get-MarkitaiRepoRoot {
    # 1) Via script location ($PSCommandPath is empty under irm | iex)
    if ($PSCommandPath) {
        $scriptDir = Split-Path -Parent $PSCommandPath
        $repoRoot = Split-Path -Parent $scriptDir
        $pyproject = Join-Path $repoRoot "packages/markitai/pyproject.toml"
        if (Test-Path $pyproject) {
            $content = Get-Content $pyproject -Raw -ErrorAction SilentlyContinue
            if ($content -match 'name\s*=\s*"markitai"') {
                return $repoRoot
            }
        }
    }

    # 2) Via current directory (irm | iex run from inside a checkout)
    $pyproject = Join-Path $PWD.Path "packages/markitai/pyproject.toml"
    if (Test-Path $pyproject) {
        $content = Get-Content $pyproject -Raw -ErrorAction SilentlyContinue
        if ($content -match 'name\s*=\s*"markitai"') {
            return $PWD.Path
        }
    }

    return $null
}

# Ask whether to install markitai from the local repo or PyPI
# Default (enter / non-interactive / not in repo) = PyPI
# Sets: $script:MARKITAI_SOURCE, $script:MarkitaiLocalPath
function Select-InstallSource {
    $repoRoot = Get-MarkitaiRepoRoot
    if (-not $repoRoot) { return }

    # Non-interactive (stdin redirected): never prompt, keep PyPI
    if (-not (Test-InteractiveInput)) {
        Clack-Info (i18n "info_repo_noninteractive")
        return
    }

    if (-not $script:CHECKOUT_ANNOUNCED) {
        Clack-Info "$(i18n 'repo_detected'): $repoRoot"
    }
    if (Clack-Confirm (i18n "confirm_local_install") "n") {
        $script:MARKITAI_SOURCE = "local"
        $script:MarkitaiLocalPath = Join-Path $repoRoot "packages/markitai"
        Clack-Info (i18n "source_local")
    } else {
        Clack-Info (i18n "source_pypi")
    }
}

# Build the markitai package spec honoring install source and extras
function Get-MarkitaiPkgSpec {
    param([string]$Extras)

    $base = "markitai"
    if ($Extras) {
        $base = "markitai[$Extras]"
    }

    if ($script:MARKITAI_SOURCE -eq "local") {
        return "$base @ $($script:MarkitaiLocalPath)"
    }
    if ($script:MarkitaiVersion) {
        return "$base==$($script:MarkitaiVersion)"
    }
    return $base
}

# ============================================================
# Installation Functions
# ============================================================

# Check if uv is installed
function Test-UV {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        return $false
    }
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $version = & uv --version 2>&1 | Select-Object -First 1
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($version -and $version -notmatch "error") {
        return $true
    }
    return $false
}

# Install uv package manager
function Install-UV {
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version ($(i18n 'already_installed'))"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    if (-not (Clack-Confirm (i18n "confirm_uv") "y")) {
        Clack-Skip "$(i18n 'uv') $(i18n 'skipped')"
        Track-Install -Component "uv" -Status "skipped"
        return $false
    }

    Clack-Step "$(i18n 'installing') $(i18n 'uv')..."

    # Build install URL (with optional version)
    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    $installResult = Invoke-PowerShellInstallerQuietly -Uri $uvUrl
    if (-not $installResult.Success) {
        Clack-Error "$(i18n 'uv') $(i18n 'failed')"
        Clack-Detail -Detail $installResult.Output -MaxLines 4
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    try {
        # Refresh PATH (User paths take precedence)
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Clack-Warn (i18n "warn_uv_restart")
            Track-Install -Component "uv" -Status "installed"
            return $false
        }

        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version ($(i18n 'installed'))"
        Track-Install -Component "uv" -Status "installed"
        return $true
    } catch {
        Clack-Error "$(i18n 'uv') $(i18n 'failed')"
        Clack-Detail -Detail @($installResult.Output + $_.Exception.Message) -MaxLines 4
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

# Detect/install Python via uv
function Install-Python {
    if (-not (Test-CommandExists "uv")) {
        Clack-Error (i18n "error_uv_required")
        return $false
    }

    # Try to find any supported Python (3.11-3.13)
    $uvPython = & uv python find ">=3.11,<3.14" 2>$null
    if ($uvPython -and (Test-Path $uvPython)) {
        $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
        if ($version) {
            $script:PYTHON_CMD = $uvPython
            Clack-Success "$(i18n 'python') $version"
            return $true
        }
    }

    # Not found, auto-install (3.13 as default)
    Clack-Step "$(i18n 'installing') $(i18n 'python') 3.13..."
    $installResult = Invoke-NativeQuietly { & uv python install 3.13 }
    if ($installResult.Success) {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Clack-Success "$(i18n 'python') $version $(i18n 'installed')"
                return $true
            }
        }
    }

    Clack-Error (i18n "error_python_required")
    Clack-Detail -Detail $installResult.Output -MaxLines 4
    return $false
}

function Get-MarkitaiToolsDir {
    $uvToolsDir = $null
    try { $uvToolsDir = & uv tool dir 2>$null } catch {}
    if ($uvToolsDir) { return $uvToolsDir }
    if ($env:WSL_DISTRO_NAME) { return "$HOME/.local/share/uv/tools" }
    return "$env:APPDATA\uv\tools"
}

# Put the folder uv installs tool commands into on PATH for the rest of the
# run. UV_TOOL_BIN_DIR or XDG_BIN_HOME can move it, and the version check,
# `doctor --suggest-extras` and `init` all run markitai.
function Add-UvToolBinToPath {
    $binDir = $null
    try { $binDir = & uv tool dir --bin 2>$null | Select-Object -First 1 } catch {}
    if (-not $binDir) { return }
    $script:UvToolBin = "$binDir".Trim()
    $separator = [IO.Path]::PathSeparator
    $entries = @($env:PATH -split [regex]::Escape([string]$separator))
    if ($entries -notcontains $script:UvToolBin) {
        $env:PATH = "$($script:UvToolBin)$separator$env:PATH"
    }
}

# Say so when markitai's command folder is not on the PATH a new shell gets
# (on Windows the saved user and machine PATH): otherwise "Getting started"
# recommends commands the shell cannot find.
function Show-UvToolBinPathWarning {
    if (-not $script:UvToolBin) { return }
    $separator = [IO.Path]::PathSeparator
    $pathText = $script:OriginalPath
    if ($script:OnWindows) {
        $pathText = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "Machine")
    }
    $entries = @("$pathText" -split [regex]::Escape([string]$separator) |
        ForEach-Object { $_.TrimEnd('\', '/') })
    if ($entries -contains $script:UvToolBin.TrimEnd('\', '/')) { return }
    Clack-Warn "$(i18n 'warn_bin_off_path'): $($script:UvToolBin)"
    Clack-DetailRow (i18n "info_bin_off_path_fix")
    Clack-Command "uv tool update-shell"
}

# Install markitai (User mode)
function Install-Markitai {
    $uvToolsDir = Get-MarkitaiToolsDir

    # Build package spec with all tracked extras
    $pkg = Get-MarkitaiPkgSpec -Extras $script:MARKITAI_EXTRAS

    # Build Python command for --python argument
    $pythonArg = $script:PYTHON_CMD

    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvExists) {
        Clack-Error "$(i18n 'markitai') $(i18n 'failed')"
        Track-Install -Component "markitai" -Status "failed"
        return $false
    }

    $markitaiToolDir = Join-Path $uvToolsDir "markitai"
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 1
    $lastOutput = @()

    if (Test-Path $markitaiToolDir) {
        # Say something before a step that can take a minute: the sh script
        # spins here, and this branch was the only silent one.
        Clack-Step "$(i18n 'installing') $(i18n 'markitai')..."
        # Only an unpinned PyPI install can use the receipt-based upgrade path.
        # Explicit versions and local sources must apply the exact spec below.
        if (
            $script:MARKITAI_SOURCE -ne "local" -and -not $script:MarkitaiVersion -and
            -not (Test-MarkitaiExtrasNeedUpdate)
        ) {
            try {
                $lastOutput = @(& uv tool upgrade markitai 2>&1)
                $exitCode = $LASTEXITCODE
            } catch {}
        }

        # Upgrade failed (or local source), try force reinstall
        if ($exitCode -ne 0) {
            try {
                $lastOutput = @(& uv tool install $pkg --python $pythonArg --force 2>&1)
                $exitCode = $LASTEXITCODE
            } catch {}
        }
    } else {
        # Fresh install
        Clack-Step "$(i18n 'installing') $(i18n 'markitai')..."
        try {
            $lastOutput = @(& uv tool install $pkg --python $pythonArg 2>&1)
            $exitCode = $LASTEXITCODE
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction

    if ($exitCode -eq 0) {
        # Refresh PATH
        if ($script:OnWindows) {
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path
        }
        Add-UvToolBinToPath

        $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
        $version = if ($markitaiCmd) { (& markitai --version 2>&1 | Select-Object -First 1).Split(' ')[-1] } else { (i18n "installed") }
        if (-not $version) { $version = (i18n "installed") }
        if (
            $script:MARKITAI_SOURCE -ne "local" -and
            $script:MarkitaiVersion -and
            $version -ne $script:MarkitaiVersion
        ) {
            Clack-Error "$(i18n 'markitai') version mismatch: expected $($script:MarkitaiVersion), got $version"
            Track-Install -Component "markitai" -Status "failed"
            return $false
        }
        Clack-Success "$(i18n 'markitai') $version"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Clack-Error "$(i18n 'markitai') $(i18n 'failed')"
    Clack-Detail -Detail $lastOutput -MaxLines 6
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# Global variable tracking all needed extras (comma-separated). A fresh
# non-interactive install starts with core only unless explicitly opted in.
$script:MARKITAI_EXTRAS = ""
$script:MARKITAI_RECEIPT_EXTRAS = @()
$script:MARKITAI_ALL_FALLBACK_EXTRAS = "browser,extra-fetch,svg,heif,legacy,mcp,ocr,serve"
# Only drop explicitly retired capabilities; preserve other receipt extras,
# including extensions supplied by a local/custom package.
$script:MARKITAI_RETIRED_EXTRAS = "kreuzberg"
# Extras the user explicitly turned down. `markitai doctor --suggest-extras`
# recommends `ocr` unconditionally, so without this list the finalize pass
# would reinstall exactly what was just declined and make the prompt a lie.
$script:MARKITAI_DECLINED_EXTRAS = @()
if ((Test-InteractiveInput) -or (Test-OptionalInstallRequested)) {
    $script:MARKITAI_EXTRAS = "browser"
}

# Return true when an extra is covered by the combined spec. `all` is
# canonical and includes every individual extra, including `serve`.
function Test-MarkitaiExtraEnabled {
    param([string]$ExtraName)

    if ($script:MARKITAI_EXTRAS -eq "all") { return $true }
    return (($script:MARKITAI_EXTRAS -split ",") -contains $ExtraName)
}

function Test-MarkitaiExtraRetired {
    param([string]$ExtraName)

    # These extras were retired in 1.0.0. An explicit pre-1.0 PyPI pin still
    # supports them; a local source takes precedence over the version pin.
    if ($script:MARKITAI_SOURCE -ne "local" -and $script:MarkitaiVersion -match '^0\.') {
        return $false
    }
    return (($script:MARKITAI_RETIRED_EXTRAS -split ",") -contains $ExtraName)
}

# Track a markitai extra for the next combined installation.
function Install-MarkitaiExtra {
    param([string]$ExtraName)

    if ([string]::IsNullOrWhiteSpace($ExtraName)) { return }
    if (Test-MarkitaiExtraRetired -ExtraName $ExtraName) { return }
    if ($ExtraName -eq "all") {
        $script:MARKITAI_EXTRAS = "all"
        return
    }
    if (Test-MarkitaiExtraEnabled -ExtraName $ExtraName) { return }
    if ([string]::IsNullOrEmpty($script:MARKITAI_EXTRAS)) {
        $script:MARKITAI_EXTRAS = $ExtraName
    } else {
        $script:MARKITAI_EXTRAS = "$($script:MARKITAI_EXTRAS),$ExtraName"
    }
}

# Return true when the user answered "no" to an extra in this run.
function Test-MarkitaiExtraDeclined {
    param([string]$ExtraName)

    return ($script:MARKITAI_DECLINED_EXTRAS -contains $ExtraName)
}

# Remember a declined extra so no later suggestion pass re-adds it.
function Deny-MarkitaiExtra {
    param([string]$ExtraName)

    if (Test-MarkitaiExtraDeclined -ExtraName $ExtraName) { return }
    $script:MARKITAI_DECLINED_EXTRAS += $ExtraName
}

# Preserve non-retired extras before asking for new capabilities. Keep the
# raw receipt too: retired entries require a reinstall even with no additions.
function Import-MarkitaiReceiptExtras {
    $uvToolsDir = Get-MarkitaiToolsDir
    $receiptFile = Join-Path $uvToolsDir "markitai/uv-receipt.toml"
    $script:MARKITAI_RECEIPT_EXTRAS = @()
    if (-not (Test-Path $receiptFile)) { return }

    $receipt = Get-Content $receiptFile -Raw -ErrorAction SilentlyContinue
    if ($receipt -match 'extras\s*=\s*\[([^\]]*)\]') {
        $extrasStr = $Matches[1] -replace '"', '' -replace '\s', ''
        $script:MARKITAI_RECEIPT_EXTRAS = @(
            $extrasStr -split ',' | Where-Object { $_ }
        )
        foreach ($extra in $script:MARKITAI_RECEIPT_EXTRAS) {
            Install-MarkitaiExtra -ExtraName $extra
        }
    }
}

function Test-MarkitaiReceiptHasExtra {
    param([string]$ExtraName)

    return (
        ($script:MARKITAI_RECEIPT_EXTRAS -contains "all") -or
        ($script:MARKITAI_RECEIPT_EXTRAS -contains $ExtraName)
    )
}

# A generic `uv tool upgrade` reuses the old requirements. Retired extras or
# additions need `uv tool install` with the cleaned spec, including core-only.
function Test-MarkitaiExtrasNeedUpdate {
    foreach ($extra in $script:MARKITAI_RECEIPT_EXTRAS) {
        if (Test-MarkitaiExtraRetired -ExtraName $extra) {
            return $true
        }
    }
    foreach ($extra in ($script:MARKITAI_EXTRAS -split ",")) {
        if ($extra -and -not (Test-MarkitaiReceiptHasExtra -ExtraName $extra)) {
            return $true
        }
    }
    return $false
}

# Resolve Web UI support before installing markitai so `serve` joins the same
# package spec as browser/all/existing extras instead of replacing them later.
function Select-MarkitaiServe {
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") { return }
    if (Confirm-OptionalInstall (i18n "confirm_serve") "y") {
        Install-MarkitaiExtra -ExtraName "serve"
    } else {
        Deny-MarkitaiExtra -ExtraName "serve"
    }
}

function Track-MarkitaiServe {
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") {
        Track-Install -Component "serve" -Status "installed"
    } else {
        Track-Install -Component "serve" -Status "skipped"
    }
}

# Resolve OCR before installing markitai, for the same reason as `serve`.
# rapidocr moved out of the core dependencies: it is ~150MB of models that
# nobody converting born-digital documents ever executes, so the guided
# installer asks instead of deciding. A repeat run skips the question when the
# uv receipt already records the extra.
function Select-MarkitaiOcr {
    if (Test-MarkitaiExtraEnabled -ExtraName "ocr") { return }
    if (Confirm-OptionalInstall (i18n "confirm_ocr") "y") {
        Install-MarkitaiExtra -ExtraName "ocr"
    } else {
        Deny-MarkitaiExtra -ExtraName "ocr"
    }
}

function Track-MarkitaiOcr {
    if (Test-MarkitaiExtraEnabled -ExtraName "ocr") {
        Track-Install -Component "ocr" -Status "installed"
    } else {
        Track-Install -Component "ocr" -Status "skipped"
    }
}

# Finalize markitai extras after all optional components are resolved.
# Merges `markitai doctor --suggest-extras` output with manually tracked
# MARKITAI_EXTRAS (from CLI install functions), so nothing is lost.
function Finalize-MarkitaiExtras {
    if (-not (Test-InteractiveInput) -and -not (Test-OptionalInstallRequested)) {
        return
    }

    # Merge suggested extras INTO manually tracked set (not replace)
    try {
        $suggested = & markitai doctor --suggest-extras 2>$null
        if ($suggested) {
            foreach ($extra in ($suggested.Trim() -split ',')) {
                if ($extra -and -not (Test-MarkitaiExtraDeclined -ExtraName $extra)) {
                    Install-MarkitaiExtra -ExtraName $extra
                }
            }
        }
    } catch {}

    # Refresh the receipt after the initial install, then compare exact extra
    # names (with `all` treated as a superset).
    Import-MarkitaiReceiptExtras
    if (-not (Test-MarkitaiExtrasNeedUpdate)) { return }

    # Reinstall with all extras (progressive fallback on failure)
    $pkg = Get-MarkitaiPkgSpec -Extras $script:MARKITAI_EXTRAS

    Clack-Step "$(i18n 'installing') $(i18n 'markitai') extras..."
    $primaryResult = Invoke-NativeQuietly {
        & uv tool install $pkg --python $script:PYTHON_CMD --force
    }
    $installOk = $primaryResult.Success
    if ($installOk) {
        Clack-Success "$(i18n 'markitai') extras: $($script:MARKITAI_EXTRAS -replace ',', ', ')"
    }
    $fallbackResult = $null

    if (-not $installOk) {
        # Full install failed: retry without SDK-dependent extras
        $sdkExtras = @("claude-agent", "copilot")
        $safeExtras = @()
        $skipped = @()
        foreach ($e in ($script:MARKITAI_EXTRAS -split ",")) {
            if ($e -eq "all") {
                $safeExtras += ($script:MARKITAI_ALL_FALLBACK_EXTRAS -split ",")
                $skipped += $sdkExtras
            } elseif ($sdkExtras -contains $e) {
                $skipped += $e
            } else {
                $safeExtras += $e
            }
        }
        if ($safeExtras.Count -gt 0) {
            $safeList = $safeExtras -join ","
            $pkg = Get-MarkitaiPkgSpec -Extras $safeList
            $fallbackResult = Invoke-NativeQuietly {
                & uv tool install $pkg --python $script:PYTHON_CMD --force
            }
            if ($fallbackResult.Success) {
                Clack-Success "$(i18n 'markitai') extras: $($safeList -replace ',', ', ')"
                if ($skipped.Count -gt 0) {
                    Clack-Warn "$(i18n 'skipped') extras: $($skipped -join ', ') (SDK $(i18n 'not_found'))"
                }
                $installOk = $true
            }
        }
        if (-not $installOk) {
            Clack-Warn "$(i18n 'markitai') extras update $(i18n 'failed')"
            $details = @($primaryResult.Output)
            if ($fallbackResult) { $details += $fallbackResult.Output }
            Clack-Detail -Detail $details -MaxLines 3 -Context "markitai extras"
        }
    }
}

# Sync project dependencies (Dev mode)
function Sync-Dependencies {
    $projectRoot = Get-ProjectRoot
    Clack-Info "$(i18n 'info_project_dir'): $projectRoot"

    Push-Location $projectRoot

    try {
        Clack-Step (i18n "info_syncing_deps")
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $syncResult = & uv sync --all-groups --all-extras --python $script:PYTHON_CMD 2>&1
        $ErrorActionPreference = $oldErrorAction
        if ($LASTEXITCODE -eq 0) {
            Clack-Success (i18n "info_deps_synced")
            Track-Install -Component "python_deps" -Status "installed"
            return $true
        } else {
            Clack-Error "$(i18n 'python_deps') $(i18n 'failed')"
            Clack-Detail -Detail $syncResult -MaxLines 6
            Track-Install -Component "python_deps" -Status "failed"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Install pre-commit hooks (Dev mode)
function Install-PreCommit {
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                $oldErrorAction = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $precommitResult = & uv run pre-commit install 2>&1
                $ErrorActionPreference = $oldErrorAction
                if ($LASTEXITCODE -eq 0) {
                    Clack-Success (i18n "info_precommit_installed")
                    Track-Install -Component "precommit" -Status "installed"
                    return $true
                } else {
                    Clack-Warn "$(i18n 'precommit') $(i18n 'failed')"
                    Clack-Detail -Detail $precommitResult -MaxLines 3
                    return $false
                }
            }
        } else {
            Clack-Skip "$(i18n 'precommit') $(i18n 'skipped')"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Check if Playwright browser is installed
function Test-PlaywrightBrowser {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
                if (Test-Path $markitaiPlaywright) {
                    $cacheDir = Join-Path $env:LOCALAPPDATA "ms-playwright"
                    if (Test-Path $cacheDir) {
                        $chromiumDirs = Get-ChildItem -Path $cacheDir -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue
                        if ($chromiumDirs) {
                            return $true
                        }
                    }
                }
            }
        } catch {}
    }
    return $false
}

# Install Playwright browser (Optional)
function Install-OptionalPlaywright {
    if (Test-PlaywrightBrowser) {
        Clack-Success "$(i18n 'playwright') $(i18n 'already_installed')"
        Track-Install -Component "playwright" -Status "installed"
        return $true
    }

    Clack-Info (i18n "info_playwright_purpose")

    if (-not (Confirm-OptionalInstall (i18n "confirm_playwright") "y")) {
        Clack-Skip "$(i18n 'playwright') $(i18n 'skipped')"
        Track-Install -Component "playwright" -Status "skipped"
        return $false
    }

    Clack-Step "$(i18n 'installing') $(i18n 'playwright')..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $lastOutput = @()

    # Method 1: Use playwright from markitai's uv tool environment
    $markitaiPlaywright = $null
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
            }
        } catch {}
    }
    if (-not $markitaiPlaywright -or -not (Test-Path $markitaiPlaywright)) {
        $markitaiPlaywright = Join-Path $env:APPDATA "uv\tools\markitai\Scripts\playwright.exe"
    }

    if (Test-Path $markitaiPlaywright) {
        try {
            $lastOutput = @(& $markitaiPlaywright install chromium 2>&1)
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "$(i18n 'playwright') $(i18n 'installed')"
                Track-Install -Component "playwright" -Status "installed"
                return $true
            }
        } catch {}
    }

    # Method 2: Fallback to Python module
    if ($script:PYTHON_CMD) {
        try {
            $lastOutput = @(& $script:PYTHON_CMD -m playwright install chromium 2>&1)
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "$(i18n 'playwright') $(i18n 'installed')"
                Track-Install -Component "playwright" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Clack-Error "$(i18n 'playwright') $(i18n 'failed')"
    Clack-Detail -Detail $lastOutput -MaxLines 4
    Track-Install -Component "playwright" -Status "failed"
    return $false
}

# Install Claude Code CLI (Optional)
function Install-OptionalClaudeCLI {
    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "$(i18n 'claude_cli'): $version $(i18n 'already_installed')"
        Install-MarkitaiExtra -ExtraName "claude-agent"
        Track-Install -Component "claude_cli" -Status "installed"
        return $true
    }

    if (-not (Confirm-OptionalInstall (i18n "confirm_claude_cli") "n")) {
        Clack-Skip "$(i18n 'claude_cli') $(i18n 'skipped')"
        Track-Install -Component "claude_cli" -Status "skipped"
        return $false
    }

    Clack-Step "$(i18n 'installing') $(i18n 'claude_cli')..."

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    $installResult = Invoke-PowerShellInstallerQuietly -Uri $claudeUrl
    $lastDetail = $installResult.Output
    if ($installResult.Success) {
        # Child-process PATH changes do not flow back; reload persisted paths.
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            # Also install the SDK extra
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }

    # Fallback: bun/pnpm/npm
    $bunCmd = Get-Command bun -ErrorAction SilentlyContinue
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($bunCmd) {
        $lastResult = Invoke-NativeQuietly { & bun add -g @anthropic-ai/claude-code }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }
    elseif ($pnpmCmd) {
        $lastResult = Invoke-NativeQuietly { & pnpm add -g @anthropic-ai/claude-code }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $lastResult = Invoke-NativeQuietly { & npm install -g @anthropic-ai/claude-code }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'claude_cli') $(i18n 'failed')"
    Clack-Detail -Detail $lastDetail -MaxLines 3
    Track-Install -Component "claude_cli" -Status "failed"
    return $false
}

# Install Copilot CLI (Optional)
function Install-OptionalCopilotCLI {
    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "$(i18n 'copilot_cli'): $version $(i18n 'already_installed')"
        Install-MarkitaiExtra -ExtraName "copilot"
        Track-Install -Component "copilot_cli" -Status "installed"
        return $true
    }

    if (-not (Confirm-OptionalInstall (i18n "confirm_copilot_cli") "n")) {
        Clack-Skip "$(i18n 'copilot_cli') $(i18n 'skipped')"
        Track-Install -Component "copilot_cli" -Status "skipped"
        return $false
    }

    Clack-Step "$(i18n 'installing') $(i18n 'copilot_cli')..."

    # Prefer WinGet on Windows
    $lastResult = $null
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $lastResult = Invoke-NativeQuietly { & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            # Also install the SDK extra
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }

    # Fallback: bun/pnpm/npm
    $bunCmd = Get-Command bun -ErrorAction SilentlyContinue
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($bunCmd) {
        $lastResult = Invoke-NativeQuietly { & bun add -g @github/copilot }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }
    elseif ($pnpmCmd) {
        $lastResult = Invoke-NativeQuietly { & pnpm add -g @github/copilot }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $lastResult = Invoke-NativeQuietly { & npm install -g @github/copilot }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'copilot_cli') $(i18n 'failed')"
    if ($lastResult) { Clack-Detail -Detail $lastResult.Output -MaxLines 3 }
    Track-Install -Component "copilot_cli" -Status "failed"
    return $false
}

# Print installation summary: one row per component, its mark saying what
# happened (installed, skipped on purpose, failed).
function Print-Summary {
    param([switch]$NoHeader)
    if (-not $NoHeader) { Clack-Section (i18n "section_summary") }
    foreach ($comp in $script:INSTALLED_COMPONENTS) {
        Clack-Success (i18n $comp)
    }
    foreach ($comp in $script:SKIPPED_COMPONENTS) {
        Clack-Skip "$(i18n $comp) $(i18n 'skipped')"
    }
    foreach ($comp in $script:FAILED_COMPONENTS) {
        Clack-Error "$(i18n $comp) $(i18n 'failed')"
    }
}

# Documentation and issue links close both completion notes.
function Print-HelpLinks {
    Clack-Info "$(i18n 'info_docs'): https://markitai.dev"
    Clack-Info "$(i18n 'info_issues'): https://github.com/Ynewtime/markitai/issues"
}

# Print user mode completion message
# Returns $true when the `mkai` short alias resolves to markitai, $false when
# it is absent or shadowed by a different command (which is warned about).
# The full `markitai` command always works.
function Test-MkaiAlias {
    if (-not (Get-Command mkai -ErrorAction SilentlyContinue)) { return $false }
    $ver = (& mkai --version 2>$null) -join ' '
    if ($ver -match 'markitai') { return $true }
    $path = (Get-Command mkai -ErrorAction SilentlyContinue).Source
    Clack-Warn "$(i18n 'mkai_conflict'): $path"
    return $false
}

function Print-UserCompletion {
    $mkaiUsable = Test-MkaiAlias
    Clack-Section (i18n "getting_started")
    Show-UvToolBinPathWarning
    Clack-Hint (i18n "convert_file") "markitai file.pdf"
    Clack-Hint (i18n "interactive_mode") "markitai -I"
    # Only advertise the serve command when its dependencies were selected.
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") {
        Clack-Hint (i18n "start_web_ui") "markitai serve"
    }
    Clack-Hint (i18n "configure_llm") "markitai init"
    Clack-Hint (i18n "show_help") "markitai --help"
    # Only advertise the short alias when it actually resolves to markitai.
    if ($mkaiUsable) {
        Clack-Info (i18n "short_alias")
    }
    Print-HelpLinks
}

# Print dev mode completion message
function Print-DevCompletion {
    Clack-Section (i18n "quick_start")
    Clack-Hint (i18n "configure_env") "copy .env.example .env"
    Clack-Hint (i18n "interactive_mode") "uv run markitai -I"
    Clack-Hint (i18n "start_web_ui") "uv run markitai serve"
    Clack-Hint (i18n "run_tests") "uv run pytest"
    Clack-Hint (i18n "run_cli") "uv run markitai --help"
    Print-HelpLinks
}

# Initialize markitai config (silent, first install only)
# Skips when a config already exists: `init --yes` would silently append
# newly detected providers to the user's model_list. The detector only records
# a local CLI after login succeeds and its finalized runtime SDK is importable.
function Initialize-Config {
    if (Test-Path "$HOME/.markitai/config.json") {
        return
    }
    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if ($markitaiExists) {
        try {
            $null = & markitai init --yes 2>$null
        } catch {}
    }
}

# ============================================================
# Main Entry Point
# ============================================================

# User mode main flow
function Run-UserSetup {
    Configure-Mirrors
    Select-InstallSource

    Clack-Section (i18n "section_core")
    if (-not (Install-UV)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Python)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Import-MarkitaiReceiptExtras
    Select-MarkitaiServe
    Select-MarkitaiOcr
    if (-not (Install-Markitai)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Track-MarkitaiServe
    Track-MarkitaiOcr

    Clack-Section (i18n "section_optional")
    Invoke-OptionalStep -Component "playwright" -Action { Install-OptionalPlaywright }

    Clack-Section (i18n "section_llm_cli")
    Invoke-OptionalStep -Component "claude_cli" -Action { Install-OptionalClaudeCLI }
    Invoke-OptionalStep -Component "copilot_cli" -Action { Install-OptionalCopilotCLI }
    # This section reports what is installed; ChatGPT installs nothing, so
    # without this line it looks absent rather than already satisfied.
    Clack-Info (i18n "info_chatgpt_route")

    # The extras step reports in the summary, not under the last section.
    Clack-Section (i18n "section_summary")
    Invoke-BestEffortStep `
        -FailureMessage "$(i18n 'markitai') extras update $(i18n 'failed')" `
        -Action { Finalize-MarkitaiExtras }

    Initialize-Config 2>$null | Out-Null

    Print-Summary -NoHeader
    Print-UserCompletion
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        Clack-OutroWarning (i18n "setup_complete_with_warnings")
    } else {
        Clack-Outro (i18n "setup_complete")
    }
}

# Dev mode main flow
function Run-DevSetup {
    Configure-Mirrors

    Clack-Section (i18n "section_prerequisites")
    if (-not (Install-UV)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Python)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }

    Clack-Section (i18n "section_dev_env")
    if (-not (Sync-Dependencies)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Invoke-OptionalStep -Component "precommit" -Action { Install-PreCommit }

    Clack-Section (i18n "section_optional")
    Invoke-OptionalStep -Component "playwright" -Action { Install-OptionalPlaywright }

    Clack-Section (i18n "section_llm_cli")
    Invoke-OptionalStep -Component "claude_cli" -Action { Install-OptionalClaudeCLI }
    Invoke-OptionalStep -Component "copilot_cli" -Action { Install-OptionalCopilotCLI }
    # This section reports what is installed; ChatGPT installs nothing, so
    # without this line it looks absent rather than already satisfied.
    Clack-Info (i18n "info_chatgpt_route")

    Print-Summary
    Print-DevCompletion
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        Clack-OutroWarning (i18n "dev_setup_complete_with_warnings")
    } else {
        Clack-Outro (i18n "dev_setup_complete")
    }
}

# Main entry point
function Main {
    Clack-Intro (i18n "welcome")
    Test-ExecutionPolicy | Out-Null
    Test-AdminWarning
    Test-WSLWarning
    Select-SetupMode
    if ($script:SETUP_MODE -eq "dev") {
        Run-DevSetup
    } else {
        Run-UserSetup
    }
}

# Run main function behind a final exception boundary so the tree is never
# left open and PowerShell never emits its verbose red stack formatting.
try {
    Main
} catch {
    Complete-UnexpectedFailure -ErrorRecord $_
    exit 1
} finally {
    # Pipeline cancellation (for example Ctrl+C) can bypass catch, but
    # PowerShell still runs finally. Best-effort closure avoids a dangling guide.
    if ($script:CLACK_SESSION_ACTIVE -and -not $script:CLACK_SESSION_CLOSED) {
        try {
            Clack-Error (i18n "error_unexpected")
            Clack-Cancel (i18n "error_setup_failed")
        } catch {}
    }
    # Under irm | iex this is the user's own console: give its encoding back.
    if ($script:OriginalOutputEncoding) {
        try { [Console]::OutputEncoding = $script:OriginalOutputEncoding } catch {}
    }
}
