<#
.SYNOPSIS
Runs the built Markitai.exe on this desktop and proves the workbench end to end.

.DESCRIPTION
Starts the app with an isolated user state (HOME, USERPROFILE, LOCALAPPDATA, APPDATA,
MARKITAI_HOME, TEMP and TMP all inside -State) and repository fixtures as arguments,
hands one more file to it through a second launch (the single-instance redirect),
waits for every conversion to finish, then reads the window through UI Automation
patterns only: rows and their states, the detail's preview and Markdown text,
History, and a Settings toggle that must reach settings.json. It takes screenshots
of Convert, History and Settings, closes the window with WM_CLOSE, and exits 1 if any
piece of evidence is missing. Evidence goes to -Out; summary.txt lists every check.

Runs in Windows PowerShell 5.1 or later, in an interactive session with the display on:

    powershell -File native\windows\smoke.ps1 -State $env:RUNNER_TEMP\markitai-smoke -Out $env:RUNNER_TEMP\markitai-evidence
#>
param(
    [Parameter(Mandatory = $true)][string]$State,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$App = (Join-Path $PSScriptRoot 'Markitai.App\bin\x64\Release\net10.0-windows10.0.26100.0\win-x64'),
    [int]$TimeoutSec = 300
)
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Drawing, System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SmokeNative {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out RECT rect);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
[SmokeNative]::SetProcessDPIAware() | Out-Null

$AE = [System.Windows.Automation.AutomationElement]
$Scope = [System.Windows.Automation.TreeScope]
$Types = [System.Windows.Automation.ControlType]

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $App 'Markitai.exe'
$State = [IO.Path]::GetFullPath($State)
$Out = [IO.Path]::GetFullPath($Out)
$profileRoot = [Environment]::GetFolderPath('UserProfile')
if ($State.StartsWith($profileRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "-State must not be inside the user's profile ($profileRoot)"
}

# --- Evidence ---------------------------------------------------------------

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$log = Join-Path $Out 'smoke.log'
$script:failures = New-Object System.Collections.Generic.List[string]
$script:checks = New-Object System.Collections.Generic.List[string]

function Log([string]$text) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $text"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Check([bool]$ok, [string]$what) {
    if ($ok) { $script:checks.Add("ok   $what"); Log "ok   $what" }
    else { $script:checks.Add("FAIL $what"); $script:failures.Add($what); Log "FAIL $what" }
}

# Each stage runs on its own, so one failure still leaves the others' evidence.
function Stage([string]$name, [scriptblock]$body) {
    Log "== $name"
    try { & $body }
    catch { Check $false "${name}: $($_.Exception.Message)" }
}

# --- Strings: the app runs in English on the runner -------------------------

$strings = Get-Content -Raw -Encoding UTF8 (Join-Path $repo 'native\strings.json') | ConvertFrom-Json
function S([string]$key) {
    $entry = $strings.$key
    if (-not $entry) { throw "native/strings.json has no $key" }
    return [string]$entry.en
}
function Before([string]$key) { return (S $key).Split('{')[0] }

# --- Isolated user state ----------------------------------------------------

if (Test-Path $State) { Remove-Item -Recurse -Force $State }
$userHome = Join-Path $State 'home'
$isolated = [ordered]@{
    HOME          = $userHome
    USERPROFILE   = $userHome
    LOCALAPPDATA  = Join-Path $userHome 'AppData\Local'
    APPDATA       = Join-Path $userHome 'AppData\Roaming'
    MARKITAI_HOME = Join-Path $State 'markitai-home'
    TEMP          = Join-Path $State 'temp'
    TMP           = Join-Path $State 'temp'
}
foreach ($folder in $isolated.Values) { New-Item -ItemType Directory -Force $folder | Out-Null }
$data = Join-Path $isolated.LOCALAPPDATA 'Markitai'
$output = Join-Path $State 'out'
New-Item -ItemType Directory -Force $data | Out-Null

# The Documents known folder ignores USERPROFILE, so the output folder is set
# here; the window size fits a 1024 x 768 runner screen.
$utf8 = New-Object System.Text.UTF8Encoding $false
$seed = [ordered]@{ 'output.folder' = $output; 'window.width' = 1000; 'window.height' = 700 }
[IO.File]::WriteAllText((Join-Path $data 'settings.json'), ($seed | ConvertTo-Json), $utf8)

$formats = Join-Path $repo 'crates\markitai-core\src\formats'
$fixtures = @(
    @{ Path = "$formats\pdf\fixtures\default-table.html"; Done = $true; Warnings = $false },
    @{ Path = "$formats\pdf\fixtures\default-table.pdf"; Done = $true; Warnings = $false },
    @{ Path = "$formats\pdf\fixtures\page-chrome\how-to-cjk-footer.pdf"; Done = $true; Warnings = $true },
    @{ Path = "$formats\native\fixtures\day-plan.xlsx"; Done = $true; Warnings = $false },
    @{ Path = "$formats\native\fixtures\office-images\diagram.png"; Done = $false; Code = 'error.image_only' }
)
$redirected = @{ Path = Join-Path $repo 'crates\markitai-core\tests\fixtures\epub\markitdown-test.epub'; Done = $true; Warnings = $false }
$expectedFolders = @('day-plan', 'default-table', 'default-table (2)', 'how-to-cjk-footer', 'markitdown-test')
foreach ($fixture in $fixtures + $redirected) {
    if (-not (Test-Path $fixture.Path)) { throw "missing fixture $($fixture.Path)" }
}

Log "app $exe"
Log "state $State"
$screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
Log "screen $($screen.Width)x$($screen.Height), session $((Get-Process -Id $PID).SessionId), interactive $([Environment]::UserInteractive)"

function Start-Markitai([string[]]$files) {
    $info = New-Object System.Diagnostics.ProcessStartInfo $exe
    $info.UseShellExecute = $false
    $info.WorkingDirectory = $State
    $info.Arguments = ($files | ForEach-Object { '"' + $_ + '"' }) -join ' '
    foreach ($pair in $isolated.GetEnumerator()) { $info.EnvironmentVariables[$pair.Key] = $pair.Value }
    return [System.Diagnostics.Process]::Start($info)
}

# --- UI Automation (patterns only, no injected input) -----------------------

$script:window = [IntPtr]::Zero
function Root { return $AE::FromHandle($script:window) }

function Condition([string]$name, $type) {
    $byName = New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty, $name)
    if (-not $type) { return $byName }
    $byType = New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, $type)
    return [System.Windows.Automation.AndCondition]::new($byName, $byType)
}

function Find([string]$name, $type = $null, [int]$seconds = 20) {
    $deadline = (Get-Date).AddSeconds($seconds)
    do {
        $element = (Root).FindFirst($Scope::Descendants, (Condition $name $type))
        if ($element) { return $element }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    $kind = if ($type) { $type.ProgrammaticName } else { 'element' }
    throw "UI Automation found no $kind named '$name'"
}

function Pattern($element, $pattern) {
    $found = $null
    if ($element.TryGetCurrentPattern($pattern, [ref]$found)) { return $found }
    throw "'$($element.Current.Name)' has no $($pattern.ProgrammaticName)"
}

function Choose([string]$name) {
    (Pattern (Find $name $Types::ListItem) ([System.Windows.Automation.SelectionItemPattern]::Pattern)).Select()
    Log "selected '$name'"
}

function Rows([string]$listName) {
    $list = Find $listName $Types::List
    $items = $list.FindAll($Scope::Descendants, (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, $Types::ListItem)))
    return @($items | ForEach-Object { $_.Current.Name })
}

function Dump([string]$path) {
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $lines = New-Object System.Collections.Generic.List[string]
    $stack = New-Object System.Collections.Stack
    $stack.Push(@((Root), 0))
    while ($stack.Count -gt 0 -and $lines.Count -lt 3000) {
        $element, $depth = $stack.Pop()
        $current = $element.Current
        $flags = @()
        if ($current.IsOffscreen) { $flags += 'offscreen' }
        if (-not $current.IsEnabled) { $flags += 'disabled' }
        $lines.Add(('  ' * $depth) + "$($current.ControlType.ProgrammaticName -replace 'ControlType.', '') [$($current.Name)] $($flags -join ',')")
        $children = @()
        $child = $walker.GetFirstChild($element)
        while ($child) { $children += $child; $child = $walker.GetNextSibling($child) }
        [array]::Reverse($children)
        foreach ($c in $children) { $stack.Push(@($c, ($depth + 1))) }
    }
    [IO.File]::WriteAllLines($path, $lines, $utf8)
}

function Shot([string]$name) {
    try { (Root).SetFocus() } catch { }
    [SmokeNative]::SetForegroundWindow($script:window) | Out-Null
    Start-Sleep -Milliseconds 800
    $rect = New-Object SmokeNative+RECT
    [SmokeNative]::GetWindowRect($script:window, [ref]$rect) | Out-Null
    $area = [System.Drawing.Rectangle]::Intersect($screen, [System.Drawing.Rectangle]::FromLTRB($rect.Left, $rect.Top, $rect.Right, $rect.Bottom))
    if ($area.Width -le 0 -or $area.Height -le 0) { throw "the window is off screen ($($rect.Left),$($rect.Top),$($rect.Right),$($rect.Bottom))" }
    $bitmap = New-Object System.Drawing.Bitmap $area.Width, $area.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($area.Location, [System.Drawing.Point]::Empty, $area.Size)
    $graphics.Dispose()
    # A blank or black capture has a handful of colours; a rendered window has many.
    $colours = New-Object System.Collections.Generic.HashSet[int]
    for ($y = 0; $y -lt $area.Height; $y += [Math]::Max(1, [int]($area.Height / 40))) {
        for ($x = 0; $x -lt $area.Width; $x += [Math]::Max(1, [int]($area.Width / 40))) {
            [void]$colours.Add($bitmap.GetPixel($x, $y).ToArgb())
        }
    }
    $path = Join-Path $Out "$name.png"
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
    Check ($colours.Count -ge 8) "screenshot $name.png, $($area.Width)x$($area.Height), $($colours.Count) sampled colours"
}

function Wait-Until([string]$what, [scriptblock]$test, [int]$seconds = $TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if ($script:app -and $script:app.HasExited) { throw "Markitai exited with code $($script:app.ExitCode) while waiting for $what" }
        $value = & $test
        if ($value) { return $value }
        Start-Sleep -Milliseconds 500
    }
    throw "timed out after $seconds s waiting for $what"
}

function Finished-Rows([int]$count) {
    $pending = @((S 'state.queued'), (S 'state.converting'))
    return Wait-Until "$count finished rows" {
        $rows = Rows (S 'a11y.conversions')
        $busy = @($rows | Where-Object { $row = $_; @($pending | Where-Object { $row.EndsWith(", $_") }).Count -gt 0 })
        if ($rows.Count -eq $count -and $busy.Count -eq 0) { return ,$rows }
        return $null
    }
}

# "Done in 1.2 s, middle dot, 1 warning": the row ends in the plural warnings phrase.
function Warned([string]$row) {
    $dot = " $([char]0x00B7) "
    $one = [regex]::Escape($dot + (S 'state.warnings.one'))
    $other = [regex]::Escape($dot + (S 'state.warnings.other')).Replace('\{count}', '[\d,.]+')
    return ($row -match "($one|$other)$")
}

function Check-Row($rows, $fixture) {
    $name = Split-Path $fixture.Path -Leaf
    $row = @($rows | Where-Object { $_.StartsWith("$name, ") })
    if ($row.Count -ne 1) { Check $false "one row for $name (rows: $($rows -join ' | '))"; return }
    $row = $row[0]
    if ($fixture.Done) {
        Check ($row.StartsWith("$name, $(Before 'state.done')") -and (Warned $row) -eq $fixture.Warnings) "row '$row' is Done$(if ($fixture.Warnings) { ' with warnings' })"
    } else {
        Check ($row.StartsWith("$name, $(Before 'row.failed')") -and $row.Contains((Before $fixture.Code))) "row '$row' failed with $($fixture.Code)"
    }
}

function Read-Json([string]$path) { return (Get-Content -Raw -Encoding UTF8 $path | ConvertFrom-Json) }

# --- The run -----------------------------------------------------------------

$script:app = $null
$started = Get-Date
try {
    Stage 'launch with fixtures' {
        $script:app = Start-Markitai ($fixtures | ForEach-Object { $_.Path })
        $script:window = Wait-Until 'the main window' { $script:app.Refresh(); if ($script:app.MainWindowHandle -ne [IntPtr]::Zero) { $script:app.MainWindowHandle } } 90
        $script:app.Refresh()
        Log "window pid $($script:app.Id), session $($script:app.SessionId), title '$($script:app.MainWindowTitle)'"
        Check ($script:app.MainWindowTitle -eq (S 'app.name')) "window title is '$(S 'app.name')'"
    }

    Stage 'conversions from the command line' {
        $rows = Finished-Rows $fixtures.Count
        foreach ($fixture in $fixtures) { Check-Row $rows $fixture }
    }

    Stage 'single-instance redirect' {
        $second = Start-Markitai @($redirected.Path)
        Check ($second.WaitForExit(30000)) "the second launch hands over and exits"
        if ($second.HasExited) { Check ($second.ExitCode -eq 0) "the second launch exits 0 (got $($second.ExitCode))" }
        $script:app.Refresh()
        Check (-not $script:app.HasExited) "the first instance keeps running"
        $rows = Finished-Rows ($fixtures.Count + 1)
        Check-Row $rows $redirected
        [IO.File]::WriteAllLines((Join-Path $Out 'convert-rows.txt'), [string[]]$rows, $utf8)
    }

    Stage 'output folders and history.json' {
        $folders = @(Get-ChildItem -Directory $output | ForEach-Object { $_.Name } | Sort-Object)
        Check (($folders -join '|') -eq (($expectedFolders | Sort-Object) -join '|')) "output folders: $($folders -join ', ')"
        $history = Wait-Until 'history.json with every finished conversion' {
            $file = Join-Path $data 'history.json'
            if (Test-Path $file) { $read = Read-Json $file; if (@($read.entries).Count -eq $expectedFolders.Count) { $read } }
        } 30
        foreach ($entry in @($history.entries)) {
            Check ((Test-Path $entry.markdown_path) -and $entry.folder.StartsWith($output)) "history entry '$($entry.title)' has its Markdown at $($entry.markdown_path)"
        }
    }

    Stage 'detail: preview and Markdown' {
        Choose ((Rows (S 'a11y.conversions')) | Where-Object { $_.StartsWith('default-table.html, ') } | Select-Object -First 1)
        $preview = Find (S 'a11y.preview')
        Wait-Until 'the preview on screen' { -not $preview.Current.IsOffscreen } 30 | Out-Null
        $page = Join-Path $data 'Preview\current.html'
        Wait-Until 'the preview page' { (Test-Path $page) -and ((Get-Content -Raw -Encoding UTF8 $page).Contains('Systems programming')) } 30 | Out-Null
        Check $true "'$(S 'a11y.preview')' is on screen and shows default-table.html"
        $document = $preview.FindFirst($Scope::Descendants, (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, $Types::Document)))
        if ($document) { Log "preview document '$($document.Current.Name)'" } else { Log 'preview document not exposed to UI Automation' }
        Shot 'convert'
        Choose (S 'detail.markdown')
        $text = (Pattern (Find (S 'a11y.markdown_text')) ([System.Windows.Automation.ValuePattern]::Pattern)).Current.Value
        Check ($text.Contains('Systems programming')) "'$(S 'a11y.markdown_text')' shows the Markdown ($($text.Length) characters)"
        Shot 'convert-markdown'
        Choose (S 'detail.preview')
    }

    Stage 'History' {
        Choose (S 'nav.history')
        $rows = Wait-Until 'History rows' { $r = Rows (S 'nav.history'); if ($r.Count -eq $expectedFolders.Count) { ,$r } } 20
        [IO.File]::WriteAllLines((Join-Path $Out 'history-rows.txt'), [string[]]$rows, $utf8)
        Check $true "History lists $($rows.Count) conversions"
        Shot 'history'
    }

    Stage 'Settings' {
        Choose (S 'nav.settings')
        $condition = [System.Windows.Automation.AndCondition]::new(
            (Condition (S 'settings.ocr')),
            (New-Object System.Windows.Automation.PropertyCondition($AE::IsTogglePatternAvailableProperty, $true)))
        $ocr = Wait-Until 'the OCR switch' { (Root).FindFirst($Scope::Descendants, $condition) } 20
        Shot 'settings'
        $toggle = Pattern $ocr ([System.Windows.Automation.TogglePattern]::Pattern)
        $toggle.Toggle()
        Log "toggled '$(S 'settings.ocr')' to $($toggle.Current.ToggleState)"
        Wait-Until 'conversion.ocr in settings.json' { (Read-Json (Join-Path $data 'settings.json')).'conversion.ocr' -eq $true } 10 | Out-Null
        Check $true 'the OCR switch persists to settings.json'
    }
}
finally {
    if ($script:window -ne [IntPtr]::Zero -and $script:app -and -not $script:app.HasExited) {
        try { Dump (Join-Path $Out 'uia-tree.txt') } catch { Log "UI Automation dump failed: $($_.Exception.Message)" }
        Stage 'quit' {
            [SmokeNative]::PostMessage($script:window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
            Check ($script:app.WaitForExit(30000)) 'WM_CLOSE quits the app'
            $settings = Read-Json (Join-Path $data 'settings.json')
            Check (($settings.'output.folder' -eq $output) -and ($settings.'conversion.ocr' -eq $true)) 'settings.json keeps the output folder and OCR after quitting'
        }
    }
    if ($script:app -and -not $script:app.HasExited) { $script:app.Kill(); Check $false 'Markitai had to be killed' }
    if (-not $script:app -or $script:failures.Count -gt 0) {
        $bounds = $screen
        $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try { $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); $bitmap.Save((Join-Path $Out 'screen.png')) } catch { Log "screen capture failed: $($_.Exception.Message)" }
        $graphics.Dispose(); $bitmap.Dispose()
        Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $started } -ErrorAction SilentlyContinue |
            Where-Object { $_.ProviderName -in '.NET Runtime', 'Application Error', 'Windows Error Reporting' } |
            Format-List TimeCreated, ProviderName, Message | Out-File -Encoding UTF8 (Join-Path $Out 'events.txt')
    }
    $copy = Join-Path $Out 'state'
    New-Item -ItemType Directory -Force $copy | Out-Null
    foreach ($file in 'settings.json', 'history.json', 'Preview\current.html') {
        if (Test-Path (Join-Path $data $file)) { Copy-Item (Join-Path $data $file) (Join-Path $copy (Split-Path $file -Leaf)) }
    }
    if (Test-Path $output) {
        Get-ChildItem -Recurse -File $output | ForEach-Object { "$($_.Length)`t$($_.FullName.Substring($output.Length + 1))" } |
            Out-File -Encoding UTF8 (Join-Path $Out 'output-files.txt')
    }
    $summary = @("Markitai smoke test, $(Get-Date -Format s)", "app: $exe", '') + $script:checks + @('', "$($script:failures.Count) failed")
    [IO.File]::WriteAllLines((Join-Path $Out 'summary.txt'), [string[]]$summary, $utf8)
}

if ($script:failures.Count -gt 0) {
    Log "$($script:failures.Count) check(s) failed"
    exit 1
}
Log 'all checks passed'
exit 0
