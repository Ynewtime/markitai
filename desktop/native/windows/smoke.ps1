<#
.SYNOPSIS
Runs the built Markitai Native.exe on this desktop and proves the workbench end to end.

.DESCRIPTION
Starts the app with an isolated user state (HOME, USERPROFILE, LOCALAPPDATA, APPDATA,
MARKITAI_HOME, TEMP and TMP all inside -State) and repository fixtures as arguments,
hands one more file to it through a second launch (the single-instance redirect),
waits for every conversion to finish, then reads the window through UI Automation
patterns only: rows and their states, the detail's preview and Markdown text,
History, and a Settings toggle that must reach settings.json. It captures Convert,
History, Settings and the quit dialog, keeps one item Converting while the window is
closed (Cancel, then Quit), and exits 1 if any piece of evidence is missing. Evidence
goes to -Out; summary.txt lists every check.

Runs in Windows PowerShell 5.1 or later, in an interactive session:

    powershell -File desktop\native\windows\smoke.ps1 -State $env:RUNNER_TEMP\markitai-smoke -Out $env:RUNNER_TEMP\markitai-evidence

-Capture screen (the default, for a disposable runner with the display on) brings the
window forward and copies it from the screen. -Capture window, for a desktop someone
uses, leaves focus alone and asks the window to draw itself (PrintWindow with
PW_RENDERFULLCONTENT).
#>
param(
    [Parameter(Mandatory = $true)][string]$State,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$App = (Join-Path $PSScriptRoot 'Markitai.App\bin\x64\Release\net10.0-windows10.0.26100.0\win-x64'),
    [int]$TimeoutSec = 300,
    [ValidateSet('screen', 'window')][string]$Capture = 'screen'
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
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr window, IntPtr after, int x, int y, int cx, int cy, uint flags);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
[SmokeNative]::SetProcessDPIAware() | Out-Null

$AE = [System.Windows.Automation.AutomationElement]
$Scope = [System.Windows.Automation.TreeScope]
$Types = [System.Windows.Automation.ControlType]

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = Join-Path $App 'Markitai Native.exe'
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

$strings = Get-Content -Raw -Encoding UTF8 (Join-Path $repo 'desktop\strings.json') | ConvertFrom-Json
function S([string]$key) {
    $entry = $strings.$key
    if (-not $entry) { throw "desktop/strings.json has no $key" }
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
$data = Join-Path $isolated.LOCALAPPDATA 'Markitai Native'
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

# Selects the item named $name: through SelectionItemPattern on it or on an ancestor
# (a list's data item can wrap the container that carries the name), else InvokePattern.
function Choose([string]$name) {
    $null = Find $name
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $seen = @()
    foreach ($candidate in (Root).FindAll($Scope::Descendants, (Condition $name))) {
        $element = $candidate
        for ($up = 0; $element -and $up -le 2; $up++) {
            $found = $null
            if ($element.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$found)) {
                $found.Select(); Log "selected '$name' ($($element.Current.ControlType.ProgrammaticName), $up up)"; return
            }
            $element = $walker.GetParent($element)
        }
        $found = $null
        if ($candidate.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$found)) {
            $found.Invoke(); Log "invoked '$name' ($($candidate.Current.ControlType.ProgrammaticName))"; return
        }
        $seen += "$($candidate.Current.ControlType.ProgrammaticName) [$(Patterns $candidate)]"
    }
    throw "nothing named '$name' can be selected or invoked: $($seen -join '; ')"
}

function Patterns($element) {
    return (@($element.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName -replace 'PatternIdentifiers.Pattern', '' }) -join ',')
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
        $patterns = Patterns $element
        if ($patterns) { $flags += $patterns }
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
    $rect = New-Object SmokeNative+RECT
    if ($Capture -eq 'window') {
        Start-Sleep -Milliseconds 800
        [SmokeNative]::GetWindowRect($script:window, [ref]$rect) | Out-Null
        $area = [System.Drawing.Rectangle]::FromLTRB(0, 0, $rect.Right - $rect.Left, $rect.Bottom - $rect.Top)
        $bitmap = New-Object System.Drawing.Bitmap $area.Width, $area.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $dc = $graphics.GetHdc()
        $drawn = [SmokeNative]::PrintWindow($script:window, $dc, 2)
        $graphics.ReleaseHdc($dc)
        $graphics.Dispose()
        if (-not $drawn) { $bitmap.Dispose(); throw "PrintWindow could not draw the window for $name" }
    } else {
        try { (Root).SetFocus() } catch { }
        [SmokeNative]::SetForegroundWindow($script:window) | Out-Null
        Start-Sleep -Milliseconds 800
        [SmokeNative]::GetWindowRect($script:window, [ref]$rect) | Out-Null
        $area = [System.Drawing.Rectangle]::Intersect($screen, [System.Drawing.Rectangle]::FromLTRB($rect.Left, $rect.Top, $rect.Right, $rect.Bottom))
        if ($area.Width -le 0 -or $area.Height -le 0) { throw "the window is off screen ($($rect.Left),$($rect.Top),$($rect.Right),$($rect.Bottom))" }
        $bitmap = New-Object System.Drawing.Bitmap $area.Width, $area.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.CopyFromScreen($area.Location, [System.Drawing.Point]::Empty, $area.Size)
        $graphics.Dispose()
    }
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
        if ($script:markitai -and $script:markitai.HasExited) { throw "Markitai exited with code $($script:markitai.ExitCode) while waiting for $what" }
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

# The Markitai process. Not $app: PowerShell names are case-insensitive, and the
# [string]$App parameter would turn the process into its name.
$script:markitai = $null
$started = Get-Date
try {
    Stage 'launch with fixtures' {
        $script:markitai = Start-Markitai ($fixtures | ForEach-Object { $_.Path })
        $script:window = Wait-Until 'the main window' { $script:markitai.Refresh(); if ($script:markitai.MainWindowHandle -ne [IntPtr]::Zero) { $script:markitai.MainWindowHandle } } 90
        $script:markitai.Refresh()
        Log "window pid $($script:markitai.Id), session $($script:markitai.SessionId), title '$($script:markitai.MainWindowTitle)'"
        Check ($script:markitai.MainWindowTitle -eq (S 'app.name.native')) "window title is '$(S 'app.name.native')'"
        # Windows places a new window down and to the right; the captures need all of it.
        # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
        [SmokeNative]::SetWindowPos($script:window, [IntPtr]::Zero, 0, 0, 0, 0, 0x0001 -bor 0x0004 -bor 0x0010) | Out-Null
        $rect = New-Object SmokeNative+RECT
        [SmokeNative]::GetWindowRect($script:window, [ref]$rect) | Out-Null
        $size = "$($rect.Right - $rect.Left)x$($rect.Bottom - $rect.Top) at $($rect.Left),$($rect.Top)"
        if ($Capture -eq 'screen') { Check ($rect.Right -le $screen.Right -and $rect.Bottom -le $screen.Bottom) "the whole window is on screen ($size)" }
        else { Log "window $size" }
    }

    Stage 'conversions from the command line' {
        $rows = Finished-Rows $fixtures.Count
        foreach ($fixture in $fixtures) { Check-Row $rows $fixture }
    }

    Stage 'single-instance redirect' {
        $second = Start-Markitai @($redirected.Path)
        Check ($second.WaitForExit(30000)) "the second launch hands over and exits"
        if ($second.HasExited) { Check ($second.ExitCode -eq 0) "the second launch exits 0 (got $($second.ExitCode))" }
        $script:markitai.Refresh()
        Check (-not $script:markitai.HasExited) "the first instance keeps running"
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
        Check (-not (Get-Content -Raw -Encoding UTF8 $page).Contains('zoom:')) 'the preview has no zoom at the default view.zoom'
        Dump (Join-Path $Out 'uia-detail.txt')
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
        Dump (Join-Path $Out 'uia-tree.txt')
    }

    # DESIGN.md 7.3 and 16, Quit: a page served by a listener that never answers keeps
    # one item Converting while the window is closed twice.
    Stage 'quit while converting' {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try {
            $url = "http://127.0.0.1:$($listener.LocalEndpoint.Port)/slow-page"
            $converting = "$($url.Substring(7)), $(S 'state.converting')"
            $before = @(Get-ChildItem -Directory $output | ForEach-Object { $_.Name })
            Choose (S 'nav.convert')
            (Pattern (Find (S 'convert.url.label') $Types::Edit) ([System.Windows.Automation.ValuePattern]::Pattern)).SetValue($url)
            (Pattern (Find (S 'convert.url.add') $Types::Button) ([System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
            Wait-Until "the row '$converting'" { (Rows (S 'a11y.conversions')) -contains $converting } 20 | Out-Null
            $claimed = Wait-Until 'its claimed folder' { Get-ChildItem -Directory $output | ForEach-Object { $_.Name } | Where-Object { $before -notcontains $_ } | Select-Object -First 1 } 20
            Check $true "'$converting' is Converting in its folder '$claimed'"

            [SmokeNative]::PostMessage($script:window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
            $null = Find (S 'quit.title') $null 15
            $null = Find (S 'quit.body.one') $null 5
            Check $true "WM_CLOSE with a conversion running shows '$(S 'quit.title')'"
            Shot 'quit-dialog'
            (Pattern (Find (S 'common.cancel') $Types::Button) ([System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
            Wait-Until 'the dialog to close' { -not (Root).FindFirst($Scope::Descendants, (Condition (S 'quit.title'))) } 10 | Out-Null
            Start-Sleep -Seconds 1
            Check ((-not $script:markitai.HasExited) -and ((Rows (S 'a11y.conversions')) -contains $converting)) 'Cancel keeps the app and the conversion running'

            $closer = $null
            if ((Root).TryGetCurrentPattern([System.Windows.Automation.WindowPattern]::Pattern, [ref]$closer)) {
                $closer.Close(); Log 'closed through WindowPattern'
            } else {
                [SmokeNative]::PostMessage($script:window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null; Log 'closed through WM_CLOSE'
            }
            (Pattern (Find (S 'quit.confirm') $Types::Button 15) ([System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
            Check ($script:markitai.WaitForExit(30000)) 'Quit in the dialog quits the app'
            Check (-not (Test-Path -LiteralPath (Join-Path $output $claimed))) "Quit deletes the folder '$claimed' of the stopped conversion"
            $history = Read-Json (Join-Path $data 'history.json')
            Check (@($history.entries | Where-Object { $_.source -eq $url }).Count -eq 0 -and @($history.entries).Count -eq $expectedFolders.Count) 'History keeps the finished conversions and not the stopped one'
            $settings = Read-Json (Join-Path $data 'settings.json')
            Check (($settings.'output.folder' -eq $output) -and ($settings.'conversion.ocr' -eq $true)) 'settings.json keeps the output folder and OCR after quitting'
        }
        finally { $listener.Stop() }
    }
}
finally {
    if ($script:window -ne [IntPtr]::Zero -and $script:markitai -and -not $script:markitai.HasExited) {
        try { Dump (Join-Path $Out 'uia-tree.txt') } catch { Log "UI Automation dump failed: $($_.Exception.Message)" }
        [SmokeNative]::PostMessage($script:window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        if (-not $script:markitai.WaitForExit(15000)) { Log 'WM_CLOSE did not quit the app' }
    }
    if ($script:markitai -and -not $script:markitai.HasExited) {
        try { $script:markitai.Kill() } catch { Log "kill failed: $($_.Exception.Message)" }
        Check $false 'Markitai quit on its own'
    }
    if ($Capture -eq 'screen' -and (-not $script:markitai -or $script:failures.Count -gt 0)) {
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
