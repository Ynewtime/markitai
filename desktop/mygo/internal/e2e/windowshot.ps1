# What the screen shows of a process's largest visible window, mygo's surface
# and the web view the system composites into it, as a PNG. The e2e driver runs
# it when MARKITAI_MYGO_E2E_WINDOW_SHOT names it (desktop/mygo/DESIGN.md 16).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File windowshot.ps1 <process ID> <PNG path>
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$Path
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class WindowShot {
    public delegate bool EnumProc(IntPtr window, IntPtr parameter);
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr context);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc proc, IntPtr parameter);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr window);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr window, int attribute, out Rect rect, int size);
}
'@
# Per-monitor aware (v2): the frame and the screen copy are both in physical pixels.
[WindowShot]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
$DwmExtendedFrameBounds = 9
$script:best = $null
$script:area = 0
$visit = [WindowShot+EnumProc] {
    param($window, $parameter)
    [uint32]$owner = 0
    [WindowShot]::GetWindowThreadProcessId($window, [ref]$owner) | Out-Null
    if ($owner -eq $ProcessId -and [WindowShot]::IsWindowVisible($window)) {
        $frame = New-Object WindowShot+Rect
        if ([WindowShot]::DwmGetWindowAttribute($window, $DwmExtendedFrameBounds, [ref]$frame, 16) -eq 0) {
            $size = ($frame.Right - $frame.Left) * ($frame.Bottom - $frame.Top)
            if ($size -gt $script:area) { $script:area = $size; $script:best = $frame }
        }
    }
    return $true
}
[WindowShot]::EnumWindows($visit, [IntPtr]::Zero) | Out-Null
if ($null -eq $script:best) { throw "process $ProcessId has no visible window" }
$frame = $script:best
$width = $frame.Right - $frame.Left
$height = $frame.Bottom - $frame.Top
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.CopyFromScreen($frame.Left, $frame.Top, 0, 0, (New-Object System.Drawing.Size $width, $height))
    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
