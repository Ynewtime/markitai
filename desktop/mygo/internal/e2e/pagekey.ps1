# Sends Ctrl+= to a process's window, as the keyboard does, for the e2e
# driver's page key check (desktop/mygo/DESIGN.md 16). It brings the window
# to the front first and sends nothing when another window stays in front.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File pagekey.ps1 <process ID> zoom-in
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][ValidateSet('zoom-in')][string]$Key
)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class PageKey {
    public delegate bool EnumProc(IntPtr window, IntPtr parameter);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc proc, IntPtr parameter);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern void keybd_event(byte key, byte scan, uint flags, UIntPtr extra);
}
'@
function Owner([IntPtr]$window) {
    [uint32]$owner = 0
    [PageKey]::GetWindowThreadProcessId($window, [ref]$owner) | Out-Null
    return $owner
}
$script:target = [IntPtr]::Zero
$visit = [PageKey+EnumProc] {
    param($window, $parameter)
    if ($script:target -eq [IntPtr]::Zero -and (Owner $window) -eq $ProcessId -and [PageKey]::IsWindowVisible($window)) { $script:target = $window }
    return $true
}
[PageKey]::EnumWindows($visit, [IntPtr]::Zero) | Out-Null
if ($script:target -eq [IntPtr]::Zero) { throw "process $ProcessId has no visible window" }
[PageKey]::SetForegroundWindow($script:target) | Out-Null
Start-Sleep -Milliseconds 300
$front = Owner ([PageKey]::GetForegroundWindow())
if ($front -ne $ProcessId) { throw "process $front is in front, not ${ProcessId}: no keys sent" }
$KeyUp = 2
# Virtual keys with their scan codes, which the web view reads as event.code.
[PageKey]::keybd_event(0x11, 0x1D, 0, [UIntPtr]::Zero)       # Ctrl
[PageKey]::keybd_event(0xBB, 0x0D, 0, [UIntPtr]::Zero)       # = (VK_OEM_PLUS, code Equal)
[PageKey]::keybd_event(0xBB, 0x0D, $KeyUp, [UIntPtr]::Zero)
[PageKey]::keybd_event(0x11, 0x1D, $KeyUp, [UIntPtr]::Zero)
"sent Ctrl+= to process $ProcessId"
