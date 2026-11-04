using System.Runtime.InteropServices;
using Markitai.Core;
using Microsoft.UI.Dispatching;
using Microsoft.Windows.AppLifecycle;

namespace MarkitaiApp;

/// <summary>What Main learned before the UI started.</summary>
public sealed record Startup(IReadOnlyList<string> Files, string LaunchDirectory, AppPaths Paths, string? CoreVersion)
{
    public bool Fatal => CoreVersion is null;
}

public sealed record AppPaths(string DataDir)
{
    /// <summary>%LOCALAPPDATA%\Markitai, read from the variable so a test run can move it.</summary>
    public static AppPaths FromEnvironment()
    {
        var local = Environment.GetEnvironmentVariable("LOCALAPPDATA");
        if (string.IsNullOrEmpty(local))
            local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return new(Path.Combine(local, "Markitai"));
    }

    public string History => Path.Combine(DataDir, "history.json");
    public string Settings => Path.Combine(DataDir, "settings.json");
    public string WebView2 => Path.Combine(DataDir, "WebView2");
    public string PreviewFile => Path.Combine(DataDir, "Preview", "current.html");
    public string ErrorLog => Path.Combine(DataDir, "error.log");
}

public static partial class Program
{
    public const string InstanceKey = "dev.markitai.Markitai";

    [STAThread]
    static int Main(string[] args)
    {
        WinRT.ComWrappersSupport.InitializeComWrappers();
        var instance = AppInstance.FindOrRegisterForKey(InstanceKey);
        if (!instance.IsCurrent)
        {
            if (args.Any(a => !string.IsNullOrWhiteSpace(a) && !Path.IsPathFullyQualified(a)))
                Relaunch(args);
            else
                Redirect(instance, AppInstance.GetCurrent().GetActivatedEventArgs());
            return 0;
        }
        var startup = Prepare(args);
        Microsoft.UI.Xaml.Application.Start(callback =>
        {
            SynchronizationContext.SetSynchronizationContext(new DispatcherQueueSynchronizationContext(DispatcherQueue.GetForCurrentThread()));
            _ = new App(startup);
        });
        return 0;
    }

    /// <summary>
    /// Checks the core's ABI, then lets it create the data folder and make it the working
    /// directory, before any window exists (native/DESIGN.md 3.3).
    /// </summary>
    static Startup Prepare(string[] args)
    {
        var launchDirectory = Environment.CurrentDirectory;
        var paths = AppPaths.FromEnvironment();
        string? version = null;
        try
        {
            if (NativeCore.AbiVersion() == NativeCore.ExpectedAbiVersion)
            {
                NativeCore.PrepareEnvironment(CoreJson.PrepareRequest(paths.DataDir));
                version = NativeCore.Version();
            }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
        }
        return new(args, launchDirectory, paths, version);
    }

    /// <summary>
    /// The redirect carries the command line but not the working directory, so the running
    /// instance cannot resolve a relative path: a launch with one starts again with full paths.
    /// </summary>
    static void Relaunch(string[] args)
    {
        var info = new System.Diagnostics.ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
        foreach (var arg in args)
            info.ArgumentList.Add(string.IsNullOrWhiteSpace(arg) ? arg : Path.GetFullPath(arg));
        using var relaunched = System.Diagnostics.Process.Start(info);
        if (relaunched is not null)
            AllowSetForegroundWindow((uint)relaunched.Id);
    }

    /// <summary>Hands this launch, file arguments included, to the running instance and lets it come forward.</summary>
    static void Redirect(AppInstance target, AppActivationArguments arguments)
    {
        AllowSetForegroundWindow(target.ProcessId);
        var done = CreateEventW(0, true, false, null);
        Task.Run(() =>
        {
            target.RedirectActivationToAsync(arguments).AsTask().Wait();
            SetEvent(done);
        });
        CoWaitForMultipleObjects(0, 0xFFFFFFFF, 1, [done], out _);
        CloseHandle(done);
    }

    /// <summary>The file arguments of a redirected launch: its command line without the program.</summary>
    public static IReadOnlyList<string> FileArguments(AppActivationArguments arguments)
    {
        if (arguments.Kind != ExtendedActivationKind.Launch || arguments.Data is not Windows.ApplicationModel.Activation.ILaunchActivatedEventArgs launch)
            return [];
        var parts = SplitCommandLine(launch.Arguments ?? "");
        if (parts.Count > 0 && string.Equals(Path.GetFileName(parts[0]), Path.GetFileName(Environment.ProcessPath), StringComparison.OrdinalIgnoreCase))
            parts.RemoveAt(0);
        return parts;
    }

    static List<string> SplitCommandLine(string commandLine)
    {
        if (string.IsNullOrWhiteSpace(commandLine))
            return [];
        var argv = CommandLineToArgvW(commandLine, out var count);
        if (argv == 0)
            return [];
        try
        {
            var result = new List<string>(count);
            for (var i = 0; i < count; i++)
                result.Add(Marshal.PtrToStringUni(Marshal.ReadIntPtr(argv, i * IntPtr.Size)) ?? "");
            return result;
        }
        finally
        {
            LocalFree(argv);
        }
    }

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool AllowSetForegroundWindow(uint processId);

    [LibraryImport("kernel32.dll", StringMarshalling = StringMarshalling.Utf16)]
    private static partial nint CreateEventW(nint attributes, [MarshalAs(UnmanagedType.Bool)] bool manualReset, [MarshalAs(UnmanagedType.Bool)] bool initialState, string? name);

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool SetEvent(nint handle);

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CloseHandle(nint handle);

    [LibraryImport("ole32.dll")]
    private static partial int CoWaitForMultipleObjects(uint flags, uint timeout, uint count, [In] nint[] handles, out uint index);

    [LibraryImport("shell32.dll", StringMarshalling = StringMarshalling.Utf16)]
    private static partial nint CommandLineToArgvW(string commandLine, out int count);

    [LibraryImport("kernel32.dll")]
    private static partial nint LocalFree(nint memory);
}
