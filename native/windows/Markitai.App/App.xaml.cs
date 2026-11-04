using System.Globalization;
using Markitai.Core;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Markup;
using Microsoft.Windows.AppLifecycle;
using Windows.System.UserProfile;

namespace MarkitaiApp;

public partial class App : Application
{
    readonly Startup startup;
    Window? window;

    public App(Startup startup)
    {
        this.startup = startup;
        Strings = Strings.For(GlobalizationPreferences.Languages.FirstOrDefault(), CultureInfo.CurrentCulture);
        // DESIGN.md 7.5: WinUI's own words (the navigation toggle, dialogs, menus) in the app's language.
        if (Strings.Language == "zh-Hans")
            Microsoft.Windows.Globalization.ApplicationLanguages.PrimaryLanguageOverride = "zh-Hans";
        UnhandledException += (_, e) => OnUnhandled(e.Exception);
        InitializeComponent();
    }

    /// <summary>Writes the error to error.log in the data folder and deletes the folders claimed for running conversions before the process ends.</summary>
    void OnUnhandled(Exception error)
    {
        Log(startup.Paths, error);
        try
        {
            Bench?.Queue.DiscardConverting();
        }
        catch (Exception e)
        {
            Log(startup.Paths, e);
        }
    }

    public static void Log(AppPaths paths, Exception error)
    {
        try
        {
            Directory.CreateDirectory(paths.DataDir);
            File.AppendAllText(paths.ErrorLog, $"{DateTimeOffset.Now:O} {error}{Environment.NewLine}");
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>The copy in the language chosen at launch (native/DESIGN.md 7.5).</summary>
    public static Strings Strings { get; private set; } = null!;

    public static Workbench Bench { get; private set; } = null!;

    protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
    {
        if (startup.Fatal)
        {
            window = new FatalWindow();
            window.Activate();
            return;
        }
        var dispatcher = DispatcherQueue.GetForCurrentThread();
        Bench = new Workbench(startup, Strings, dispatcher);
        var main = new MainWindow();
        window = main;
        main.Activate();
        AppInstance.GetCurrent().Activated += (_, e) =>
        {
            try
            {
                var files = Program.FileArguments(e);
                dispatcher.TryEnqueue(() => main.OnRedirected(files));
            }
            catch (Exception error)
            {
                Log(startup.Paths, error);
            }
        };
        Bench.AddFiles(startup.Files, startup.LaunchDirectory);
    }
}

/// <summary><c>{m:S Key=nav.convert}</c>: a string from native/strings.json.</summary>
[MarkupExtensionReturnType(ReturnType = typeof(string))]
public sealed partial class S : MarkupExtension
{
    public string Key { get; set; } = "";

    protected override object ProvideValue() => App.Strings[Key];
}
