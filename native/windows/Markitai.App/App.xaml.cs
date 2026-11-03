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
        InitializeComponent();
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
            var files = Program.FileArguments(e);
            dispatcher.TryEnqueue(() => main.OnRedirected(files));
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
