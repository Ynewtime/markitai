using System.Reflection;
using Markitai.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Imaging;

namespace MarkitaiApp.Views;

/// <summary>The Settings page (desktop/DESIGN.md 6.6). Every change is saved at once.</summary>
public sealed partial class SettingsPage : UserControl
{
    readonly Workbench bench = App.Bench;
    readonly Strings s = App.Strings;
    string? configPath;
    bool configExists;
    bool loading;

    public SettingsPage()
    {
        InitializeComponent();
        AboutIcon.Source = new BitmapImage(new Uri(Path.Combine(AppContext.BaseDirectory, "Assets", "icon.png"))) { DecodePixelWidth = 32 };
        AppVersion.Text = AppVersionText();
        CoreVersion.Text = bench.CoreVersion;
        bench.SettingsChanged += ShowOutputFolder;
        Refresh();
    }

    static string AppVersionText()
    {
        var version = typeof(App).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "";
        var plus = version.IndexOf('+');
        return plus < 0 ? version : version[..plus];
    }

    /// <summary>Reloads every control from the settings and the configuration file's state; called when the page opens.</summary>
    public void Refresh()
    {
        loading = true;
        var settings = bench.Settings;
        Llm.IsOn = settings.Llm;
        Ocr.IsOn = settings.Ocr;
        Screenshot.IsOn = settings.Screenshot;
        Desc.IsOn = settings.Desc;
        Alt.IsOn = settings.Alt;
        UseUser.IsChecked = settings.Config == ConfigSource.User;
        UseBuiltin.IsChecked = settings.Config == ConfigSource.Builtin;
        loading = false;
        ShowDependents();
        ShowOutputFolder();
        OutputError.IsOpen = false;
        ConfigError.IsOpen = false;
        ShowConfigFile();
    }

    void ShowDependents()
    {
        DescCard.IsEnabled = Llm.IsOn;
        AltCard.IsEnabled = Llm.IsOn;
    }

    void ShowOutputFolder() => OutputPath.Text = bench.OutputFolder;

    void ShowConfigFile()
    {
        var answer = CoreJson.Parse(NativeCore.ConfigFile());
        configPath = answer.Ok ? CoreJson.Text(answer.Result, "path") : null;
        configExists = answer.Ok && answer.Result.TryGetProperty("exists", out var exists) && exists.ValueKind == System.Text.Json.JsonValueKind.True;
        UserNote.Text = configExists && configPath is not null ? configPath : s["settings.config.user.missing"];
        ConfigFileCard.Header = s[configExists ? "settings.config.open" : "settings.config.create"];
        ConfigFileCard.Description = configPath ?? "";
        ConfigFileCard.IsEnabled = configPath is not null;
    }

    void Toggle_Toggled(object sender, RoutedEventArgs e)
    {
        if (loading)
            return;
        bench.UpdateSettings(settings => settings with
        {
            Llm = Llm.IsOn,
            Ocr = Ocr.IsOn,
            Screenshot = Screenshot.IsOn,
            Desc = Desc.IsOn,
            Alt = Alt.IsOn,
        });
        ShowDependents();
    }

    void Config_Checked(object sender, RoutedEventArgs e)
    {
        if (loading)
            return;
        var source = UseBuiltin.IsChecked == true ? ConfigSource.Builtin : ConfigSource.User;
        bench.UpdateSettings(settings => settings with { Config = source });
    }

    async void ChangeOutput_Click(object sender, RoutedEventArgs e)
    {
        var picker = new Microsoft.Windows.Storage.Pickers.FolderPicker(MainWindow.Current.AppWindow.Id);
        var picked = await picker.PickSingleFolderAsync();
        if (picked is null)
            return;
        var folder = picked.Path;
        if (!await Task.Run(() => UserFiles.CanWrite(folder)))
        {
            OutputError.IsOpen = true;
            return;
        }
        OutputError.IsOpen = false;
        bench.UpdateSettings(settings => settings with { OutputFolder = folder });
    }

    void ShowOutput_Click(object sender, RoutedEventArgs e) => MainWindow.Current.OpenFolder(bench.OutputFolder);

    void ConfigFileCard_Click(object sender, RoutedEventArgs e)
    {
        if (configPath is null)
            return;
        if (!configExists)
        {
            try
            {
                UserFiles.CreateConfiguration(configPath);
            }
            catch (IOException) when (File.Exists(configPath))
            {
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                ShowConfigError();
                return;
            }
            ShowConfigFile();
        }
        ConfigError.IsOpen = false;
        if (!MainWindow.Current.TryOpenFile(configPath))
            ShowConfigError();
    }

    void ShowConfigError()
    {
        ConfigError.Message = s.Format("error.open_failed", ("name", Path.GetFileName(configPath ?? "")));
        ConfigError.IsOpen = true;
    }
}
