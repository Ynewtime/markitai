using System.ComponentModel;
using System.Globalization;
using Markitai.Core;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.Web.WebView2.Core;

namespace MarkitaiApp.Views;

/// <summary>
/// The detail of the selected Convert item or History entry (native/DESIGN.md 6.4), with
/// the one preview web view the window has (section 8).
/// </summary>
public sealed partial class DetailView : UserControl
{
    const double MarkdownFontSize = 13;

    readonly Workbench bench = App.Bench;
    readonly Strings s = App.Strings;
    readonly DispatcherQueueTimer elapsedTimer;
    readonly DispatcherQueueTimer copiedTimer;
    readonly Dictionary<string, bool> expanded = [];
    object? current;
    ConversionItem? watched;
    Finished? finished;
    string? markdown;
    string? baseDir;
    int version;
    string? previewShown;
    string? textShown;
    Task<bool>? webView;
    bool rendering;

    public DetailView()
    {
        InitializeComponent();
        elapsedTimer = DispatcherQueue.CreateTimer();
        elapsedTimer.Interval = TimeSpan.FromSeconds(1);
        elapsedTimer.Tick += (_, _) => ShowElapsed();
        copiedTimer = DispatcherQueue.CreateTimer();
        copiedTimer.Interval = TimeSpan.FromSeconds(1.5);
        copiedTimer.IsRepeating = false;
        copiedTimer.Tick += (_, _) => ShowCopied(false);
        bench.SettingsChanged += OnSettingsChanged;
        MarkdownText.FontSize = MarkdownFontSize * bench.Settings.Zoom;
        Render();
    }

    bool PreviewChosen => ViewSwitch.SelectedIndex != 1;

    /// <summary>Shows a ConversionItem, a HistoryEntry or nothing. The same selection again changes nothing.</summary>
    public void Show(object? selection)
    {
        if (ReferenceEquals(selection, current))
            return;
        if (watched is not null)
            watched.PropertyChanged -= Item_PropertyChanged;
        current = selection;
        watched = selection as ConversionItem;
        if (watched is not null)
            watched.PropertyChanged += Item_PropertyChanged;
        Render();
    }

    void Item_PropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ConversionItem.State) && ReferenceEquals(sender, current))
            Render();
    }

    enum Body
    {
        None,
        Content,
        Message,
        Error,
    }

    void Render()
    {
        version++;
        elapsedTimer.Stop();
        finished = null;
        markdown = null;
        baseDir = null;
        previewShown = null;
        switch (current)
        {
            case ConversionItem item:
                RenderItem(item);
                break;
            case HistoryEntry entry:
                RenderEntry(entry);
                break;
            default:
                Layout(bar: false, header: false, Body.Message, footer: null);
                ShowMessage(null, s["detail.empty"], button: false);
                break;
        }
    }

    void RenderItem(ConversionItem item)
    {
        ShowHeader(item.Title, item.Source.Kind, item.Source.Value);
        switch (item.State)
        {
            case Queued:
                ShowStatus(Ui.Queued, null, s["state.queued"]);
                Layout(bar: false, header: true, Body.Message, footer: null);
                ShowMessage(null, s["detail.queued"], button: false);
                break;
            case Converting:
                ShowStatus(null, null, "");
                ShowElapsed();
                elapsedTimer.Start();
                Layout(bar: false, header: true, Body.Message, footer: null);
                ShowMessage(null, s["detail.converting.note"], button: false);
                break;
            case Done done:
                ShowStatus(Ui.DoneGlyph, "SystemFillColorSuccessBrush", s.Format("state.done", ("duration", bench.Text.Duration(done.Result.Duration))));
                ShowWarnings(item.Id.ToString(), done.Result.Warnings);
                Layout(bar: true, header: true, Body.Content, footer: bench.Text.Footer(done.Result), warnings: done.Result.Warnings.Count > 0);
                finished = Finished.Of(item, done);
                ShowContent(done.Result.Markdown, Path.GetDirectoryName(done.Result.MarkdownPath));
                break;
            case Failed failed:
                ShowStatus(Ui.FailedGlyph, "SystemFillColorCriticalBrush", s["state.failed"]);
                Layout(bar: false, header: true, Body.Error, footer: null);
                ShowError(failed);
                break;
        }
    }

    void RenderEntry(HistoryEntry entry)
    {
        ShowHeader(entry.Title, entry.Kind, entry.Source);
        ShowStatus(Ui.DoneGlyph, "SystemFillColorSuccessBrush", bench.Text.HistoryStatus(entry, DateTimeOffset.Now));
        Layout(bar: false, header: true, Body.None, footer: null);
        _ = LoadEntryAsync(entry, version);
    }

    /// <summary>Reads the entry's Markdown off the main thread; a missing file shows only Remove from History.</summary>
    async Task LoadEntryAsync(HistoryEntry entry, int expected)
    {
        string? text;
        try
        {
            text = await Task.Run(() => File.ReadAllText(entry.MarkdownPath));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            text = null;
        }
        if (expected != version)
            return;
        if (text is null)
        {
            Layout(bar: false, header: true, Body.Message, footer: null);
            ShowMessage(s["detail.missing.title"], s.Format("detail.missing.body", ("path", entry.MarkdownPath)), button: true);
            return;
        }
        ShowWarnings(entry.Id, entry.Warnings);
        Layout(bar: true, header: true, Body.Content, footer: bench.Text.Footer(entry.Duration, entry.Usage), warnings: entry.Warnings.Count > 0);
        finished = Finished.Of(entry) with { Markdown = text };
        ShowContent(text, Path.GetDirectoryName(entry.MarkdownPath));
    }

    void Layout(bool bar, bool header, Body body, string? footer, bool warnings = false)
    {
        Bar.Visibility = Ui.Show(bar);
        Header.Visibility = Ui.Show(header);
        WarningsBox.Visibility = Ui.Show(warnings);
        Message.Visibility = Ui.Show(body == Body.Message);
        ErrorBlock.Visibility = Ui.Show(body == Body.Error);
        if (body != Body.Content)
        {
            Preview.Visibility = Visibility.Collapsed;
            PreviewMissing.Visibility = Visibility.Collapsed;
            MarkdownText.Visibility = Visibility.Collapsed;
            MarkdownText.Text = "";
            textShown = null;
        }
        Footer.Text = footer ?? "";
        Footer.Visibility = Ui.Show(footer is not null);
        ShowCopied(false);
    }

    void ShowHeader(string title, SourceKind kind, string source)
    {
        TitleText.Text = title;
        SourceGlyph.Glyph = Ui.KindGlyphOf(kind);
        SourceText.Text = source;
        ToolTipService.SetToolTip(SourceText, source);
    }

    void ShowStatus(string? glyph, string? brush, string text)
    {
        var converting = glyph is null;
        StatusRing.Visibility = Ui.Show(converting);
        StatusRing.IsActive = converting;
        StatusGlyph.Visibility = Ui.Show(!converting);
        if (glyph is not null)
        {
            StatusGlyph.Glyph = glyph;
            StatusGlyph.Foreground = (Brush)Application.Current.Resources[brush ?? "TextFillColorSecondaryBrush"];
        }
        StatusText.Text = text;
    }

    void ShowElapsed()
    {
        if (current is ConversionItem { State: Converting converting })
            StatusText.Text = s.Format("detail.converting", ("elapsed", Present.Elapsed(DateTimeOffset.Now - converting.StartedAt)));
        else
            elapsedTimer.Stop();
    }

    void ShowMessage(string? title, string body, bool button)
    {
        MessageTitle.Text = title ?? "";
        MessageTitle.Visibility = Ui.Show(title is not null);
        MessageBody.Text = body;
        MessageButton.Visibility = Ui.Show(button);
    }

    void ShowWarnings(string key, IReadOnlyList<string> warnings)
    {
        WarningsLabel.Text = bench.Text.Warnings(warnings.Count);
        WarningLines.Children.Clear();
        foreach (var warning in warnings)
            WarningLines.Children.Add(new TextBlock { Text = warning, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true });
        rendering = true;
        WarningsBox.IsExpanded = expanded.GetValueOrDefault(key);
        rendering = false;
    }

    string? ExpandedKey => current switch
    {
        ConversionItem item => item.Id.ToString(),
        HistoryEntry entry => entry.Id,
        _ => null,
    };

    void WarningsBox_Expanding(Expander sender, ExpanderExpandingEventArgs args)
    {
        if (!rendering && ExpandedKey is { } key)
            expanded[key] = true;
    }

    void WarningsBox_Collapsed(Expander sender, ExpanderCollapsedEventArgs args)
    {
        if (!rendering && ExpandedKey is { } key)
            expanded[key] = false;
    }

    void ShowError(Failed failed)
    {
        ErrorSentence.Text = bench.Text.ErrorSentence(failed, bench.OutputFolder);
        ErrorDetails.Text = failed.Message.Length > 0 ? failed.Message : failed.Code;
        var action = Present.ActionFor(failed.Code);
        SecondButton.Visibility = Ui.Show(action != ErrorAction.None);
        SecondButton.Content = action switch
        {
            ErrorAction.Remove => s["convert.remove"],
            ErrorAction.OpenSettings => s["action.open_settings"],
            ErrorAction.OpenConfiguration => s["settings.config.open"],
            _ => "",
        };
        SecondButton.Tag = action;
    }

    void ShowContent(string text, string? folder)
    {
        markdown = text;
        baseDir = folder;
        ShowView();
    }

    /// <summary>Shows Preview or Markdown for the current content, rendering the preview when it is out of date.</summary>
    void ShowView()
    {
        if (markdown is null)
            return;
        var preview = PreviewChosen;
        MarkdownText.Visibility = Ui.Show(!preview);
        if (!preview)
        {
            if (!ReferenceEquals(textShown, markdown))
            {
                MarkdownText.Text = markdown;
                textShown = markdown;
            }
            Preview.Visibility = Visibility.Collapsed;
            PreviewMissing.Visibility = Visibility.Collapsed;
            return;
        }
        _ = ShowPreviewAsync(version);
    }

    async Task ShowPreviewAsync(int expected)
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var zoom = bench.Settings.Zoom;
        var key = $"{expected}|{dark}|{zoom.ToString(CultureInfo.InvariantCulture)}";
        if (previewShown == key)
        {
            Preview.Visibility = Visibility.Visible;
            return;
        }
        if (!await EnsureWebViewAsync())
        {
            if (expected == version && PreviewChosen)
            {
                Preview.Visibility = Visibility.Collapsed;
                PreviewMissing.Visibility = Visibility.Visible;
            }
            return;
        }
        var text = markdown!;
        var folder = baseDir;
        var file = bench.Paths.PreviewFile;
        var written = await Task.Run(() =>
        {
            var answer = CoreJson.Parse(NativeCore.PreviewHtml(CoreJson.PreviewRequest(text, dark, folder)));
            if (!answer.Ok || CoreJson.Text(answer.Result, "html") is not { } html)
                return false;
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(file)!);
                AtomicFile.Write(file, System.Text.Encoding.UTF8.GetBytes(PreviewRules.WithZoom(html, zoom)));
                return true;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        });
        if (expected != version || !PreviewChosen || !written)
            return;
        previewShown = key;
        PreviewMissing.Visibility = Visibility.Collapsed;
        Preview.Visibility = Visibility.Visible;
        Preview.CoreWebView2.Navigate(new Uri(file).AbsoluteUri);
    }

    Task<bool> EnsureWebViewAsync() => webView ??= StartWebViewAsync();

    /// <summary>Starts WebView2 with the rules of native/DESIGN.md 8. False when the runtime is missing.</summary>
    async Task<bool> StartWebViewAsync()
    {
        try
        {
            var environment = await CoreWebView2Environment.CreateWithOptionsAsync(null, bench.Paths.WebView2, new CoreWebView2EnvironmentOptions());
            await Preview.EnsureCoreWebView2Async(environment);
        }
        catch (Exception e) when (e is System.Runtime.InteropServices.COMException or FileNotFoundException or ArgumentException or InvalidOperationException or UnauthorizedAccessException)
        {
            return false;
        }
        var core = Preview.CoreWebView2;
        var settings = core.Settings;
        settings.IsScriptEnabled = false;
        settings.AreDefaultScriptDialogsEnabled = false;
        settings.AreDevToolsEnabled = false;
        settings.AreBrowserAcceleratorKeysEnabled = false;
        settings.IsStatusBarEnabled = false;
        settings.IsZoomControlEnabled = false;
        core.AddWebResourceRequestedFilter("*", CoreWebView2WebResourceContext.All);
        core.WebResourceRequested += (_, args) =>
        {
            if (PreviewRules.Blocks(args.Request.Uri))
                args.Response = core.Environment.CreateWebResourceResponse(null, 403, "Forbidden", "");
        };
        core.NavigationStarting += (_, args) =>
        {
            var decision = PreviewRules.Navigation(args.Uri, bench.Paths.PreviewFile);
            if (decision == LinkDecision.Allow)
                return;
            args.Cancel = true;
            OpenOutside(args.Uri, decision);
        };
        core.NewWindowRequested += (_, args) =>
        {
            args.Handled = true;
            OpenOutside(args.Uri, PreviewRules.Navigation(args.Uri, bench.Paths.PreviewFile));
        };
        core.ContextMenuRequested += (_, args) =>
        {
            var items = args.MenuItems;
            for (var i = items.Count - 1; i >= 0; i--)
                if (items[i].Name != "copy")
                    items.RemoveAt(i);
            if (items.Count == 0)
                args.Handled = true;
        };
        return true;
    }

    /// <summary>http, https and mailto links open in their default apps; everything else is refused.</summary>
    static void OpenOutside(string uri, LinkDecision decision)
    {
        if (decision == LinkDecision.OpenOutside)
            _ = Windows.System.Launcher.LaunchUriAsync(new Uri(uri));
    }

    void OnSettingsChanged()
    {
        MarkdownText.FontSize = MarkdownFontSize * bench.Settings.Zoom;
        if (markdown is not null && PreviewChosen)
            ShowView();
    }

    void DetailView_ActualThemeChanged(FrameworkElement sender, object args)
    {
        if (markdown is not null && PreviewChosen)
            ShowView();
    }

    void ViewSwitch_SelectionChanged(object sender, SelectionChangedEventArgs e) => ShowView();

    void SwitchView_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        ViewSwitch.SelectedIndex = PreviewChosen ? 1 : 0;
    }

    async void Copy_Click(object sender, RoutedEventArgs e)
    {
        if (finished is null)
            return;
        await MainWindow.Current.CopyAsync(finished);
        ShowCopied(true);
    }

    void ShowCopied(bool copied)
    {
        copiedTimer.Stop();
        CopyGlyph.Glyph = copied ? Ui.Accept : Ui.Copy;
        CopyButton.Label = s[copied ? "notice.copied" : "detail.copy"];
        if (copied)
        {
            Ui.Announce(CopyButton, s["notice.copied"]);
            copiedTimer.Start();
        }
    }

    async void SaveAs_Click(object sender, RoutedEventArgs e)
    {
        if (finished is not null)
            await MainWindow.Current.SaveAsAsync(finished);
    }

    void Show_Click(object sender, RoutedEventArgs e)
    {
        if (finished is not null)
            MainWindow.Current.ShowInExplorer(finished.MarkdownPath);
    }

    void Retry_Click(object sender, RoutedEventArgs e)
    {
        if (current is ConversionItem { State: Failed } item)
            MainWindow.Current.Retry(item);
    }

    void SecondButton_Click(object sender, RoutedEventArgs e)
    {
        var window = MainWindow.Current;
        switch (SecondButton.Tag)
        {
            case ErrorAction.Remove when current is ConversionItem item:
                window.Remove(item);
                break;
            case ErrorAction.OpenSettings:
                window.OpenSettings();
                break;
            case ErrorAction.OpenConfiguration:
                window.OpenConfiguration();
                break;
        }
    }

    void MessageButton_Click(object sender, RoutedEventArgs e)
    {
        if (current is HistoryEntry entry)
            MainWindow.Current.Remove(entry);
    }
}
