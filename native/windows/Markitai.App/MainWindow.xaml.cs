using System.Diagnostics;
using System.Runtime.InteropServices;
using Markitai.Core;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.ApplicationModel.DataTransfer;
using Windows.System;

namespace MarkitaiApp;

public enum Page
{
    Convert,
    History,
    Settings,
}

/// <summary>A Done item or a History entry: what Copy, Save As and Show act on.</summary>
public sealed record Finished(string Title, string Folder, string MarkdownPath, string? Markdown)
{
    public static Finished Of(ConversionItem item, Done done) => new(item.Title, done.Folder, done.Result.MarkdownPath, done.Result.Markdown);

    public static Finished Of(HistoryEntry entry) => new(entry.Title, entry.Folder, entry.MarkdownPath, null);

    public async Task<string> ReadAsync() => Markdown ?? await Task.Run(() => File.ReadAllText(MarkdownPath));
}

public sealed partial class MainWindow : Window
{
    const double StackBelow = 760;

    readonly Workbench bench = App.Bench;
    readonly Strings s = App.Strings;
    Page page = Page.Convert;
    bool stacked;
    bool detailShown;
    bool allowClose;
    bool dialogOpen;

    public MainWindow()
    {
        InitializeComponent();
        Current = this;
        Title = s["app.name"];
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        var assets = Path.Combine(AppContext.BaseDirectory, "Assets");
        if (File.Exists(Path.Combine(assets, "Markitai.ico")))
            AppWindow.SetIcon(Path.Combine(assets, "Markitai.ico"));
        TitleIcon.Source = new BitmapImage(new Uri(Path.Combine(assets, "icon.png"))) { DecodePixelWidth = 32 };
        if (MicaController.IsSupported())
            SystemBackdrop = new MicaBackdrop();
        else
            Root.Background = (Brush)Application.Current.Resources["SolidBackgroundFillColorBaseBrush"];
        RestoreWindow();
        AddCodeAccelerators();

        Nav.SelectedItem = ConvertItem;
        bench.Queue.Items.CollectionChanged += (_, _) => ShowRegions();
        bench.History.Entries.CollectionChanged += (_, _) => ShowRegions();
        bench.Announce += text => Ui.Announce(Root, text);
        bench.Added += OnAdded;
        ConvertPane.Selected += (item, chosen) => OnSelected(Page.Convert, item, chosen);
        HistoryPane.Selected += (entry, chosen) => OnSelected(Page.History, entry, chosen);
        AppWindow.Closing += AppWindow_Closing;
        Closed += (_, _) =>
        {
            if (bench.Queue.ConvertingCount > 0)
                Environment.Exit(0);
        };
        ShowRegions();
        ConvertEmptyView.FocusChoose();
    }

    public static new MainWindow Current { get; private set; } = null!;

    public Page Page => page;

    public void Go(Page target)
    {
        Nav.SelectedItem = target switch
        {
            Page.Convert => ConvertItem,
            Page.History => HistoryItem,
            _ => SettingsItem,
        };
    }

    void Nav_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        page = (args.SelectedItem as NavigationViewItem)?.Tag switch
        {
            "history" => Page.History,
            "settings" => Page.Settings,
            _ => Page.Convert,
        };
        detailShown = false;
        if (page == Page.Settings)
            SettingsView.Refresh();
        Detail.Show(page switch
        {
            Page.Convert => ConvertPane.SelectedItem,
            Page.History => HistoryPane.SelectedEntry,
            _ => null,
        });
        ShowRegions();
    }

    void OnSelected(Page from, object? selection, bool chosen)
    {
        if (from != page)
            return;
        Detail.Show(selection);
        if (chosen && stacked && selection is not null)
            detailShown = true;
        ShowRegions();
    }

    void OnAdded(IReadOnlyList<ConversionItem> added, bool wasEmpty)
    {
        if (page != Page.Convert)
            Go(Page.Convert);
        if (wasEmpty)
            ConvertPane.Select(added[0]);
    }

    public void OnRedirected(IReadOnlyList<string> files)
    {
        bench.AddFiles(files, Environment.CurrentDirectory);
        Activate();
        SetForegroundWindow(WinRT.Interop.WindowNative.GetWindowHandle(this));
    }

    /// <summary>Which of list, detail, empty states and Settings show, and whether list and detail stack.</summary>
    void ShowRegions()
    {
        var convertEmpty = page == Page.Convert && bench.Queue.Items.Count == 0;
        var historyEmpty = page == Page.History && bench.History.Entries.Count == 0;
        var panes = page != Page.Settings && !convertEmpty && !historyEmpty;
        stacked = Workspace.ActualWidth > 0 && Workspace.ActualWidth < StackBelow;
        if (!panes || !stacked)
            detailShown = false;
        Panes.Visibility = Ui.Show(panes);
        ConvertPane.Visibility = Ui.Show(page == Page.Convert);
        HistoryPane.Visibility = Ui.Show(page == Page.History);
        ConvertEmptyView.Visibility = Ui.Show(convertEmpty);
        HistoryEmptyView.Visibility = Ui.Show(historyEmpty);
        SettingsView.Visibility = Ui.Show(page == Page.Settings);
        ListColumn.Width = !stacked ? new GridLength(320) : detailShown ? new GridLength(0) : new GridLength(1, GridUnitType.Star);
        DetailColumn.Width = stacked && !detailShown ? new GridLength(0) : new GridLength(1, GridUnitType.Star);
        Detail.Visibility = Ui.Show(!stacked || detailShown);
        Nav.IsBackButtonVisible = detailShown ? NavigationViewBackButtonVisible.Visible : NavigationViewBackButtonVisible.Collapsed;
    }

    void Workspace_SizeChanged(object sender, SizeChangedEventArgs e) => ShowRegions();

    void Nav_DisplayModeChanged(NavigationView sender, NavigationViewDisplayModeChangedEventArgs args) =>
        Workspace.Margin = args.DisplayMode == NavigationViewDisplayMode.Minimal ? new Thickness(0, 40, 0, 0) : new Thickness(0);

    void Nav_BackRequested(NavigationView sender, NavigationViewBackRequestedEventArgs args) => Back();

    bool Back()
    {
        if (!detailShown)
            return false;
        detailShown = false;
        ShowRegions();
        if (page == Page.Convert)
            ConvertPane.FocusList();
        else
            HistoryPane.FocusList();
        return true;
    }

    public void OpenSettings() => Go(Page.Settings);

    public async Task ChooseFilesAsync()
    {
        var picker = new Microsoft.Windows.Storage.Pickers.FileOpenPicker(AppWindow.Id);
        picker.FileTypeFilter.Add("*");
        var picked = await picker.PickMultipleFilesAsync();
        if (picked is { Count: > 0 })
            bench.AddFiles(picked.Select(f => f.Path), Environment.CurrentDirectory);
    }

    public async Task CopyAsync(Finished finished)
    {
        var package = new DataPackage();
        package.SetText(await finished.ReadAsync());
        Clipboard.SetContent(package);
    }

    public async Task SaveAsAsync(Finished finished)
    {
        var picker = new Microsoft.Windows.Storage.Pickers.FileSavePicker(AppWindow.Id)
        {
            SuggestedFileName = Path.GetFileName(finished.Folder) + ".md",
            DefaultFileExtension = ".md",
        };
        picker.FileTypeChoices.Add(s["detail.markdown"], [".md"]);
        var target = await picker.PickSaveFileAsync();
        if (target is null)
            return;
        try
        {
            var text = await finished.ReadAsync();
            await Task.Run(() => UserFiles.SaveMarkdown(target.Path, text, finished.Folder));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            bench.ShowNotice(s["error.save_failed"], InfoBarSeverity.Error);
        }
    }

    public void ShowInExplorer(string file)
    {
        var argument = File.Exists(file) ? $"/select,\"{file}\"" : $"\"{Path.GetDirectoryName(file)}\"";
        Process.Start(new ProcessStartInfo("explorer.exe", argument) { UseShellExecute = false });
    }

    /// <summary>Opens a file in its default app, or says it could not.</summary>
    public void OpenFile(string path)
    {
        if (!TryOpenFile(path))
            bench.ShowNotice(s.Format("error.open_failed", ("name", Path.GetFileName(path))), InfoBarSeverity.Error);
    }

    public bool TryOpenFile(string path)
    {
        try
        {
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
            return true;
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or IOException)
        {
            return false;
        }
    }

    /// <summary>Opens a folder in File Explorer, creating it first (the default output folder appears with the first conversion).</summary>
    public void OpenFolder(string folder)
    {
        try
        {
            Directory.CreateDirectory(folder);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
        Process.Start(new ProcessStartInfo("explorer.exe", $"\"{folder}\"") { UseShellExecute = false });
    }

    public void OpenConfiguration()
    {
        if (CoreJson.Text(CoreJson.Parse(NativeCore.ConfigFile()).Result, "path") is { } path)
            OpenFile(path);
    }

    public void Retry(ConversionItem item) => bench.Retry(item);

    public void Remove(ConversionItem item) => ConvertPane.Remove(item);

    public void Remove(HistoryEntry entry) => HistoryPane.Remove(entry);

    async void AppWindow_Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        SaveWindow();
        if (allowClose || bench.Queue.ConvertingCount == 0)
            return;
        args.Cancel = true;
        if (dialogOpen)
            return;
        var body = s.Plural("quit.body", bench.Queue.ConvertingCount);
        if (bench.Queue.QueuedCount > 0)
            body += " " + s["quit.body.queued"];
        dialogOpen = true;
        var quit = await Ui.ConfirmAsync(Root.XamlRoot, s["quit.title"], body, s["quit.confirm"]);
        dialogOpen = false;
        if (quit)
        {
            bench.Queue.DiscardConverting();
            allowClose = true;
            Close();
        }
    }

    double Scale => GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(this)) / 96.0;

    void RestoreWindow()
    {
        var settings = bench.Settings;
        var scale = Scale;
        AppWindow.Resize(new Windows.Graphics.SizeInt32((int)(settings.WindowWidth * scale), (int)(settings.WindowHeight * scale)));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.PreferredMinimumWidth = (int)(640 * scale);
            presenter.PreferredMinimumHeight = (int)(480 * scale);
            if (settings.WindowMaximized)
                presenter.Maximize();
        }
    }

    void SaveWindow()
    {
        var maximized = AppWindow.Presenter is OverlappedPresenter { State: OverlappedPresenterState.Maximized };
        var scale = Scale;
        bench.UpdateSettings(settings => maximized
            ? settings with { WindowMaximized = true }
            : settings with { WindowMaximized = false, WindowWidth = Math.Round(AppWindow.Size.Width / scale), WindowHeight = Math.Round(AppWindow.Size.Height / scale) });
    }

    void AddCodeAccelerators()
    {
        void Add(VirtualKey key, VirtualKeyModifiers modifiers, Action action)
        {
            var accelerator = new KeyboardAccelerator { Key = key, Modifiers = modifiers };
            accelerator.Invoked += (_, e) =>
            {
                action();
                e.Handled = true;
            };
            Root.KeyboardAccelerators.Add(accelerator);
        }

        void Zoom(double to) => bench.UpdateSettings(settings => settings with { Zoom = Math.Round(Math.Clamp(to, AppSettings.MinZoom, AppSettings.MaxZoom), 2) });

        Add((VirtualKey)188, VirtualKeyModifiers.Control, () => Go(Page.Settings));
        Add((VirtualKey)187, VirtualKeyModifiers.Control, () => Zoom(bench.Settings.Zoom + 0.1));
        Add(VirtualKey.Add, VirtualKeyModifiers.Control, () => Zoom(bench.Settings.Zoom + 0.1));
        Add((VirtualKey)189, VirtualKeyModifiers.Control, () => Zoom(bench.Settings.Zoom - 0.1));
        Add(VirtualKey.Subtract, VirtualKeyModifiers.Control, () => Zoom(bench.Settings.Zoom - 0.1));
        Add(VirtualKey.Number0, VirtualKeyModifiers.Control, () => Zoom(1.0));
        Add(VirtualKey.NumberPad0, VirtualKeyModifiers.Control, () => Zoom(1.0));
    }

    async void ChooseFiles_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        Go(Page.Convert);
        await ChooseFilesAsync();
    }

    void AddUrl_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        Go(Page.Convert);
        if (bench.Queue.Items.Count == 0)
            ConvertEmptyView.FocusUrl();
        else
            ConvertPane.FocusUrl();
    }

    async void Paste_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        if (Ui.FocusInText(Root.XamlRoot) || page == Page.Settings)
            return;
        args.Handled = true;
        var content = Clipboard.GetContent();
        if (content.Contains(StandardDataFormats.StorageItems))
        {
            var items = await content.GetStorageItemsAsync();
            bench.AddFiles(items.Select(i => i.Path), Environment.CurrentDirectory);
        }
        else if (content.Contains(StandardDataFormats.Text))
        {
            bench.AddUrl(await content.GetTextAsync());
        }
    }

    void GoConvert_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        Go(Page.Convert);
    }

    void GoHistory_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        Go(Page.History);
    }

    void Search_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        Go(Page.History);
        HistoryPane.FocusSearch();
    }

    void Remove_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        if (Ui.FocusInText(Root.XamlRoot))
            return;
        if (page == Page.Convert && ConvertPane.SelectedItem is { } item)
            args.Handled = ConvertPane.Remove(item);
        else if (page == Page.History && HistoryPane.SelectedEntry is { } entry)
            args.Handled = HistoryPane.Remove(entry);
    }

    void Back_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args) => args.Handled = Back();

    void SetDragging(bool dragging)
    {
        ConvertPane.SetDragging(dragging);
        ConvertEmptyView.SetDragging(dragging);
    }

    void Root_DragOver(object sender, DragEventArgs e)
    {
        var view = e.DataView;
        var accepted = page != Page.Settings
                       && (view.Contains(StandardDataFormats.StorageItems) || view.Contains(StandardDataFormats.WebLink) || view.Contains(StandardDataFormats.Text));
        e.AcceptedOperation = accepted ? DataPackageOperation.Copy : DataPackageOperation.None;
        if (accepted)
            e.DragUIOverride.Caption = s["convert.drop.release"];
        SetDragging(accepted);
    }

    void Root_DragLeave(object sender, DragEventArgs e) => SetDragging(false);

    async void Root_Drop(object sender, DragEventArgs e)
    {
        SetDragging(false);
        if (page == Page.Settings)
            return;
        var deferral = e.GetDeferral();
        try
        {
            var view = e.DataView;
            if (view.Contains(StandardDataFormats.StorageItems))
                bench.AddFiles((await view.GetStorageItemsAsync()).Select(i => i.Path), Environment.CurrentDirectory);
            else if (view.Contains(StandardDataFormats.WebLink))
                bench.AddUrl((await view.GetWebLinkAsync()).AbsoluteUri);
            else if (view.Contains(StandardDataFormats.Text))
                bench.AddUrl(await view.GetTextAsync());
        }
        finally
        {
            deferral.Complete();
        }
    }

    [LibraryImport("user32.dll")]
    private static partial uint GetDpiForWindow(nint hwnd);

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool SetForegroundWindow(nint hwnd);
}
