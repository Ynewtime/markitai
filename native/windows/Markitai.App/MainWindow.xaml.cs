using System.Diagnostics;
using System.Runtime.InteropServices;
using Markitai.Core;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Input;
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

/// <summary>A section 11 shortcut of the window. Forwarded ones also work with focus in the preview.</summary>
public sealed record Shortcut(VirtualKey Key, VirtualKeyModifiers Modifiers, Func<bool> Run, bool Forwarded = true);

public sealed partial class MainWindow : Window
{
    const double StackBelow = 760;
    const uint WmQueryEndSession = 0x0011;
    const uint WmEndSession = 0x0016;

    readonly Workbench bench = App.Bench;
    readonly Strings s = App.Strings;
    Page page = Page.Convert;
    bool stacked;
    bool detailShown;
    bool allowClose;
    bool askingToQuit;
    ContentDialog? quitDialog;
    bool nothingLeftToLose;
    bool blockingSessionEnd;
    bool queueHadItems;
    readonly List<Shortcut> shortcuts = [];
    static nint previousWindowProc;

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
        AddShortcuts();
        HookSessionEnd();

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
        Activated += FocusAtLaunch;
    }

    nint Hwnd => WinRT.Interop.WindowNative.GetWindowHandle(this);

    /// <summary>DESIGN.md 6.2: Choose Files… has focus once the window is up (focus set before activation does not stick).</summary>
    void FocusAtLaunch(object sender, WindowActivatedEventArgs args)
    {
        if (args.WindowActivationState == WindowActivationState.Deactivated)
            return;
        Activated -= FocusAtLaunch;
        DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () =>
        {
            if (page == Page.Convert && bench.Queue.Items.Count == 0)
                ConvertEmptyView.FocusChoose();
        });
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

    /// <summary>A second launch's files, full paths (Program.Relaunch makes relative ones full first).</summary>
    public void OnRedirected(IReadOnlyList<string> files)
    {
        bench.AddFiles(files, Environment.CurrentDirectory);
        Activate();
        SetForegroundWindow(Hwnd);
    }

    /// <summary>Which of list, detail, empty states and Settings show, and whether list and detail stack.</summary>
    void ShowRegions()
    {
        var convertEmpty = page == Page.Convert && bench.Queue.Items.Count == 0;
        if (convertEmpty && queueHadItems)
            DispatcherQueue.TryEnqueue(() => ConvertEmptyView.FocusChoose());
        queueHadItems = bench.Queue.Items.Count > 0;
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
        await AskToQuitAsync();
    }

    string QuitBody => bench.Text.QuitBody(bench.Queue.ConvertingCount, bench.Queue.QueuedCount);

    /// <summary>
    /// The quit dialog (DESIGN.md 7.3), after closing any other dialog. Its count follows the
    /// queue; once nothing is Converting it closes and the app quits.
    /// </summary>
    async Task AskToQuitAsync()
    {
        if (askingToQuit)
            return;
        askingToQuit = true;
        try
        {
            await Dialogs.CloseAsync();
            if (bench.Queue.ConvertingCount == 0)
            {
                Quit();
                return;
            }
            nothingLeftToLose = false;
            quitDialog = Dialogs.Confirmation(Root.XamlRoot, s["quit.title"], QuitBody, s["quit.confirm"]);
            bench.Queue.StateChanged += FollowQueue;
            bench.Queue.Items.CollectionChanged += FollowQueue;
            ContentDialogResult? answer;
            try
            {
                answer = await Dialogs.ShowAsync(quitDialog);
            }
            finally
            {
                bench.Queue.StateChanged -= FollowQueue;
                bench.Queue.Items.CollectionChanged -= FollowQueue;
                quitDialog = null;
                if (blockingSessionEnd)
                    BlockSessionEnd(false);
            }
            if (answer == ContentDialogResult.Primary || nothingLeftToLose)
                Quit();
        }
        finally
        {
            askingToQuit = false;
        }
    }

    void FollowQueue(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) => FollowQueue();

    void FollowQueue()
    {
        if (quitDialog is null)
            return;
        if (bench.Queue.ConvertingCount == 0)
        {
            nothingLeftToLose = true;
            quitDialog.Hide();
            return;
        }
        quitDialog.Content = QuitBody;
        if (blockingSessionEnd)
            BlockSessionEnd(true);
    }

    /// <summary>A confirmed Quit: delete what running conversions claimed, then close.</summary>
    void Quit()
    {
        bench.Queue.DiscardConverting();
        allowClose = true;
        Close();
    }

    /// <summary>Logoff, restart and shutdown reach the window as WM_QUERYENDSESSION and WM_ENDSESSION, not AppWindow.Closing.</summary>
    unsafe void HookSessionEnd() =>
        previousWindowProc = SetWindowLongPtrW(Hwnd, -4, (nint)(delegate* unmanaged<nint, uint, nint, nint, nint>)&WindowProc);

    [UnmanagedCallersOnly]
    static nint WindowProc(nint hwnd, uint message, nint wParam, nint lParam)
    {
        try
        {
            if (message == WmQueryEndSession && Current.QueryEndSession())
                return 0;
            if (message == WmEndSession)
                Current.EndSession(wParam != 0);
        }
        catch (Exception e)
        {
            App.Log(App.Bench.Paths, e);
        }
        return CallWindowProcW(previousWindowProc, hwnd, message, wParam, lParam);
    }

    /// <summary>True blocks the session end: an item is Converting, so Windows shows the quit sentence and the window shows the quit dialog.</summary>
    bool QueryEndSession()
    {
        SaveWindow();
        if (bench.Queue.ConvertingCount == 0)
            return false;
        BlockSessionEnd(true);
        DispatcherQueue.TryEnqueue(async () => await AskToQuitAsync());
        return true;
    }

    /// <summary>The session ends whatever the dialog says: do what a confirmed Quit does before Windows ends the process.</summary>
    void EndSession(bool ending)
    {
        if (!ending)
        {
            BlockSessionEnd(false);
            return;
        }
        bench.Queue.DiscardConverting();
        Environment.Exit(0);
    }

    void BlockSessionEnd(bool block)
    {
        blockingSessionEnd = block;
        if (block)
            ShutdownBlockReasonCreate(Hwnd, QuitBody);
        else
            ShutdownBlockReasonDestroy(Hwnd);
    }

    double Scale => GetDpiForWindow(Hwnd) / 96.0;

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

    /// <summary>Section 11's window shortcuts, as keyboard accelerators of the root and as the keys the preview forwards.</summary>
    void AddShortcuts()
    {
        const VirtualKeyModifiers Control = VirtualKeyModifiers.Control;
        bool Zoom(double to)
        {
            bench.UpdateSettings(settings => settings with { Zoom = Math.Round(Math.Clamp(to, AppSettings.MinZoom, AppSettings.MaxZoom), 2) });
            return true;
        }

        shortcuts.AddRange([
            new(VirtualKey.O, Control, ChooseFiles),
            new(VirtualKey.L, Control, AddUrl),
            new(VirtualKey.V, Control, Paste),
            new(VirtualKey.Number1, Control, () => Go(Page.Convert, true)),
            new(VirtualKey.Number2, Control, () => Go(Page.History, true)),
            new(VirtualKey.F, Control, Search),
            new((VirtualKey)188, Control, () => Go(Page.Settings, true)),
            new((VirtualKey)187, Control, () => Zoom(bench.Settings.Zoom + 0.1)),
            new(VirtualKey.Add, Control, () => Zoom(bench.Settings.Zoom + 0.1)),
            new((VirtualKey)189, Control, () => Zoom(bench.Settings.Zoom - 0.1)),
            new(VirtualKey.Subtract, Control, () => Zoom(bench.Settings.Zoom - 0.1)),
            new(VirtualKey.Number0, Control, () => Zoom(1.0)),
            new(VirtualKey.NumberPad0, Control, () => Zoom(1.0)),
            new(VirtualKey.Escape, VirtualKeyModifiers.None, Back),
            new(VirtualKey.Left, VirtualKeyModifiers.Menu, Back),
            new(VirtualKey.Delete, VirtualKeyModifiers.None, RemoveSelected, Forwarded: false),
        ]);
        foreach (var shortcut in shortcuts)
        {
            var accelerator = new KeyboardAccelerator { Key = shortcut.Key, Modifiers = shortcut.Modifiers };
            accelerator.Invoked += (_, e) => e.Handled = shortcut.Run();
            Root.KeyboardAccelerators.Add(accelerator);
        }
    }

    /// <summary>The window shortcuts the preview forwards (its web view keeps the keys it is given).</summary>
    public IEnumerable<Shortcut> ForwardedShortcuts => shortcuts.Where(s => s.Forwarded);

    /// <summary>Runs a window shortcut pressed in the preview; false when none matched or it did nothing.</summary>
    public bool RunForwarded(VirtualKey key, VirtualKeyModifiers modifiers) =>
        ForwardedShortcuts.FirstOrDefault(s => s.Key == key && s.Modifiers == modifiers) is { } shortcut && shortcut.Run();

    /// <summary>
    /// Ctrl+, from a text field: the field takes Ctrl with the comma key before keyboard
    /// accelerators see it (Ctrl+2 and Ctrl+L still reach them), so the window looks first.
    /// </summary>
    void Root_PreviewKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == (VirtualKey)188 && Pressed(VirtualKey.Control) && !Pressed(VirtualKey.Shift) && !Pressed(VirtualKey.Menu) && Ui.FocusInText(Root.XamlRoot))
            e.Handled = Go(Page.Settings, true);
    }

    static bool Pressed(VirtualKey key) => InputKeyboardSource.GetKeyStateForCurrentThread(key).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);

    bool Go(Page target, bool handled)
    {
        Go(target);
        return handled;
    }

    bool ChooseFiles()
    {
        Go(Page.Convert);
        _ = ChooseFilesAsync();
        return true;
    }

    bool AddUrl()
    {
        Go(Page.Convert);
        if (bench.Queue.Items.Count == 0)
            ConvertEmptyView.FocusUrl();
        else
            ConvertPane.FocusUrl();
        return true;
    }

    bool Paste()
    {
        if (Ui.FocusInText(Root.XamlRoot) || page == Page.Settings)
            return false;
        _ = PasteAsync();
        return true;
    }

    async Task PasteAsync()
    {
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

    bool Search()
    {
        Go(Page.History);
        HistoryPane.FocusSearch();
        return true;
    }

    bool RemoveSelected()
    {
        if (Ui.FocusInText(Root.XamlRoot))
            return false;
        if (page == Page.Convert && ConvertPane.SelectedItem is { } item)
            return ConvertPane.Remove(item);
        if (page == Page.History && HistoryPane.SelectedEntry is { } entry)
            return HistoryPane.Remove(entry);
        return false;
    }

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
    private static partial nint SetWindowLongPtrW(nint hwnd, int index, nint value);

    [LibraryImport("user32.dll")]
    private static partial nint CallWindowProcW(nint previous, nint hwnd, uint message, nint wParam, nint lParam);

    [LibraryImport("user32.dll", StringMarshalling = StringMarshalling.Utf16)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool ShutdownBlockReasonCreate(nint hwnd, string reason);

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool ShutdownBlockReasonDestroy(nint hwnd);

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool SetForegroundWindow(nint hwnd);
}
