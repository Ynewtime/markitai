using Markitai.Core;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Controls;
using Windows.Globalization;
using Windows.Globalization.DateTimeFormatting;

namespace MarkitaiApp;

/// <summary>A message in the list column's InfoBar (desktop/DESIGN.md 6.7).</summary>
public sealed record Notice(string Text, InfoBarSeverity Severity);

/// <summary>
/// The app's state on the UI thread: settings, the Convert queue, History, and the
/// rules for adding sources. Windows read it and call it; it knows no window.
/// </summary>
public sealed class Workbench
{
    readonly DispatcherQueueTimer noticeTimer;

    public Workbench(Startup startup, Strings strings, DispatcherQueue ui)
    {
        Paths = startup.Paths;
        S = strings;
        CoreVersion = startup.CoreVersion!;
        var (medium, time) = DateFormats(strings);
        Text = new Present(strings, medium, time);
        Settings = AppSettings.Load(Paths.Settings);
        Queue = new(NativeCore.Instance, () => OutputFolder, action => ui.TryEnqueue(() => action()), TimeProvider.System);
        Queue.Finished += OnFinished;
        History = HistoryStore.Load(Paths.History, DateTimeOffset.UtcNow, out var setAside);
        noticeTimer = ui.CreateTimer();
        noticeTimer.Interval = TimeSpan.FromSeconds(8);
        noticeTimer.IsRepeating = false;
        noticeTimer.Tick += (_, _) => SetNotice(null);
        if (setAside is not null)
            ShowNotice(S.Format("history.reset", ("name", setAside)));
    }

    /// <summary>
    /// The medium date and short time of the app's language, with that language's clock
    /// (desktop/DESIGN.md 6.5): en is US English with the 12-hour clock, zh-Hans is Chinese
    /// with the 24-hour clock, whatever the region's formats or clock.
    /// </summary>
    static (Func<DateTimeOffset, string>, Func<DateTimeOffset, string>) DateFormats(Strings strings)
    {
        var (language, region, clock) = strings.Language == "zh-Hans"
            ? ("zh-Hans-CN", "CN", ClockIdentifiers.TwentyFourHour)
            : ("en-US", "US", ClockIdentifiers.TwelveHour);
        try
        {
            var medium = new DateTimeFormatter("month.abbreviated day year hour minute", [language], region, CalendarIdentifiers.Gregorian, clock);
            var time = new DateTimeFormatter("hour minute", [language], region, CalendarIdentifiers.Gregorian, clock);
            return (t => medium.Format(t), t => time.Format(t));
        }
        catch (ArgumentException)
        {
            var culture = System.Globalization.CultureInfo.GetCultureInfo(language);
            return (t => t.ToLocalTime().ToString("g", culture), t => t.ToLocalTime().ToString("t", culture));
        }
    }

    public AppPaths Paths { get; }
    public Strings S { get; }
    public Present Text { get; }
    public string CoreVersion { get; }
    public ConversionQueue Queue { get; }
    public HistoryStore History { get; }
    public AppSettings Settings { get; private set; }
    public Notice? Notice { get; private set; }

    public event Action? SettingsChanged;
    public event Action? NoticeChanged;
    public event Action<string>? Announce;

    /// <summary>Raised with the items one add created, so the list can select the first one.</summary>
    public event Action<IReadOnlyList<ConversionItem>, bool>? Added;

    public string OutputFolder => Settings.OutputFolder ?? DefaultOutputFolder;

    /// <summary>FOLDERID_Documents (which follows OneDrive redirection) + Markitai.</summary>
    public static string DefaultOutputFolder => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Markitai");

    public void UpdateSettings(Func<AppSettings, AppSettings> change)
    {
        Settings = change(Settings);
        try
        {
            Settings.Save(Paths.Settings);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
        SettingsChanged?.Invoke();
    }

    public void AddFiles(IEnumerable<string> paths, string baseDirectory)
    {
        var sorted = Intake.Paths(paths, baseDirectory);
        if (sorted.Sources.Count == 0 && sorted.RefusedFolders.Count == 0)
            return;
        SetNotice(null);
        Add(sorted.Sources);
        if (sorted.RefusedFolders.Count == 1)
            ShowNotice(S.Format("notice.folder", ("name", sorted.RefusedFolders[0])));
        else if (sorted.RefusedFolders.Count > 1)
            ShowNotice(S.Format("notice.folders", ("count", sorted.RefusedFolders.Count.ToString(S.Culture))));
    }

    /// <summary>Adds what the URL field holds as a web address; false when the text is not one.</summary>
    public bool AddUrl(string text) => AddWebAddress(Intake.WebAddress(text));

    /// <summary>Adds pasted or dropped text that is one http or https URL with its scheme; false otherwise.</summary>
    public bool AddLink(string text) => AddWebAddress(Intake.Link(text));

    bool AddWebAddress(string? url)
    {
        if (url is null)
            return false;
        SetNotice(null);
        Add([Source.Url(url)]);
        return true;
    }

    void Add(IReadOnlyList<Source> sources)
    {
        if (sources.Count == 0)
            return;
        var wasEmpty = Queue.Items.Count == 0;
        var result = Queue.Add(sources, Settings.Options);
        if (result.Duplicates.Count > 0)
            ShowNotice(S.Format("notice.duplicate", ("name", result.Duplicates[0].DisplayName)));
        if (result.Added.Count > 0)
            Added?.Invoke(result.Added, wasEmpty);
    }

    public void Retry(ConversionItem item) => Queue.Retry(item, Settings.Options);

    /// <summary>Removes History entries; their files stay. A History file that cannot be written keeps the change for this session.</summary>
    public void RemoveFromHistory(HistoryEntry entry) => KeepHistory(() => History.Remove(entry));

    public void ClearHistory() => KeepHistory(History.Clear);

    static void KeepHistory(Action change)
    {
        try
        {
            change();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    public void ShowNotice(string text, InfoBarSeverity severity = InfoBarSeverity.Warning)
    {
        SetNotice(new Notice(text, severity));
        noticeTimer.Stop();
        noticeTimer.Start();
    }

    public void SetNotice(Notice? notice)
    {
        if (notice is null)
            noticeTimer.Stop();
        Notice = notice;
        NoticeChanged?.Invoke();
    }

    void OnFinished(ConversionItem item)
    {
        if (item.State is Done done)
        {
            KeepHistory(() => History.Add(HistoryEntry.From(item, done, CoreVersion, DateTimeOffset.UtcNow)));
            Announce?.Invoke(S.Format("a11y.done", ("title", item.Title)));
        }
        else
        {
            Announce?.Invoke(S.Format("a11y.failed", ("title", item.Title)));
        }
    }
}
