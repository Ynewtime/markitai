using System.Globalization;

namespace Markitai.Core;

/// <summary>The second button of a Failed item's error block (desktop/DESIGN.md 9).</summary>
public enum ErrorAction
{
    None,
    Remove,
    OpenSettings,
    OpenConfiguration,
}

/// <summary>User-visible text for items, history entries and results (desktop/DESIGN.md 4.1, 4.2, 6.4, 6.5, 9).</summary>
public sealed class Present(Strings strings, Func<DateTimeOffset, string> mediumDateTime, Func<DateTimeOffset, string> shortTime)
{
    static readonly Dictionary<string, (string Key, ErrorAction Action)> Errors = new()
    {
        ["not_found"] = ("error.not_found", ErrorAction.Remove),
        ["is_directory"] = ("error.is_directory", ErrorAction.Remove),
        ["no_model_configured"] = ("error.no_model_configured", ErrorAction.OpenSettings),
        ["invalid_input"] = ("error.invalid_input", ErrorAction.None),
        ["unsupported"] = ("error.unsupported", ErrorAction.Remove),
        ["conversion_error"] = ("error.conversion_error", ErrorAction.None),
        ["fetch_error"] = ("error.fetch_error", ErrorAction.None),
        ["config_error"] = ("error.config_error", ErrorAction.OpenConfiguration),
        ["io_error"] = ("error.io_error", ErrorAction.OpenSettings),
        ["output_unavailable"] = ("error.output_unavailable", ErrorAction.OpenSettings),
        ["image_only"] = ("error.image_only", ErrorAction.OpenSettings),
        ["skipped"] = ("error.skipped", ErrorAction.None),
        ["invalid_json"] = ("error.internal", ErrorAction.None),
        ["internal_error"] = ("error.internal", ErrorAction.None),
    };

    public Strings Strings => strings;

    public static ErrorAction ActionFor(string code) => Errors.TryGetValue(code, out var row) ? row.Action : ErrorAction.None;

    /// <summary>The mapped sentence; {folder} is the run's output folder, else <paramref name="outputFolder"/>.</summary>
    public string ErrorSentence(Failed failed, string outputFolder) =>
        strings.Format(Errors.TryGetValue(failed.Code, out var row) ? row.Key : "error.unknown",
            ("folder", failed.Folder ?? outputFolder), ("settings", strings["nav.settings"]), ("reason", failed.Message));

    /// <summary>The quit dialog's body (desktop/DESIGN.md 7.3). Chinese sentences join without a space.</summary>
    public string QuitBody(int converting, int queued)
    {
        var body = strings.Plural("quit.body", converting);
        if (queued == 0)
            return body;
        return body + (strings.Language == "zh-Hans" ? "" : " ") + strings["quit.body.queued"];
    }

    /// <summary>One decimal below 60 s, at least 0.1 s (desktop/DESIGN.md 4.2); minutes and seconds above.</summary>
    public string Duration(double seconds)
    {
        if (seconds < 60)
            return strings.Format("unit.seconds", ("value", Math.Max(seconds, 0.1).ToString("0.0", strings.Culture)));
        var whole = (long)Math.Round(seconds);
        return strings.Format("unit.minutes", ("minutes", (whole / 60).ToString(strings.Culture)), ("seconds", (whole % 60).ToString(strings.Culture)));
    }

    public static string Elapsed(TimeSpan elapsed)
    {
        var seconds = Math.Max(0, (long)elapsed.TotalSeconds);
        return $"{seconds / 60}:{seconds % 60:00}";
    }

    public static string Cost(double usd) =>
        "$" + usd.ToString(usd < 1 ? "0.0000" : "0.00", CultureInfo.InvariantCulture);

    public string Warnings(int count) => strings.Plural("state.warnings", count);

    /// <summary>"Converted in 18.6 s · 24,310 tokens · $0.0412".</summary>
    public string Footer(ConversionResult result) => Footer(result.Duration, result.Usage);

    public string Footer(double duration, Usage? usage)
    {
        var text = strings.Format("footer.duration", ("duration", Duration(duration)));
        if (usage is null)
            return text;
        var cost = usage.Cost switch
        {
            CostStatus.Complete => Cost(usage.CostUsd),
            CostStatus.Partial => strings.Format("footer.cost.at_least", ("cost", Cost(usage.CostUsd))),
            _ => strings["footer.cost.unknown"],
        };
        return $"{text} · {strings.Format("footer.tokens", ("count", usage.Tokens.ToString("N0", strings.Culture)))} · {cost}";
    }

    /// <summary>A list row's subtitle (desktop/DESIGN.md 4.1).</summary>
    public string Subtitle(ItemState state, string outputFolder) => state switch
    {
        Queued => strings["state.queued"],
        Converting => strings["state.converting"],
        Done done => WithWarnings(strings.Format("state.done", ("duration", Duration(done.Result.Duration))), done.Result.Warnings.Count),
        Failed failed => strings.Format("row.failed", ("reason", ErrorSentence(failed, outputFolder))),
        _ => "",
    };

    public string HistorySubtitle(HistoryEntry entry, DateTimeOffset now) => HistorySubtitle(entry.FinishedAt, entry.Warnings.Count, now);

    public string HistorySubtitle(DateTimeOffset finishedAt, int warnings, DateTimeOffset now) => WithWarnings(When(finishedAt, now), warnings);

    /// <summary>The History detail's status line: "Done in 18.6 s · Today, 10:42".</summary>
    public string HistoryStatus(HistoryEntry entry, DateTimeOffset now) =>
        $"{strings.Format("state.done", ("duration", Duration(entry.Duration)))} · {When(entry.FinishedAt, now)}";

    /// <summary>Today, Yesterday, or the platform's medium date and short time, in local time.</summary>
    public string When(DateTimeOffset finished, DateTimeOffset now)
    {
        var day = finished.ToLocalTime().Date;
        var today = now.ToLocalTime().Date;
        if (day == today)
            return strings.Format("date.today", ("time", shortTime(finished)));
        if (day == today.AddDays(-1))
            return strings.Format("date.yesterday", ("time", shortTime(finished)));
        return mediumDateTime(finished);
    }

    string WithWarnings(string text, int count) => count == 0 ? text : $"{text} · {Warnings(count)}";
}
