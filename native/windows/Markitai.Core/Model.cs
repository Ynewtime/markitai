using System.ComponentModel;
using System.Text.Json.Serialization;

namespace Markitai.Core;

public enum SourceKind
{
    [JsonStringEnumMemberName("file")] File,
    [JsonStringEnumMemberName("url")] Url,
}

/// <summary>An absolute file path or an http(s) URL, as the core receives it.</summary>
public sealed record Source(SourceKind Kind, string Value)
{
    public static Source File(string path) => new(SourceKind.File, path);

    public static Source Url(string url) => new(SourceKind.Url, url);

    /// <summary>A file's name, or the URL without its scheme: the title before a page names itself.</summary>
    public string DisplayName => Kind == SourceKind.File
        ? FileName(Value)
        : Value[(Value.IndexOf("://", StringComparison.Ordinal) is var i and >= 0 ? i + 3 : 0)..];

    public bool SameAs(Source other) =>
        Kind == other.Kind && string.Equals(Value, other.Value,
            Kind == SourceKind.File && OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    static string FileName(string path)
    {
        var trimmed = path.TrimEnd('/', '\\');
        var cut = trimmed.LastIndexOfAny(['/', '\\']);
        return cut < 0 ? trimmed : trimmed[(cut + 1)..];
    }
}

public enum ConfigSource
{
    [JsonStringEnumMemberName("user")] User,
    [JsonStringEnumMemberName("builtin")] Builtin,
}

public sealed record ConversionOptions(bool Llm, bool Ocr, bool Screenshot, bool Alt, bool Desc, ConfigSource Config);

public enum CostStatus
{
    [JsonStringEnumMemberName("complete")] Complete,
    [JsonStringEnumMemberName("partial")] Partial,
    [JsonStringEnumMemberName("unknown")] Unknown,
}

/// <summary>Model use of a conversion; absent when no request was made.</summary>
public sealed record Usage(long Requests, long InputTokens, long OutputTokens, double CostUsd, CostStatus Cost)
{
    [JsonIgnore] public long Tokens => InputTokens + OutputTokens;
}

public sealed record ConversionResult(
    string Markdown,
    string MarkdownPath,
    double Duration,
    IReadOnlyList<string> Warnings,
    Usage? Usage,
    string? PageTitle);

public abstract record ItemState;

public sealed record Queued : ItemState
{
    public static readonly Queued Instance = new();
}

public sealed record Converting(DateTimeOffset StartedAt) : ItemState;

public sealed record Done(string Folder, ConversionResult Result) : ItemState;

public sealed record Failed(string Code, string Message) : ItemState;

/// <summary>One source in the Convert list. Owned by the UI thread.</summary>
public sealed class ConversionItem : INotifyPropertyChanged
{
    ItemState state = Queued.Instance;

    internal ConversionItem(Source source, ConversionOptions options, DateTimeOffset addedAt, long order)
    {
        Source = source;
        Options = options;
        AddedAt = addedAt;
        Order = order;
    }

    public Guid Id { get; } = Guid.NewGuid();

    public Source Source { get; }

    public DateTimeOffset AddedAt { get; }

    public ConversionOptions Options { get; internal set; }

    internal long Order { get; set; }

    public ItemState State
    {
        get => state;
        internal set
        {
            state = value;
            PropertyChanged?.Invoke(this, new(nameof(State)));
            PropertyChanged?.Invoke(this, new(nameof(Title)));
        }
    }

    public string Title => State is Done { Result.PageTitle: { } page } && Source.Kind == SourceKind.Url
        ? page
        : Source.DisplayName;

    public event PropertyChangedEventHandler? PropertyChanged;
}
