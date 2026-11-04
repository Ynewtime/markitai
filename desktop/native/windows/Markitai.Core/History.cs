using System.Collections.ObjectModel;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Markitai.Core;

/// <summary>One finished conversion in history.json (desktop/DESIGN.md 4.4).</summary>
public sealed record HistoryEntry(
    string Id,
    string Source,
    SourceKind Kind,
    string Title,
    string Folder,
    string MarkdownPath,
    [property: JsonConverter(typeof(UtcSecondsConverter))] DateTimeOffset FinishedAt,
    double Duration,
    IReadOnlyList<string> Warnings,
    Usage? Usage,
    ConversionOptions Options,
    string MarkitaiVersion)
{
    public static HistoryEntry From(ConversionItem item, Done done, string markitaiVersion, DateTimeOffset finishedAt) => new(
        item.Id.ToString(), item.Source.Value, item.Source.Kind, item.Title, done.Folder, done.Result.MarkdownPath,
        finishedAt, done.Result.Duration, done.Result.Warnings, done.Result.Usage, item.Options, markitaiVersion);

    /// <summary>The Markdown its Convert item showed: the file at <paramref name="path"/> without its front matter (desktop/DESIGN.md 3.3).</summary>
    public static string ReadMarkdown(string path)
    {
        var text = File.ReadAllText(path);
        var answer = CoreJson.Parse(NativeCore.MarkdownBody(CoreJson.BodyRequest(text)));
        return answer.Ok ? CoreJson.Text(answer.Result, "markdown") ?? text : text;
    }
}

public sealed record HistoryFile(int Version, IReadOnlyList<HistoryEntry> Entries);

/// <summary>history.json: newest first, at most 500, rewritten whole after every change.</summary>
public sealed class HistoryStore
{
    public const int Limit = 500;

    HistoryStore(string path, IEnumerable<HistoryEntry> entries)
    {
        Path = path;
        Entries = new(entries);
    }

    public string Path { get; }

    public ObservableCollection<HistoryEntry> Entries { get; }

    /// <summary>
    /// Reads the file. A missing file is an empty history; an unreadable one is renamed to
    /// history-unreadable-&lt;UTC time&gt;.json, whose name comes back in <paramref name="setAside"/>.
    /// </summary>
    public static HistoryStore Load(string path, DateTimeOffset now, out string? setAside)
    {
        setAside = null;
        if (!File.Exists(path))
            return new(path, []);
        try
        {
            var file = JsonSerializer.Deserialize(File.ReadAllBytes(path), CoreJsonContext.Default.HistoryFile);
            if (file is { Version: 1 })
                return new(path, file.Entries.Take(Limit));
        }
        catch (JsonException)
        {
        }
        setAside = $"history-unreadable-{now.UtcDateTime.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture)}.json";
        File.Move(path, System.IO.Path.Combine(System.IO.Path.GetDirectoryName(path)!, setAside));
        return new(path, []);
    }

    public void Add(HistoryEntry entry)
    {
        Entries.Insert(0, entry);
        while (Entries.Count > Limit)
            Entries.RemoveAt(Entries.Count - 1);
        Save();
    }

    public void Remove(HistoryEntry entry)
    {
        if (Entries.Remove(entry))
            Save();
    }

    public void Clear()
    {
        Entries.Clear();
        Save();
    }

    void Save() => AtomicFile.Write(Path, JsonSerializer.SerializeToUtf8Bytes(new HistoryFile(1, [.. Entries]), CoreJsonContext.Default.HistoryFile));
}

sealed class UtcSecondsConverter : JsonConverter<DateTimeOffset>
{
    public override DateTimeOffset Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        DateTimeOffset.TryParse(reader.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var value)
            ? value
            : throw new JsonException("finished_at is not an RFC 3339 time");

    public override void Write(Utf8JsonWriter writer, DateTimeOffset value, JsonSerializerOptions options) =>
        writer.WriteStringValue(value.UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture));
}

public static class AtomicFile
{
    /// <summary>Writes a temporary file in the same folder, flushes it and renames it over <paramref name="path"/>.</summary>
    public static void Write(string path, byte[] bytes)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        var temporary = path + ".tmp";
        using (var stream = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }
        File.Move(temporary, path, overwrite: true);
    }
}
