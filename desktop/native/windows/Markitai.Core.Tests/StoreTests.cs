using System.Globalization;
using System.Text.Json;
using Markitai.Core;

namespace Markitai.Core.Tests;

public sealed class StoreTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("markitai-store-").FullName;

    public void Dispose() => Directory.Delete(dir, recursive: true);

    static readonly ConversionOptions Ai = new(true, false, false, false, false, ConfigSource.User);

    HistoryEntry Entry(int n)
    {
        var folder = Path.Combine(dir, "out", $"doc {n}");
        Directory.CreateDirectory(folder);
        var markdown = Path.Combine(folder, $"doc {n}.pdf.md");
        File.WriteAllText(markdown, $"# {n}");
        return new($"{Guid.NewGuid()}", $"/in/doc {n}.pdf", SourceKind.File, $"doc {n}.pdf", folder, markdown,
            new DateTimeOffset(2026, 11, 4, 2, 42, 18, TimeSpan.Zero).AddMinutes(n), 18.6, ["a warning"],
            new Usage(6, 19870, 4440, 0.0412, CostStatus.Complete), Ai, "2.3.2");
    }

    [Fact]
    public void The_file_has_the_specified_shape()
    {
        var path = Path.Combine(dir, "Markitai Native", "history.json");
        var store = HistoryStore.Load(path, DateTimeOffset.UtcNow, out var setAside);
        Assert.Null(setAside);
        store.Add(Entry(1));

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        Assert.Equal(1, root.GetProperty("version").GetInt32());
        var entry = root.GetProperty("entries")[0];
        Assert.Equal(
            ["id", "source", "kind", "title", "folder", "markdown_path", "finished_at", "duration", "warnings", "usage", "options", "markitai_version"],
            entry.EnumerateObject().Select(p => p.Name));
        Assert.Equal("file", entry.GetProperty("kind").GetString());
        Assert.Equal("2026-11-04T02:43:18Z", entry.GetProperty("finished_at").GetString());
        Assert.Equal(["requests", "input_tokens", "output_tokens", "cost_usd", "cost"], entry.GetProperty("usage").EnumerateObject().Select(p => p.Name));
        Assert.Equal("complete", entry.GetProperty("usage").GetProperty("cost").GetString());
        Assert.Equal("""{"llm":true,"ocr":false,"screenshot":false,"alt":false,"desc":false,"config":"user"}""",
            JsonSerializer.Serialize(entry.GetProperty("options")));
        Assert.False(File.Exists(path + ".tmp"));
    }

    [Fact]
    public void Five_hundred_and_one_additions_keep_the_newest_five_hundred()
    {
        var path = Path.Combine(dir, "history.json");
        var store = HistoryStore.Load(path, DateTimeOffset.UtcNow, out _);
        var first = Entry(0);
        store.Add(first);
        for (var n = 1; n <= 500; n++)
            store.Add(Entry(n) with { Folder = first.Folder });
        Assert.Equal(500, store.Entries.Count);
        Assert.DoesNotContain(first, store.Entries);
        Assert.Equal("doc 500.pdf", store.Entries[0].Title);

        var reread = HistoryStore.Load(path, DateTimeOffset.UtcNow, out _);
        Assert.Equal(store.Entries, reread.Entries, (a, b) => a.Id == b.Id && a.FinishedAt == b.FinishedAt && a.Usage == b.Usage);
    }

    [Fact]
    public void Remove_and_clear_keep_the_converted_files()
    {
        var path = Path.Combine(dir, "history.json");
        var store = HistoryStore.Load(path, DateTimeOffset.UtcNow, out _);
        var (a, b) = (Entry(1), Entry(2));
        store.Add(a);
        store.Add(b);
        store.Remove(a);
        Assert.Equal([b.Id], HistoryStore.Load(path, DateTimeOffset.UtcNow, out _).Entries.Select(e => e.Id));
        store.Clear();
        Assert.Empty(HistoryStore.Load(path, DateTimeOffset.UtcNow, out _).Entries);
        Assert.True(File.Exists(a.MarkdownPath));
        Assert.True(File.Exists(b.MarkdownPath));
    }

    [Theory]
    [InlineData("{not json")]
    [InlineData("""{"version":2,"entries":[]}""")]
    [InlineData("""{"version":1,"entries":[{"id":"x"}]}""")]
    public void An_unreadable_file_is_set_aside_once_and_history_starts_empty(string content)
    {
        var path = Path.Combine(dir, "history.json");
        File.WriteAllText(path, content);
        var now = new DateTimeOffset(2026, 11, 4, 2, 42, 18, TimeSpan.Zero);
        var store = HistoryStore.Load(path, now, out var setAside);
        Assert.Equal("history-unreadable-20261104T024218Z.json", setAside);
        Assert.Empty(store.Entries);
        Assert.Equal(content, File.ReadAllText(Path.Combine(dir, setAside!)));
        Assert.False(File.Exists(path));

        HistoryStore.Load(path, now.AddSeconds(1), out var again);
        Assert.Null(again);
    }

    [Fact]
    public void Settings_default_round_trip_and_survive_bad_files()
    {
        var path = Path.Combine(dir, "Markitai Native", "settings.json");
        var defaults = AppSettings.Load(path);
        Assert.Equal(new AppSettings(), defaults);
        Assert.Equal(1.0, defaults.Zoom);
        Assert.Equal((1200, 780), (defaults.WindowWidth, defaults.WindowHeight));
        Assert.Null(defaults.OutputFolder);

        var changed = defaults with { Llm = true, Alt = true, OutputFolder = Path.Combine(dir, "out"), Config = ConfigSource.Builtin, Zoom = 1.25, WindowMaximized = true };
        changed.Save(path);
        using (var doc = JsonDocument.Parse(File.ReadAllText(path)))
        {
            Assert.Equal(
                ["conversion.llm", "conversion.ocr", "conversion.screenshot", "conversion.desc", "conversion.alt", "output.folder",
                 "configuration.source", "view.zoom", "window.width", "window.height", "window.maximized"],
                doc.RootElement.EnumerateObject().Select(p => p.Name));
            Assert.Equal("builtin", doc.RootElement.GetProperty("configuration.source").GetString());
        }
        Assert.Equal(changed, AppSettings.Load(path));
        Assert.Equal(new ConversionOptions(true, false, false, true, false, ConfigSource.Builtin), changed.Options);

        File.WriteAllText(path, """{"view.zoom": 9, "output.folder": "relative\\path", "window.width": -5}""");
        var clamped = AppSettings.Load(path);
        Assert.Equal(AppSettings.MaxZoom, clamped.Zoom);
        Assert.Null(clamped.OutputFolder);
        Assert.Equal(1200, clamped.WindowWidth);

        File.WriteAllText(path, "][");
        Assert.Equal(new AppSettings(), AppSettings.Load(path));

        // A key the file leaves out keeps its default, however the serializer builds the record.
        File.WriteAllText(path, "{}");
        Assert.Equal(new AppSettings(), AppSettings.Load(path));
        File.WriteAllText(path, """{"output.folder": null, "window.width": 1000}""");
        Assert.Equal(new AppSettings { WindowWidth = 1000 }, AppSettings.Load(path));
    }

    [Fact]
    public void Save_as_copies_missing_assets_and_never_replaces_one()
    {
        var conversion = Path.Combine(dir, "report");
        Directory.CreateDirectory(Path.Combine(conversion, ".markitai", "assets"));
        File.WriteAllText(Path.Combine(conversion, ".markitai", "assets", "a.png"), "new a");
        File.WriteAllText(Path.Combine(conversion, ".markitai", "assets", "b.png"), "new b");
        var target = Path.Combine(dir, "saved", "report.md");
        Directory.CreateDirectory(Path.Combine(dir, "saved", ".markitai", "assets"));
        File.WriteAllText(Path.Combine(dir, "saved", ".markitai", "assets", "a.png"), "kept a");

        UserFiles.SaveMarkdown(target, "![x](.markitai/assets/a.png)", conversion);
        Assert.Equal("![x](.markitai/assets/a.png)", File.ReadAllText(target));
        Assert.Equal("kept a", File.ReadAllText(Path.Combine(dir, "saved", ".markitai", "assets", "a.png")));
        Assert.Equal("new b", File.ReadAllText(Path.Combine(dir, "saved", ".markitai", "assets", "b.png")));

        var plain = Path.Combine(dir, "plain", "p.md");
        Directory.CreateDirectory(Path.GetDirectoryName(plain)!);
        UserFiles.SaveMarkdown(plain, "no images", conversion);
        Assert.False(Directory.Exists(Path.Combine(dir, "plain", ".markitai")));
    }

    [Fact]
    public void Writable_folders_and_the_configuration_file()
    {
        Assert.True(UserFiles.CanWrite(Path.Combine(dir, "new", "folder")));
        Assert.Empty(Directory.GetFiles(Path.Combine(dir, "new", "folder")));
        var file = Path.Combine(dir, "a-file");
        File.WriteAllText(file, "x");
        Assert.False(UserFiles.CanWrite(Path.Combine(file, "below")));

        var config = Path.Combine(dir, "home", ".markitai", "config.json");
        UserFiles.CreateConfiguration(config);
        Assert.Equal("{}\n", File.ReadAllText(config));
        File.WriteAllText(config, """{"llm":{}}""");
        Assert.Throws<IOException>(() => UserFiles.CreateConfiguration(config));
        Assert.Equal("""{"llm":{}}""", File.ReadAllText(config));
    }
}

public sealed class IntakeTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("markitai-intake-").FullName;

    public void Dispose() => Directory.Delete(dir, recursive: true);

    [Theory]
    [InlineData("  example.com/a  ", "https://example.com/a")]
    [InlineData("http://example.com", "http://example.com")]
    [InlineData("HTTPS://Example.com/x?y=1#z", "HTTPS://Example.com/x?y=1#z")]
    [InlineData("ftp://example.com/file", null)]
    [InlineData("example", null)]
    [InlineData("https://", null)]
    [InlineData("mailto:someone@example.com", null)]
    [InlineData("two words.com", null)]
    [InlineData("", null)]
    public void Web_addresses(string text, string? expected) => Assert.Equal(expected, Intake.WebAddress(text));

    [Fact]
    public void Folders_are_refused_except_numbers_packages()
    {
        Directory.CreateDirectory(Path.Combine(dir, "Reports"));
        Directory.CreateDirectory(Path.Combine(dir, "Budget.NUMBERS"));
        File.WriteAllText(Path.Combine(dir, "a.pdf"), "x");
        var sorted = Intake.Paths(["a.pdf", Path.Combine(dir, "Reports"), Path.Combine(dir, "Budget.NUMBERS") + Path.DirectorySeparatorChar, "missing.docx", " "], dir);
        Assert.Equal([Path.Combine(dir, "a.pdf"), Path.Combine(dir, "Budget.NUMBERS"), Path.Combine(dir, "missing.docx")], sorted.Sources.Select(s => s.Value));
        Assert.Equal(["Reports"], sorted.RefusedFolders);
    }
}

public sealed class StringsTests
{
    static string Windows
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "Markitai.App")))
                dir = dir.Parent;
            return dir?.FullName ?? throw new DirectoryNotFoundException("desktop/native/windows above the test binaries");
        }
    }

    [Fact]
    public void Language_follows_the_first_preferred_language()
    {
        var culture = CultureInfo.InvariantCulture;
        Assert.Equal("zh-Hans", Strings.For("zh-Hant-TW", culture).Language);
        Assert.Equal("zh-Hans", Strings.For("ZH-cn", culture).Language);
        Assert.Equal("en", Strings.For("ja-JP", culture).Language);
        Assert.Equal("en", Strings.For(null, culture).Language);
        Assert.Equal("历史记录", Strings.For("zh-CN", culture)["nav.history"]);
        var en = Strings.For("en", CultureInfo.GetCultureInfo("en-US"));
        Assert.Equal("1,234 warnings", en.Plural("state.warnings", 1234));
        Assert.Equal("1 warning", en.Plural("state.warnings", 1));
        Assert.Equal("Clear all 7 entries?", en.Plural("history.clear.title", 7));
        Assert.Equal("Clear the 1 entry?", en.Plural("history.clear.title", 1));
    }

    /// <summary>Every copy key the app's code and XAML name exists in desktop/strings.json, so no visible text bypasses it.</summary>
    [Fact]
    public void Every_key_the_app_uses_is_in_strings_json()
    {
        var app = Path.Combine(Windows, "Markitai.App");
        Assert.True(Directory.Exists(app), app);
        var pattern = new System.Text.RegularExpressions.Regex("""(?:Key=|(?:\[|\? |: |Format\(|Plural\(|Entry\()")([a-z0-9_]+(?:\.[a-z0-9_]+)+)""");
        var used = Directory.EnumerateFiles(app, "*.*", SearchOption.AllDirectories)
            .Where(f => (f.EndsWith(".xaml") || f.EndsWith(".cs")) && !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .SelectMany(f => pattern.Matches(File.ReadAllText(f)).Select(m => m.Groups[1].Value))
            .ToHashSet();
        Assert.NotEmpty(used);
        var keys = Strings.Keys.ToHashSet();
        var plurals = keys.Where(k => k.EndsWith(".one")).Select(k => k[..^4]);
        Assert.Empty(used.Except(keys).Except(plurals));
    }
}
