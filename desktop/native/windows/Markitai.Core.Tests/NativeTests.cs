using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Markitai.Core;

namespace Markitai.Core.Tests;

/// <summary>
/// The real markitai_app library through the app's own P/Invoke layer. Runs when
/// MARKITAI_APP_LIBRARY names the built library and MARKITAI_HOME an isolated
/// folder (build.py --check sets both, with HOME and the Windows profile folders).
/// </summary>
public sealed class NativeTests : IDisposable
{
    static readonly string? Library = Environment.GetEnvironmentVariable("MARKITAI_APP_LIBRARY");
    static readonly string? MarkitaiHome = Environment.GetEnvironmentVariable("MARKITAI_HOME");
    static readonly ConversionOptions Builtin = new(false, false, false, false, false, ConfigSource.Builtin);
    static readonly ConversionOptions User = Builtin with { Config = ConfigSource.User };

    readonly string dir = Directory.CreateTempSubdirectory("markitai-native-").FullName;
    readonly ManualThread ui = new();

    // prepare_environment makes the data folder the working directory on Windows, and
    // Windows cannot delete a process's working directory.
    readonly string workingDirectory = Directory.GetCurrentDirectory();

    static NativeTests()
    {
        if (!string.IsNullOrEmpty(Library))
            NativeLibrary.SetDllImportResolver(typeof(NativeCore).Assembly,
                (name, _, _) => name == "markitai_app" ? NativeLibrary.Load(Library) : IntPtr.Zero);
    }

    public NativeTests()
    {
        Assert.SkipWhen(string.IsNullOrEmpty(Library) || string.IsNullOrEmpty(MarkitaiHome),
            "MARKITAI_APP_LIBRARY and an isolated MARKITAI_HOME are not set; build.py --check sets them");
        Assert.NotEqual(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".markitai"), Path.GetFullPath(MarkitaiHome!));
        Directory.CreateDirectory(MarkitaiHome!);
        File.Delete(ConfigPath);
    }

    public void Dispose()
    {
        if (!string.IsNullOrEmpty(MarkitaiHome))
            File.Delete(ConfigPath);
        Directory.SetCurrentDirectory(workingDirectory);
        Directory.Delete(dir, recursive: true);
    }

    static string ConfigPath => Path.Combine(MarkitaiHome!, "config.json");

    static string Repository
    {
        get
        {
            var at = new DirectoryInfo(AppContext.BaseDirectory);
            while (at is not null && !File.Exists(Path.Combine(at.FullName, "crates", "markitai-app", "Cargo.toml")))
                at = at.Parent;
            return at?.FullName ?? throw new DirectoryNotFoundException("the repository above the test binaries");
        }
    }

    static string Fixture(params string[] parts) => Path.Combine([Repository, "crates", "markitai-core", "src", "formats", .. parts]);

    ConversionQueue Queue(string output) => new(NativeCore.Instance, () => output, ui.Post, TimeProvider.System);

    void Finish(ConversionQueue queue)
    {
        while (queue.Items.Any(i => i.State is Queued or Converting))
            ui.RunOne(TimeSpan.FromMinutes(2));
    }

    string Author(string name, string text)
    {
        var path = Path.Combine(dir, "in", name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    [Fact]
    public void The_library_answers_its_abi_version_previews_and_folder_claims()
    {
        Assert.Equal(NativeCore.ExpectedAbiVersion, NativeCore.AbiVersion());
        Assert.Matches(@"^\d+\.\d+\.\d+", NativeCore.Version());
        if (OperatingSystem.IsWindows())
        {
            var prepared = CoreJson.Parse(NativeCore.PrepareEnvironment(JsonSerializer.Serialize(new { data_dir = Path.Combine(dir, "data") })));
            Assert.True(prepared.Ok, prepared.Message);
            Assert.Equal("inherited", CoreJson.Text(prepared.Result, "login_environment"));
            Assert.Equal(Path.Combine(dir, "data"), Directory.GetCurrentDirectory());
        }

        var baseDir = Path.Combine(dir, "report");
        Directory.CreateDirectory(Path.Combine(baseDir, ".markitai", "assets"));
        File.WriteAllBytes(Path.Combine(baseDir, ".markitai", "assets", "a.png"), [0x89, 0x50]);
        var markdown = "# 你好\n\n| a | b |\n|---|--:|\n| 1 | 2 |\n\n- [x] done\n\n<b>raw</b> note[^1]\n\n![local](.markitai/assets/a.png) ![remote](https://example.com/x.png)\n\n[^1]: A footnote.\n";
        var request = CoreJson.PreviewRequest(markdown, dark: true, baseDir);
        var first = CoreJson.Parse(NativeCore.PreviewHtml(request));
        Assert.True(first.Ok, first.Message);
        var html = CoreJson.Text(first.Result, "html")!;
        Assert.StartsWith("<!doctype html>", html);
        Assert.Contains("data-theme=\"dark\"", html);
        Assert.Contains("<h1>你好</h1>", html);
        Assert.Contains("&lt;b&gt;raw&lt;/b&gt;", html);
        Assert.Contains("type=\"checkbox\"", html);
        Assert.Contains("footnote-definition", html);
        Assert.Contains("src=\"file://", html);
        Assert.Contains("class=\"img-remote\" title=\"https://example.com/x.png\"", html);
        Assert.Equal(html, CoreJson.Text(CoreJson.Parse(NativeCore.PreviewHtml(request)).Result, "html"));

        var parent = Path.Combine(dir, "out");
        var claims = new[] { Path.Combine(dir, "x", "report.pdf"), Path.Combine(dir, "y", "report.docx") }
            .Select(source => CoreJson.Parse(NativeCore.Instance.ClaimOutputFolder(CoreJson.ClaimRequest(parent, Source.File(source)))))
            .ToArray();
        Assert.Equal(["report", "report (2)"], claims.Select(c => CoreJson.Text(c.Result, "name")));
        Assert.All(claims, c => Assert.True(Directory.Exists(CoreJson.Text(c.Result, "path"))));

        var config = CoreJson.Parse(NativeCore.ConfigFile());
        Assert.Equal(ConfigPath, CoreJson.Text(config.Result, "path"));
        Assert.False(config.Result.GetProperty("exists").GetBoolean());
    }

    /// <summary>DESIGN.md 3.3: an entry shows the Markdown its Convert item did, also from a file edited since.</summary>
    [Theory]
    [InlineData("---\ntitle: Report\n---\n\n# Report\n", "# Report\n")]
    [InlineData("---\n\nIntro after a rule.\n\n---\n\nSecond part.\n", "---\n\nIntro after a rule.\n\n---\n\nSecond part.\n")]
    [InlineData("---\r\ntitle: CRLF\r\n---\r\n\r\n# CRLF body\r\n", "# CRLF body\r\n")]
    [InlineData("\uFEFF---\ntitle: BOM\n---\n\n# BOM body\n", "# BOM body\n")]
    [InlineData("---\ntitle: Only\n---\n", "")]
    public void A_history_entry_reads_its_markdown_without_front_matter(string file, string shown)
    {
        var path = Author("doc.md", file);
        Assert.Equal(shown, HistoryEntry.ReadMarkdown(path));
    }

    /// <summary>DESIGN.md 16, States and Folders: Done with and without warnings, and every Failed row a real input reaches.</summary>
    [Fact]
    public void Repository_fixtures_reach_done_and_each_failure_and_failures_leave_no_folder()
    {
        var output = Path.Combine(dir, "Markitai");
        var queue = Queue(output);
        var notes = Author("notes.md", "# Notes\n\n你好, plain text.\n");
        var folder = Path.Combine(dir, "in", "a folder");
        Directory.CreateDirectory(folder);
        var sources = new Dictionary<string, Source>
        {
            ["done"] = Source.File(notes),
            ["warnings"] = Source.File(Fixture("pdf", "fixtures", "page-chrome", "how-to-cjk-footer.pdf")),
            ["not_found"] = Source.File(Path.Combine(dir, "in", "missing.pdf")),
            ["is_directory"] = Source.File(folder),
            ["unsupported"] = Source.File(Author("blob.xyz", "\0\u0001binary")),
            ["conversion_error"] = Source.File(Author("broken.docx", "not a zip")),
            ["image_only"] = Source.File(Fixture("native", "fixtures", "office-images", "diagram.png")),
            ["invalid_input"] = Source.Url("ftp://example.com/a"),
            ["fetch_error"] = Source.Url("http://127.0.0.1:9/unreachable"),
        };
        var added = queue.Add(sources.Values, Builtin);
        Assert.Equal(sources.Count, added.Added.Count);
        Assert.Equal(2, queue.ConvertingCount);
        Finish(queue);

        ItemState StateOf(string key) => queue.Items.Single(i => i.Source == sources[key]).State;
        var done = Assert.IsType<Done>(StateOf("done"));
        Assert.Empty(done.Result.Warnings);
        Assert.Null(done.Result.Usage);
        Assert.Contains("你好", done.Result.Markdown);
        Assert.Equal(Path.Combine(output, "notes"), done.Folder);
        Assert.True(File.Exists(done.Result.MarkdownPath));
        var warned = Assert.IsType<Done>(StateOf("warnings"));
        Assert.NotEmpty(warned.Result.Warnings);
        foreach (var code in sources.Keys.Except(["done", "warnings"]))
            Assert.Equal(code, Assert.IsType<Failed>(StateOf(code)).Code);
        Assert.Equal(new[] { done.Folder, warned.Folder }.Order(StringComparer.Ordinal), Directory.GetDirectories(output).Order(StringComparer.Ordinal));

        var again = queue.Add([Source.File(notes)], Builtin);
        Finish(queue);
        Assert.Equal(Path.Combine(output, "notes (2)"), Assert.IsType<Done>(again.Added[0].State).Folder);

        var file = Author("not-a-folder", "x");
        var blocked = Queue(Path.Combine(file, "below"));
        blocked.Add([Source.File(notes)], Builtin);
        Finish(blocked);
        Assert.Equal("output_unavailable", Assert.IsType<Failed>(blocked.Items[0].State).Code);
    }

    [Fact]
    public void A_broken_or_keyless_configuration_fails_with_its_own_code()
    {
        var output = Path.Combine(dir, "Markitai");
        var notes = Author("config.md", $"# Config {Guid.NewGuid()}\n\nText.\n");

        File.WriteAllText(ConfigPath, "{ not json");
        var broken = Queue(output);
        broken.Add([Source.File(notes)], User);
        Finish(broken);
        Assert.Equal("config_error", Assert.IsType<Failed>(broken.Items[0].State).Code);

        File.WriteAllText(ConfigPath, """
            {"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/gpt-4.1-mini","api_key":"env:MARKITAI_TEST_UNSET_KEY"}}]}}
            """);
        var keyless = Queue(output);
        keyless.Add([Source.File(notes)], User with { Llm = true });
        Finish(keyless);
        Assert.Equal("no_model_configured", Assert.IsType<Failed>(keyless.Items[0].State).Code);
        Assert.False(Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any());
    }

    /// <summary>DESIGN.md 16, States: Done with AI usage, against a model served on this machine.</summary>
    [Fact]
    public void Enhance_with_ai_reports_usage_from_a_local_model()
    {
        var text = $"# Stub {Guid.NewGuid():N}\n\nA paragraph for the local model.\n";
        using var model = new StubModel(text);
        File.WriteAllText(ConfigPath, $$$"""
            {"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/gpt-4.1-mini","api_base":"http://127.0.0.1:{{{model.Port}}}/v1","api_key":"stub-not-a-key"}}]}}
            """);
        var queue = Queue(Path.Combine(dir, "Markitai"));
        queue.Add([Source.File(Author("ai.md", text))], User with { Llm = true, Alt = true });
        Finish(queue);

        var done = Assert.IsType<Done>(queue.Items[0].State);
        Assert.EndsWith(".llm.md", done.Result.MarkdownPath);
        Assert.Equal(text, done.Result.Markdown);
        Assert.True(model.Requests >= 1);
        var usage = Assert.IsType<Usage>(done.Result.Usage);
        Assert.Equal(model.Requests, usage.Requests);
        Assert.Equal(1500 * model.Requests, usage.Tokens);
        Assert.Equal(CostStatus.Unknown, usage.Cost);
        var present = new Present(Strings.For("en", CultureInfo.GetCultureInfo("en-US")), t => "", t => "");
        Assert.EndsWith($" · {usage.Tokens:N0} tokens · cost unknown", present.Footer(done.Result));

        var history = HistoryStore.Load(Path.Combine(dir, "history.json"), DateTimeOffset.UtcNow, out _);
        history.Add(HistoryEntry.From(queue.Items[0], done, NativeCore.Version(), DateTimeOffset.UtcNow));
        using var file = JsonDocument.Parse(File.ReadAllText(history.Path));
        var entry = file.RootElement.GetProperty("entries")[0];
        Assert.Equal(done.Result.MarkdownPath, entry.GetProperty("markdown_path").GetString());
        Assert.Equal("unknown", entry.GetProperty("usage").GetProperty("cost").GetString());
        Assert.True(entry.GetProperty("options").GetProperty("alt").GetBoolean());
    }
}

/// <summary>An OpenAI-compatible chat endpoint on 127.0.0.1 that hands the document back unchanged.</summary>
sealed class StubModel : IDisposable
{
    readonly TcpListener listener = new(IPAddress.Loopback, 0);
    readonly byte[] answer;
    int requests;

    public StubModel(string cleanedMarkdown)
    {
        var content = JsonSerializer.Serialize(new { cleaned_markdown = cleanedMarkdown, frontmatter = new { description = "A note the local stub model returned.", tags = new[] { "stub" } } });
        answer = JsonSerializer.SerializeToUtf8Bytes(new
        {
            id = "chatcmpl-stub",
            @object = "chat.completion",
            created = 0,
            model = "gpt-4.1-mini",
            choices = new[] { new { index = 0, message = new { role = "assistant", content }, finish_reason = "stop" } },
            usage = new { prompt_tokens = 1200, completion_tokens = 300, total_tokens = 1500 },
        });
        listener.Start();
        _ = Task.Run(Serve);
    }

    public int Port => ((IPEndPoint)listener.LocalEndpoint).Port;

    public int Requests => Volatile.Read(ref requests);

    async Task Serve()
    {
        while (true)
        {
            TcpClient client;
            try
            {
                client = await listener.AcceptTcpClientAsync();
            }
            catch (Exception e) when (e is SocketException or ObjectDisposedException)
            {
                return;
            }
            _ = Task.Run(() => Answer(client));
        }
    }

    async Task Answer(TcpClient client)
    {
        using var _ = client;
        var stream = client.GetStream();
        var received = new List<byte>();
        var buffer = new byte[8192];
        int headerEnd;
        while ((headerEnd = IndexOf(received, "\r\n\r\n"u8)) < 0)
        {
            var n = await stream.ReadAsync(buffer);
            if (n == 0)
                return;
            received.AddRange(buffer.AsSpan(0, n));
        }
        var headers = Encoding.ASCII.GetString([.. received.Take(headerEnd)]);
        var length = headers.Split("\r\n").Where(h => h.StartsWith("content-length:", StringComparison.OrdinalIgnoreCase))
            .Select(h => int.Parse(h[15..].Trim(), CultureInfo.InvariantCulture)).FirstOrDefault();
        while (received.Count < headerEnd + 4 + length)
        {
            var n = await stream.ReadAsync(buffer);
            if (n == 0)
                return;
            received.AddRange(buffer.AsSpan(0, n));
        }
        Interlocked.Increment(ref requests);
        var head = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {answer.Length}\r\nConnection: close\r\n\r\n";
        await stream.WriteAsync(Encoding.ASCII.GetBytes(head));
        await stream.WriteAsync(answer);
    }

    static int IndexOf(List<byte> bytes, ReadOnlySpan<byte> pattern)
    {
        for (var i = 0; i + pattern.Length <= bytes.Count; i++)
        {
            var match = true;
            for (var j = 0; j < pattern.Length && match; j++)
                match = bytes[i + j] == pattern[j];
            if (match)
                return i;
        }
        return -1;
    }

    public void Dispose() => listener.Stop();
}
