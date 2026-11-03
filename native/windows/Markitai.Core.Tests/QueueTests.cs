using System.Collections.Concurrent;
using System.Text.Json;
using Markitai.Core;

namespace Markitai.Core.Tests;

/// <summary>A converter whose conversions wait until the test releases them, answering as the core would.</summary>
sealed class GatedConverter : IConverter
{
    readonly ConcurrentDictionary<string, TaskCompletionSource<string>> gates = new();
    public readonly ConcurrentQueue<string> Started = new();
    public string Parent = "";
    public bool RefuseClaims;

    public string ClaimOutputFolder(string request)
    {
        if (RefuseClaims)
            return """{"ok":false,"error":{"code":"output_unavailable","message":"Access is denied."}}""";
        using var doc = JsonDocument.Parse(request);
        var source = doc.RootElement.GetProperty("source").GetString()!;
        var folder = Path.Combine(doc.RootElement.GetProperty("parent").GetString()!, Path.GetFileNameWithoutExtension(source) + "-" + Guid.NewGuid().ToString("N")[..6]);
        Directory.CreateDirectory(folder);
        return JsonSerializer.Serialize(new { ok = true, result = new { path = folder, name = Path.GetFileName(folder) } });
    }

    public string ConvertJson(string request)
    {
        using var doc = JsonDocument.Parse(request);
        var source = doc.RootElement.GetProperty("source").GetString()!;
        Interlocked.Increment(ref running);
        Started.Enqueue(source);
        var folder = doc.RootElement.GetProperty("options").GetProperty("output_dir").GetString()!;
        File.WriteAllText(Path.Combine(folder, "partial.md"), "written before the outcome");
        try
        {
            return Gate(source).Task.Result.Replace("{folder}", JsonEncodedText.Encode(folder).ToString());
        }
        finally
        {
            Interlocked.Decrement(ref running);
        }
    }

    int running;

    public int Running => Volatile.Read(ref running);

    public void ReleaseAll(string answer)
    {
        foreach (var gate in gates.Values)
            gate.TrySetResult(answer);
    }

    public TaskCompletionSource<string> Gate(string source) => gates.GetOrAdd(source, _ => new(TaskCreationOptions.RunContinuationsAsynchronously));

    public static string Ok(string markdown) =>
        $$$"""{"ok":true,"result":{"markdown":"{{{markdown}}}","llm_markdown":null,"output_path":"{folder}/x.md","llm_output_path":null,"frontmatter":{},"usage":{"cost_usd":0,"requests":0,"input_tokens":0,"output_tokens":0,"by_model":{}},"skip_reason":null,"duration":1.5,"warnings":[]}}""";

    public const string NotFound = """{"ok":false,"error":{"code":"not_found","message":"No such file"}}""";
}

/// <summary>Runs posted callbacks when the test pumps, like a UI thread's dispatcher.</summary>
sealed class ManualThread
{
    readonly BlockingCollection<Action> posted = new();

    public void Post(Action action) => posted.Add(action);

    public void RunOne(TimeSpan? timeout = null)
    {
        Assert.True(posted.TryTake(out var action, timeout ?? TimeSpan.FromSeconds(10)), "a conversion finished");
        action();
    }
}

public sealed class QueueTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("markitai-queue-").FullName;
    readonly GatedConverter converter = new();
    readonly ManualThread ui = new();
    readonly ConversionQueue queue;
    static readonly ConversionOptions Plain = new(false, false, false, false, false, ConfigSource.User);

    public QueueTests()
    {
        queue = new(converter, () => dir, ui.Post, TimeProvider.System);
    }

    public void Dispose()
    {
        foreach (var item in queue.Items)
            converter.Gate(item.Source.Value);
        converter.ReleaseAll(GatedConverter.NotFound);
        WaitFor(() => converter.Running == 0);
        Thread.Sleep(50);
        Directory.Delete(dir, recursive: true);
    }

    static Source F(string name) => Source.File(Path.Combine(Path.GetTempPath(), name));

    static void WaitFor(Func<bool> condition)
    {
        var until = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < until, "condition reached in time");
            Thread.Sleep(5);
        }
    }

    [Fact]
    public void Two_convert_at_once_and_the_oldest_waiting_item_starts_next()
    {
        queue.Add([F("a.pdf"), F("b.pdf"), F("c.pdf")], Plain);
        queue.Add([F("d.pdf")], Plain);

        Assert.Equal(["d.pdf", "a.pdf", "b.pdf", "c.pdf"], queue.Items.Select(i => i.Title));
        Assert.Equal(2, queue.ConvertingCount);
        Assert.IsType<Converting>(queue.Items[1].State);
        Assert.IsType<Converting>(queue.Items[2].State);
        Assert.IsType<Queued>(queue.Items[3].State);
        Assert.IsType<Queued>(queue.Items[0].State);

        converter.Gate(F("b.pdf").Value).SetResult(GatedConverter.Ok("b"));
        ui.RunOne();
        Assert.IsType<Done>(queue.Items[2].State);
        Assert.IsType<Converting>(queue.Items[3].State);
        Assert.IsType<Queued>(queue.Items[0].State);
        Assert.Equal(2, queue.ConvertingCount);
    }

    [Fact]
    public void A_source_already_waiting_or_converting_is_a_duplicate_until_it_finishes()
    {
        var first = queue.Add([F("a.pdf"), F("a.pdf")], Plain);
        Assert.Single(first.Added);
        Assert.Equal([F("a.pdf")], first.Duplicates);

        Assert.Single(queue.Add([F("a.pdf")], Plain).Duplicates);

        converter.Gate(F("a.pdf").Value).SetResult(GatedConverter.Ok("a"));
        ui.RunOne();
        var again = queue.Add([F("a.pdf")], Plain);
        Assert.Single(again.Added);
        Assert.Equal(2, queue.Items.Count);
    }

    [Fact]
    public void A_failed_conversion_deletes_its_folder_and_a_done_one_keeps_it()
    {
        queue.Add([F("ok.pdf"), F("gone.pdf")], Plain);
        converter.Gate(F("ok.pdf").Value).SetResult(GatedConverter.Ok("fine"));
        converter.Gate(F("gone.pdf").Value).SetResult(GatedConverter.NotFound);
        ui.RunOne();
        ui.RunOne();

        var done = Assert.IsType<Done>(queue.Items.Single(i => i.Title == "ok.pdf").State);
        Assert.True(Directory.Exists(done.Folder));
        Assert.Equal("fine", done.Result.Markdown);
        var failed = Assert.IsType<Failed>(queue.Items.Single(i => i.Title == "gone.pdf").State);
        Assert.Equal("not_found", failed.Code);
        Assert.Single(Directory.GetDirectories(dir));
    }

    [Fact]
    public void A_refused_folder_claim_fails_as_output_unavailable_without_converting()
    {
        converter.RefuseClaims = true;
        queue.Add([F("a.pdf")], Plain);
        ui.RunOne();
        var failed = Assert.IsType<Failed>(queue.Items[0].State);
        Assert.Equal("output_unavailable", failed.Code);
        Assert.Equal("Access is denied.", failed.Message);
        Assert.Empty(converter.Started);
    }

    [Fact]
    public void Retry_requeues_with_new_options_and_converting_items_cannot_be_removed()
    {
        var finished = new List<ConversionItem>();
        queue.Finished += finished.Add;
        queue.Add([F("a.pdf")], Plain);
        var item = queue.Items[0];
        Assert.False(queue.Remove(item));

        converter.Gate(F("a.pdf").Value).SetResult(GatedConverter.NotFound);
        ui.RunOne();
        Assert.Equal([item], finished);

        var withAi = Plain with { Llm = true };
        queue.Retry(item, withAi);
        Assert.Equal(withAi, item.Options);
        Assert.IsType<Converting>(item.State);
        WaitFor(() => converter.Started.Count == 2);
    }

    /// <summary>DESIGN.md 7.3: Quit deletes what the Converting items claimed and keeps what finished before it.</summary>
    [Fact]
    public void Quitting_deletes_the_folders_of_converting_items_and_keeps_finished_ones()
    {
        queue.Add([F("a.pdf"), F("b.pdf"), F("c.pdf")], Plain);
        converter.Gate(F("a.pdf").Value).SetResult(GatedConverter.Ok("a"));
        ui.RunOne();
        var done = Assert.IsType<Done>(queue.Items.Single(i => i.Title == "a.pdf").State);
        WaitFor(() => converter.Started.Count == 3);
        Assert.Equal(3, Directory.GetDirectories(dir).Length);

        queue.DiscardConverting();

        Assert.Equal([done.Folder], Directory.GetDirectories(dir));
        Assert.True(File.Exists(Path.Combine(done.Folder, "partial.md")));
        converter.ReleaseAll(GatedConverter.Ok("late"));
        ui.RunOne();
        Assert.All(queue.Items.Where(i => i.Title != "a.pdf"), i => Assert.IsType<Converting>(i.State));
    }

    [Fact]
    public void Clear_finished_keeps_waiting_and_converting_items()
    {
        queue.Add([F("a.pdf"), F("b.pdf"), F("c.pdf")], Plain);
        converter.Gate(F("a.pdf").Value).SetResult(GatedConverter.Ok("a"));
        ui.RunOne();
        Assert.True(queue.HasFinished);
        queue.ClearFinished();
        Assert.Equal(["b.pdf", "c.pdf"], queue.Items.Select(i => i.Title));
        Assert.False(queue.HasFinished);
    }

    [Fact]
    public void The_request_sends_alt_and_desc_only_with_llm_and_an_empty_config_for_builtin()
    {
        var plain = JsonDocument.Parse(CoreJson.ConvertRequest(F("a.pdf"), "/out/a", Plain with { Alt = true, Desc = true })).RootElement;
        var options = plain.GetProperty("options");
        Assert.Equal("/out/a", options.GetProperty("output_dir").GetString());
        Assert.False(options.GetProperty("llm").GetBoolean());
        Assert.False(options.TryGetProperty("alt", out _));
        Assert.False(options.TryGetProperty("desc", out _));
        Assert.False(options.TryGetProperty("config", out _));

        var ai = JsonDocument.Parse(CoreJson.ConvertRequest(F("a.pdf"), "/out/a", Plain with { Llm = true, Desc = true, Config = ConfigSource.Builtin })).RootElement.GetProperty("options");
        Assert.True(ai.GetProperty("desc").GetBoolean());
        Assert.False(ai.GetProperty("alt").GetBoolean());
        Assert.Equal(JsonValueKind.Object, ai.GetProperty("config").ValueKind);
        Assert.Empty(ai.GetProperty("config").EnumerateObject());
    }
}
