using System.Globalization;
using Markitai.Core;

namespace Markitai.Core.Tests;

public sealed class ResultTests
{
    static readonly Strings En = Strings.For("en-US", CultureInfo.GetCultureInfo("en-US"));
    static readonly Strings Zh = Strings.For("zh-Hans-CN", CultureInfo.GetCultureInfo("zh-CN"));
    static readonly Present EnText = new(En, t => t.ToString("MMM d, yyyy, HH:mm", CultureInfo.InvariantCulture), t => t.ToLocalTime().ToString("HH:mm", CultureInfo.InvariantCulture));

    const string Usage = """
        "usage":{"cost_usd":0.0412,"requests":6,"input_tokens":19870,"output_tokens":4440,
        "by_model":{"openai/a":{"cost_status":"complete"},"openai/b":{"cost_status":"complete"}}}
        """;

    static string Envelope(string fields) => "{\"ok\":true,\"result\":{" + fields + "}}";

    [Fact]
    public void The_llm_markdown_and_its_file_win_and_ai_usage_is_kept()
    {
        var done = Assert.IsType<Done>(CoreJson.Conversion(Envelope($$"""
            "markdown":"base","llm_markdown":"tidy","output_path":"C:/o/r.pdf.md","llm_output_path":"C:/o/r.pdf.llm.md",
            "frontmatter":{"title":"  Quarterly  "},"skip_reason":null,"duration":18.6,
            "warnings":["PDF page 7: one","PDF page 12: two"],{{Usage}}
            """), "C:/o"));
        Assert.Equal("C:/o", done.Folder);
        Assert.Equal("tidy", done.Result.Markdown);
        Assert.Equal("C:/o/r.pdf.llm.md", done.Result.MarkdownPath);
        Assert.Equal(["PDF page 7: one", "PDF page 12: two"], done.Result.Warnings);
        Assert.Equal(new Usage(6, 19870, 4440, 0.0412, CostStatus.Complete), done.Result.Usage);
        Assert.Equal("Quarterly", done.Result.PageTitle);
        Assert.Equal("Converted in 18.6 s · 24,310 tokens · $0.0412", EnText.Footer(done.Result));
        Assert.Equal("Done in 18.6 s · 2 warnings", EnText.Subtitle(done, "C:/o"));
    }

    [Fact]
    public void No_requests_means_no_usage_and_a_plain_footer()
    {
        var done = Assert.IsType<Done>(CoreJson.Conversion(Envelope("""
            "markdown":"x","output_path":"/o/x.md","skip_reason":null,"duration":75.4,"warnings":[],
            "usage":{"cost_usd":0,"requests":0,"input_tokens":0,"output_tokens":0,"by_model":{}}
            """), "/o"));
        Assert.Null(done.Result.Usage);
        Assert.Equal("Converted in 1 min 15 s", EnText.Footer(done.Result));
        Assert.Equal("Done in 1 min 15 s", EnText.Subtitle(done, "/o"));
    }

    [Theory]
    [InlineData("partial", 0.5, CostStatus.Partial, "at least $0.5000")]
    [InlineData("unknown", 0.0, CostStatus.Unknown, "cost unknown")]
    [InlineData("complete", 12.345, CostStatus.Complete, "$12.35")]
    public void Cost_follows_every_model_row(string second, double cost, CostStatus status, string shown)
    {
        var usage = "\"usage\":{\"cost_usd\":" + cost.ToString(CultureInfo.InvariantCulture)
            + ",\"requests\":2,\"input_tokens\":1000,\"output_tokens\":234,\"by_model\":{\"a\":{\"cost_status\":\"complete\"},\"b\":{\"cost_status\":\"" + second + "\"}}}";
        var done = Assert.IsType<Done>(CoreJson.Conversion(Envelope(
            "\"markdown\":\"x\",\"output_path\":\"/o/x.md\",\"skip_reason\":null,\"duration\":2,\"warnings\":[]," + usage), "/o"));
        Assert.Equal(status, done.Result.Usage!.Cost);
        Assert.EndsWith(" · 1,234 tokens · " + shown, EnText.Footer(done.Result));
    }

    [Fact]
    public void Skip_reasons_and_errors_become_failures()
    {
        Assert.Equal(new Failed("skipped", "exists"), CoreJson.Conversion(Envelope("\"markdown\":\"\",\"skip_reason\":\"exists\""), "/o"));
        Assert.Equal(new Failed("image_only", "x.png has no text"), CoreJson.Conversion("""{"ok":false,"error":{"code":"image_only","message":"x.png has no text"}}""", "/o"));
        Assert.Equal(new Failed("skipped", "empty"), CoreJson.Conversion(Envelope("\"markdown\":\"\",\"skip_reason\":\"empty\""), "/o"));
        Assert.Equal(new Failed("fetch_error", "HTTP 404"), CoreJson.Conversion("""{"ok":false,"error":{"code":"fetch_error","message":"HTTP 404"}}""", "/o"));
        Assert.Equal("invalid_json", Assert.IsType<Failed>(CoreJson.Conversion("not json", "/o")).Code);
        Assert.Equal("internal_error", Assert.IsType<Failed>(CoreJson.Conversion(Envelope("\"markdown\":\"x\""), "/o")).Code);
    }

    [Theory]
    [InlineData("not_found", "The file is no longer there. It may have been moved or deleted.", ErrorAction.Remove)]
    [InlineData("is_directory", "This is a folder. Add the files inside it instead.", ErrorAction.Remove)]
    [InlineData("no_model_configured", "Enhance with AI needs a model. Add one to your markitai configuration, or turn off Enhance with AI.", ErrorAction.OpenSettings)]
    [InlineData("invalid_input", "Markitai couldn't read this input.", ErrorAction.None)]
    [InlineData("unsupported", "Markitai can't convert this kind of file.", ErrorAction.Remove)]
    [InlineData("conversion_error", "The document couldn't be converted. Check that it opens in its own app.", ErrorAction.None)]
    [InlineData("fetch_error", "Couldn't download this page. Check the address and your internet connection.", ErrorAction.None)]
    [InlineData("config_error", "Your markitai configuration has a problem.", ErrorAction.OpenConfiguration)]
    [InlineData("io_error", "Markitai couldn't read or write a file. Check that the output folder exists, can be written and has free space.", ErrorAction.OpenSettings)]
    [InlineData("output_unavailable", @"Markitai couldn't create a folder in C:\Out. Choose another output folder in Settings.", ErrorAction.OpenSettings)]
    [InlineData("image_only", "This image has no text layer. Turn on OCR or Enhance with AI in Settings to read its text.", ErrorAction.OpenSettings)]
    [InlineData("skipped", "Markitai skipped this file (empty).", ErrorAction.None)]
    [InlineData("invalid_json", "Markitai hit an internal error. Retry, and report the details if it happens again.", ErrorAction.None)]
    [InlineData("internal_error", "Markitai hit an internal error. Retry, and report the details if it happens again.", ErrorAction.None)]
    [InlineData("something_new", "The conversion failed.", ErrorAction.None)]
    public void Every_error_row_of_section_9(string code, string sentence, ErrorAction action)
    {
        Assert.Equal(sentence, EnText.ErrorSentence(new Failed(code, "empty"), @"C:\Out"));
        Assert.Equal(action, Present.ActionFor(code));
        Assert.Equal("Failed · " + sentence, EnText.Subtitle(new Failed(code, "empty"), @"C:\Out"));
    }

    [Fact]
    public void A_web_page_takes_its_title_once_done()
    {
        var item = new ConversionItem(Source.Url("https://example.com/blog/markdown-for-language-models"), new(false, false, false, false, false, ConfigSource.User), DateTimeOffset.UtcNow, 0);
        Assert.Equal("example.com/blog/markdown-for-language-models", item.Title);
        var changed = new List<string?>();
        item.PropertyChanged += (_, e) => changed.Add(e.PropertyName);
        item.State = new Done("/o", new ConversionResult("x", "/o/x.md", 1, [], null, "Why Markdown Works"));
        Assert.Equal("Why Markdown Works", item.Title);
        Assert.Contains("Title", changed);

        var file = new ConversionItem(Source.File(@"C:\Users\example\Downloads\Quarterly Report Q3 2026.pdf"), item.Options, DateTimeOffset.UtcNow, 0);
        Assert.Equal("Quarterly Report Q3 2026.pdf", file.Title);
        file.State = new Done("/o", new ConversionResult("x", "/o/x.md", 1, [], null, "Ignored for files"));
        Assert.Equal("Quarterly Report Q3 2026.pdf", file.Title);
    }

    [Fact]
    public void A_long_page_title_is_cut_at_200_characters()
    {
        var long_ = new string('a', 199) + "😀tail";
        var done = Assert.IsType<Done>(CoreJson.Conversion(Envelope($$"""
            "markdown":"x","output_path":"/o/x.md","skip_reason":null,"duration":1,"warnings":[],"frontmatter":{"title":"{{long_}}"}
            """), "/o"));
        Assert.Equal(new string('a', 199), done.Result.PageTitle);
    }

    [Fact]
    public void Durations_dates_and_chinese_copy()
    {
        var zh = new Present(Zh, t => t.ToString("yyyy年M月d日 HH:mm", CultureInfo.InvariantCulture), t => t.ToLocalTime().ToString("HH:mm", CultureInfo.InvariantCulture));
        Assert.Equal("18.6 秒", zh.Duration(18.6));
        Assert.Equal("2 分 5 秒", zh.Duration(125.4));
        Assert.Equal("1 条警告", zh.Warnings(1));
        Assert.Equal("3 条警告", zh.Warnings(3));
        Assert.Equal("0:00", Present.Elapsed(TimeSpan.FromSeconds(-1)));
        Assert.Equal("12:07", Present.Elapsed(TimeSpan.FromSeconds(727.9)));

        var now = new DateTimeOffset(2026, 11, 4, 12, 0, 0, TimeSpan.Zero).ToLocalTime();
        var at = now.AddHours(-1);
        Assert.Equal($"Today, {at:HH:mm}", EnText.When(at, now));
        Assert.Equal($"Yesterday, {at:HH:mm}", EnText.When(at.AddDays(-1), now));
        Assert.StartsWith("今天 ", zh.When(at, now));
        Assert.Equal(at.AddDays(-7).ToString("MMM d, yyyy, HH:mm", CultureInfo.InvariantCulture), EnText.When(at.AddDays(-7), now));
    }
}
