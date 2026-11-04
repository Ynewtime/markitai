using System.Text.Json;
using System.Text.Json.Nodes;

namespace Markitai.Core;

/// <summary>The core's answers, parsed into domain values at the boundary.</summary>
public static class CoreJson
{
    public readonly record struct Answer(JsonElement Result, string? Code, string Message)
    {
        public bool Ok => Code is null;
    }

    public static Answer Parse(string json)
    {
        try
        {
            var root = JsonSerializer.Deserialize(json, CoreJsonContext.Default.JsonElement);
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("ok", out var ok))
            {
                if (ok.ValueKind == JsonValueKind.True && root.TryGetProperty("result", out var result))
                    return new(result, null, "");
                if (ok.ValueKind == JsonValueKind.False && root.TryGetProperty("error", out var error))
                    return new(default, Text(error, "code") ?? "internal_error", Text(error, "message") ?? "");
            }
        }
        catch (JsonException e)
        {
            return new(default, "invalid_json", e.Message);
        }
        return new(default, "invalid_json", "The core answered without an envelope.");
    }

    public static string? Text(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    static double Number(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble()
            : 0;

    public static string PrepareRequest(string dataDir) => new JsonObject { ["data_dir"] = dataDir }.ToJsonString();

    public static string ClaimRequest(string parent, Source source) =>
        new JsonObject { ["parent"] = parent, ["source"] = source.Value }.ToJsonString();

    /// <summary>The convert_json request (desktop/DESIGN.md 4.1): alt and desc only with llm, config {} for built-in defaults.</summary>
    public static string ConvertRequest(Source source, string folder, ConversionOptions options)
    {
        var o = new JsonObject
        {
            ["output_dir"] = folder,
            ["llm"] = options.Llm,
            ["ocr"] = options.Ocr,
            ["screenshot"] = options.Screenshot,
        };
        if (options.Llm)
        {
            o["alt"] = options.Alt;
            o["desc"] = options.Desc;
        }
        if (options.Config == ConfigSource.Builtin)
            o["config"] = new JsonObject();
        return new JsonObject { ["source"] = source.Value, ["options"] = o }.ToJsonString();
    }

    public static string PreviewRequest(string markdown, bool dark, string? baseDir) =>
        new JsonObject { ["markdown"] = markdown, ["theme"] = dark ? "dark" : "light", ["base_dir"] = baseDir }.ToJsonString();

    /// <summary>A convert_json envelope as the item's final state (desktop/DESIGN.md 4.1, step 4).</summary>
    public static ItemState Conversion(string json, string folder)
    {
        var answer = Parse(json);
        if (!answer.Ok)
            return new Failed(answer.Code!, answer.Message);
        var result = answer.Result;
        if (Text(result, "skip_reason") is { } reason)
            return new Failed("skipped", reason);
        var markdown = Text(result, "llm_markdown") ?? Text(result, "markdown");
        var path = Text(result, "llm_output_path") ?? Text(result, "output_path");
        if (markdown is null || path is null)
            return new Failed("internal_error", "The conversion answered without Markdown or an output path.");
        var warnings = result.TryGetProperty("warnings", out var w) && w.ValueKind == JsonValueKind.Array
            ? w.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).ToArray()
            : [];
        return new Done(folder, new ConversionResult(
            markdown, path, Number(result, "duration"), warnings,
            result.TryGetProperty("usage", out var usage) ? UsageOf(usage) : null,
            PageTitle(result)));
    }

    static Usage? UsageOf(JsonElement usage)
    {
        var requests = (long)Number(usage, "requests");
        if (requests <= 0)
            return null;
        var cost = Number(usage, "cost_usd");
        var rows = usage.TryGetProperty("by_model", out var byModel) && byModel.ValueKind == JsonValueKind.Object
            ? byModel.EnumerateObject().Select(row => Text(row.Value, "cost_status")).ToArray()
            : [];
        var status = rows.All(s => s == "complete") ? CostStatus.Complete
            : cost > 0 ? CostStatus.Partial
            : CostStatus.Unknown;
        return new Usage(requests, (long)Number(usage, "input_tokens"), (long)Number(usage, "output_tokens"), cost, status);
    }

    static string? PageTitle(JsonElement result)
    {
        if (!result.TryGetProperty("frontmatter", out var frontmatter) || Text(frontmatter, "title")?.Trim() is not { Length: > 0 } title)
            return null;
        if (title.Length <= 200)
            return title;
        return title[..(char.IsHighSurrogate(title[199]) ? 199 : 200)];
    }
}
