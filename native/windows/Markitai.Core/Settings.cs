using System.Text.Json;
using System.Text.Json.Serialization;

namespace Markitai.Core;

/// <summary>settings.json, one flat object (native/DESIGN.md 4.5). A missing or unreadable file means defaults.</summary>
public sealed record AppSettings
{
    [JsonPropertyName("conversion.llm")] public bool Llm { get; init; }
    [JsonPropertyName("conversion.ocr")] public bool Ocr { get; init; }
    [JsonPropertyName("conversion.screenshot")] public bool Screenshot { get; init; }
    [JsonPropertyName("conversion.desc")] public bool Desc { get; init; }
    [JsonPropertyName("conversion.alt")] public bool Alt { get; init; }
    [JsonPropertyName("output.folder")] public string? OutputFolder { get; init; }
    [JsonPropertyName("configuration.source")] public ConfigSource Config { get; init; }
    [JsonPropertyName("view.zoom")] public double Zoom { get; init; } = 1.0;
    [JsonPropertyName("window.width")] public double WindowWidth { get; init; } = 1200;
    [JsonPropertyName("window.height")] public double WindowHeight { get; init; } = 780;
    [JsonPropertyName("window.maximized")] public bool WindowMaximized { get; init; }

    public const double MinZoom = 0.5;
    public const double MaxZoom = 3.0;

    [JsonIgnore] public ConversionOptions Options => new(Llm, Ocr, Screenshot, Alt, Desc, Config);

    public static AppSettings Load(string path)
    {
        try
        {
            var read = JsonSerializer.Deserialize(File.ReadAllBytes(path), CoreJsonContext.Default.AppSettings) ?? new();
            return read with
            {
                Zoom = double.IsFinite(read.Zoom) ? Math.Clamp(read.Zoom, MinZoom, MaxZoom) : 1.0,
                OutputFolder = read.OutputFolder is { } folder && Path.IsPathFullyQualified(folder) ? folder : null,
                WindowWidth = Positive(read.WindowWidth, 1200),
                WindowHeight = Positive(read.WindowHeight, 780),
            };
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException)
        {
            return new();
        }
    }

    public void Save(string path) => AtomicFile.Write(path, JsonSerializer.SerializeToUtf8Bytes(this, CoreJsonContext.Default.AppSettings));

    static double Positive(double value, double fallback) => double.IsFinite(value) && value > 0 ? value : fallback;
}

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UseStringEnumConverter = true,
    WriteIndented = true,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(HistoryFile))]
[JsonSerializable(typeof(AppSettings))]
partial class CoreJsonContext : JsonSerializerContext;
