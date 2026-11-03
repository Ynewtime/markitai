using System.Globalization;
using System.Text.Json;

namespace Markitai.Core;

/// <summary>native/strings.json, embedded, in the one language the app chose at launch (native/DESIGN.md 7.5).</summary>
public sealed class Strings
{
    static readonly Lazy<Dictionary<string, Dictionary<string, string>>> Table = new(() =>
    {
        using var stream = typeof(Strings).Assembly.GetManifestResourceStream("strings.json")!;
        return JsonSerializer.Deserialize(stream, StringsJsonContext.Default.DictionaryStringDictionaryStringString)!;
    });

    Strings(string language, CultureInfo culture)
    {
        Language = language;
        Culture = culture;
    }

    /// <summary>"en" or "zh-Hans".</summary>
    public string Language { get; }

    /// <summary>Formats numbers and times.</summary>
    public CultureInfo Culture { get; }

    public static IReadOnlyCollection<string> Keys => Table.Value.Keys;

    /// <summary>zh-Hans when the first preferred UI language starts with "zh", else English.</summary>
    public static Strings For(string? firstPreferredLanguage, CultureInfo culture) =>
        new(firstPreferredLanguage?.StartsWith("zh", StringComparison.OrdinalIgnoreCase) == true ? "zh-Hans" : "en", culture);

    public string this[string key] => Table.Value[key][Language];

    public string Format(string key, params (string Name, string Value)[] args)
    {
        var text = this[key];
        foreach (var (name, value) in args)
            text = text.Replace("{" + name + "}", value, StringComparison.Ordinal);
        return text;
    }

    /// <summary><paramref name="key"/>.one when count is 1, else .other, with {count} filled in.</summary>
    public string Plural(string key, long count, params (string Name, string Value)[] args) =>
        Format(count == 1 ? key + ".one" : key + ".other", [("count", count.ToString("N0", Culture)), .. args]);
}

[System.Text.Json.Serialization.JsonSerializable(typeof(Dictionary<string, Dictionary<string, string>>))]
partial class StringsJsonContext : System.Text.Json.Serialization.JsonSerializerContext;
