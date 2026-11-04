namespace Markitai.Core;

/// <summary>What the user hands the app, turned into sources before any core call (desktop/DESIGN.md 7.1).</summary>
public static class Intake
{
    public sealed record Sorted(IReadOnlyList<Source> Sources, IReadOnlyList<string> RefusedFolders);

    /// <summary>
    /// Trims, prefixes https:// when there is no scheme and the text has a dot, and accepts only
    /// http and https URLs with a host. Null means the text is not a web address.
    /// </summary>
    public static string? WebAddress(string? text)
    {
        var trimmed = text?.Trim() ?? "";
        if (trimmed.Length == 0 || trimmed.Any(char.IsWhiteSpace))
            return null;
        if (!HasScheme(trimmed) && trimmed.Contains('.'))
            trimmed = "https://" + trimmed;
        return Uri.TryCreate(trimmed, UriKind.Absolute, out var uri)
               && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps)
               && uri.Host.Length > 0
            ? trimmed
            : null;
    }

    /// <summary>"mailto:x" and "ftp://x" have a scheme; "example.com:8080/x" is a host and port.</summary>
    static bool HasScheme(string text)
    {
        var colon = text.IndexOf(':');
        if (colon <= 0 || !char.IsAsciiLetter(text[0]) || !text[..colon].All(c => char.IsAsciiLetterOrDigit(c) || c is '+' or '-' or '.'))
            return false;
        var rest = text[(colon + 1)..];
        var port = rest.IndexOfAny(['/', '?', '#']) is var end and >= 0 ? rest[..end] : rest;
        return port.Length == 0 || !port.All(char.IsAsciiDigit);
    }

    /// <summary>
    /// Files become sources with absolute paths; folders are refused, except a directory whose
    /// name ends in .numbers (a Numbers document). Paths that do not exist pass through: the core reports them.
    /// </summary>
    public static Sorted Paths(IEnumerable<string> paths, string baseDirectory)
    {
        var sources = new List<Source>();
        var refused = new List<string>();
        foreach (var raw in paths)
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;
            var path = Path.GetFullPath(raw, baseDirectory);
            var trimmed = Path.TrimEndingDirectorySeparator(path);
            if (Directory.Exists(path) && !trimmed.EndsWith(".numbers", StringComparison.OrdinalIgnoreCase))
                refused.Add(Path.GetFileName(trimmed) is { Length: > 0 } name ? name : trimmed);
            else
                sources.Add(Source.File(trimmed));
        }
        return new(sources, refused);
    }
}

/// <summary>Files the app writes for the user (desktop/DESIGN.md 6.6, 7.2).</summary>
public static class UserFiles
{
    /// <summary>
    /// Writes the shown Markdown to <paramref name="target"/>. When it refers to .markitai/, copies the
    /// conversion's .markitai folder next to the saved file, adding missing files and never replacing one.
    /// </summary>
    public static void SaveMarkdown(string target, string markdown, string conversionFolder)
    {
        File.WriteAllText(target, markdown);
        if (!markdown.Contains(".markitai/", StringComparison.Ordinal))
            return;
        var from = Path.Combine(conversionFolder, ".markitai");
        if (!Directory.Exists(from))
            return;
        var to = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(target))!, ".markitai");
        foreach (var file in Directory.EnumerateFiles(from, "*", SearchOption.AllDirectories))
        {
            var destination = Path.Combine(to, Path.GetRelativePath(from, file));
            if (File.Exists(destination))
                continue;
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            File.Copy(file, destination, overwrite: false);
        }
    }

    /// <summary>Whether a test file can be created in <paramref name="folder"/> and deleted again.</summary>
    public static bool CanWrite(string folder)
    {
        try
        {
            Directory.CreateDirectory(folder);
            var probe = Path.Combine(folder, $".markitai-write-test-{Guid.NewGuid():N}");
            File.WriteAllBytes(probe, []);
            File.Delete(probe);
            return true;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            return false;
        }
    }

    /// <summary>Creates the configuration file with "{}" and a newline, unless it already exists.</summary>
    public static void CreateConfiguration(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write);
        stream.Write("{}\n"u8);
    }
}
