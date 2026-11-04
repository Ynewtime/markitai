using System.Globalization;

namespace Markitai.Core;

/// <summary>What a navigation inside the preview may do (desktop/DESIGN.md 8).</summary>
public enum LinkDecision
{
    /// <summary>The preview document itself, including its own fragment links.</summary>
    Allow,

    /// <summary>An http, https or mailto link: cancel, then open it in its default app.</summary>
    OpenOutside,

    /// <summary>Anything else: cancel.</summary>
    Refuse,
}

/// <summary>The preview web view's rules that do not need a web view.</summary>
public static class PreviewRules
{
    public static LinkDecision Navigation(string uri, string previewFile)
    {
        if (!Uri.TryCreate(uri, UriKind.Absolute, out var parsed))
            return LinkDecision.Refuse;
        if (parsed.IsFile && string.Equals(parsed.LocalPath, previewFile, StringComparison.OrdinalIgnoreCase))
            return LinkDecision.Allow;
        return parsed.Scheme is "http" or "https" or "mailto" ? LinkDecision.OpenOutside : LinkDecision.Refuse;
    }

    /// <summary>Every request but file: is answered with 403.</summary>
    public static bool Blocks(string uri) => !uri.StartsWith("file:", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// view.zoom for the preview. WinUI's WebView2 control has no ZoomFactor, so the page
    /// carries it as CSS zoom (the preview's CSP allows inline styles).
    /// </summary>
    public static string WithZoom(string html, double zoom)
    {
        if (Math.Abs(zoom - 1.0) < 0.001)
            return html;
        var at = html.IndexOf("</head>", StringComparison.Ordinal);
        if (at < 0)
            return html;
        return html.Insert(at, $"<style>html{{zoom:{zoom.ToString("0.##", CultureInfo.InvariantCulture)}}}</style>\n");
    }
}
