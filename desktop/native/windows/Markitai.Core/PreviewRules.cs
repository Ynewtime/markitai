using System.Globalization;
using System.Text.Json;

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

/// <summary>
/// What the preview's host script posts (desktop/DESIGN.md 8): an app shortcut pressed in the
/// page, the mouse's back button, or Ctrl with the wheel.
/// </summary>
public abstract record PreviewMessage
{
    public sealed record Key(int Code, bool Ctrl, bool Shift, bool Alt) : PreviewMessage;

    public sealed record Back : PreviewMessage;

    /// <summary>The wheel event's wheelDeltaY: 120 a notch, positive away from the user.</summary>
    public sealed record Wheel(double Delta) : PreviewMessage;

    /// <summary>The message, or null for anything the script does not post.</summary>
    public static PreviewMessage? Parse(string json)
    {
        try
        {
            using var message = JsonDocument.Parse(json);
            var root = message.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return null;
            if (root.TryGetProperty("key", out var key))
                return new Key(key.GetInt32(), root.GetProperty("ctrl").GetBoolean(), root.GetProperty("shift").GetBoolean(), root.GetProperty("alt").GetBoolean());
            if (root.TryGetProperty("back", out var back) && back.GetBoolean())
                return new Back();
            if (root.TryGetProperty("wheel", out var wheel))
                return new Wheel(wheel.GetDouble());
            return null;
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException or FormatException)
        {
            return null;
        }
    }
}

/// <summary>
/// Ctrl with the wheel over the preview, in steps of view.zoom: one a notch, at any zoom. The
/// page's wheelDeltaY is 120 a notch (the preview's zoom is CSS zoom, so the page's own zoom
/// stays 1); a precision touchpad gives parts of one, which add up. Turning the wheel the other
/// way starts the count again.
/// </summary>
public sealed class WheelSteps
{
    double notches;

    /// <summary>The steps this delta completes: positive to zoom in, negative to zoom out.</summary>
    public int Add(double delta)
    {
        var turn = delta / 120;
        if (turn * notches < 0)
            notches = 0;
        notches += turn;
        var steps = (int)Math.Truncate(notches);
        notches -= steps;
        return steps;
    }
}
