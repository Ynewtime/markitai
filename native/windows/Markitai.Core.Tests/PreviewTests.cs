using Markitai.Core;

namespace Markitai.Core.Tests;

/// <summary>DESIGN.md 8: what the preview may load, follow and open.</summary>
public sealed class PreviewTests
{
    static readonly string PreviewFile = Path.Combine(Path.GetTempPath(), "Markitai", "Preview", "current.html");
    static readonly string PreviewUri = new Uri(PreviewFile).AbsoluteUri;

    [Fact]
    public void Only_the_preview_document_loads_and_web_links_open_outside()
    {
        Assert.Equal(LinkDecision.Allow, PreviewRules.Navigation(PreviewUri, PreviewFile));
        Assert.Equal(LinkDecision.Allow, PreviewRules.Navigation(PreviewUri + "#fn-1", PreviewFile));
        Assert.Equal(LinkDecision.OpenOutside, PreviewRules.Navigation("https://example.com/a", PreviewFile));
        Assert.Equal(LinkDecision.OpenOutside, PreviewRules.Navigation("http://example.com/a", PreviewFile));
        Assert.Equal(LinkDecision.OpenOutside, PreviewRules.Navigation("mailto:someone@example.com", PreviewFile));
        Assert.Equal(LinkDecision.Refuse, PreviewRules.Navigation(new Uri(Path.Combine(Path.GetTempPath(), "other.html")).AbsoluteUri, PreviewFile));
        Assert.Equal(LinkDecision.Refuse, PreviewRules.Navigation("javascript:alert(1)", PreviewFile));
        Assert.Equal(LinkDecision.Refuse, PreviewRules.Navigation("ms-settings:privacy", PreviewFile));
        Assert.Equal(LinkDecision.Refuse, PreviewRules.Navigation("relative/page.html", PreviewFile));
    }

    [Fact]
    public void Requests_other_than_local_files_are_blocked()
    {
        Assert.False(PreviewRules.Blocks(PreviewUri));
        Assert.False(PreviewRules.Blocks("FILE:///C:/x/.markitai/assets/a.png"));
        Assert.True(PreviewRules.Blocks("https://example.com/x.png"));
        Assert.True(PreviewRules.Blocks("data:image/png;base64,AAAA"));
    }

    [Fact]
    public void Zoom_is_written_into_the_page_head_with_an_invariant_number()
    {
        const string html = "<!doctype html><html><head><title>t</title></head><body></body></html>";
        Assert.Same(html, PreviewRules.WithZoom(html, 1.0));
        Assert.Equal("<!doctype html><html><head><title>t</title><style>html{zoom:1.25}</style>\n</head><body></body></html>",
            PreviewRules.WithZoom(html, 1.25));
        Assert.Equal("<p>no head</p>", PreviewRules.WithZoom("<p>no head</p>", 2));
    }
}
