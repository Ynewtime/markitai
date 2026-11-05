using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Foundation;

namespace MarkitaiApp;

/// <summary>
/// A list row's title on one line (desktop/DESIGN.md 4.1): a file name that does not fit keeps
/// its start and its extension with "…" between them; a URL keeps WinUI's tail trimming. WinUI
/// trims only the end, so the title is laid out as a panel: each measure is given the width the
/// row has, and the TextBlock gets the longest middle-trimmed text that fits it.
/// </summary>
public sealed partial class RowTitle : Panel
{
    public static readonly DependencyProperty TextProperty = DependencyProperty.Register(
        nameof(Text), typeof(string), typeof(RowTitle), new PropertyMetadata("", Changed));

    public static readonly DependencyProperty MiddleProperty = DependencyProperty.Register(
        nameof(Middle), typeof(bool), typeof(RowTitle), new PropertyMetadata(true, Changed));

    readonly TextBlock shown = new() { TextWrapping = TextWrapping.NoWrap, TextTrimming = TextTrimming.CharacterEllipsis };

    // Measures candidates outside the tree, so trying them never invalidates the row's layout.
    readonly TextBlock probe = new() { TextWrapping = TextWrapping.NoWrap };

    public RowTitle() => Children.Add(shown);

    public string Text
    {
        get => (string)GetValue(TextProperty);
        set => SetValue(TextProperty, value);
    }

    public bool Middle
    {
        get => (bool)GetValue(MiddleProperty);
        set => SetValue(MiddleProperty, value);
    }

    static void Changed(DependencyObject d, DependencyPropertyChangedEventArgs e) => ((RowTitle)d).InvalidateMeasure();

    protected override Size MeasureOverride(Size availableSize)
    {
        var text = Text ?? "";
        var fitted = Middle ? Trimmed(text, availableSize.Width) : text;
        if (shown.Text != fitted)
            shown.Text = fitted;
        shown.Measure(availableSize);
        return shown.DesiredSize;
    }

    protected override Size ArrangeOverride(Size finalSize)
    {
        shown.Arrange(new Rect(0, 0, finalSize.Width, finalSize.Height));
        return finalSize;
    }

    /// <summary>The longest text made of the start and end of <paramref name="text"/> around "…" that fits <paramref name="width"/>.</summary>
    string Trimmed(string text, double width)
    {
        if (double.IsInfinity(width))
            return text;
        probe.FontFamily = shown.FontFamily;
        probe.FontSize = shown.FontSize;
        probe.FontWeight = shown.FontWeight;
        probe.FontStyle = shown.FontStyle;
        probe.FontStretch = shown.FontStretch;
        probe.CharacterSpacing = shown.CharacterSpacing;
        bool Fits(string candidate)
        {
            probe.Text = candidate;
            probe.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            return probe.DesiredSize.Width <= width;
        }
        if (Fits(text))
            return text;
        var elements = new List<string>();
        var walk = StringInfo.GetTextElementEnumerator(text);
        while (walk.MoveNext())
            elements.Add(walk.GetTextElement());
        string Keep(int count)
        {
            var head = (count + 1) / 2;
            return string.Concat(elements.Take(head)) + "…" + string.Concat(elements.Skip(elements.Count - (count - head)));
        }
        int low = 0, high = elements.Count - 1;
        while (low < high)
        {
            var middle = (low + high + 1) / 2;
            if (Fits(Keep(middle)))
                low = middle;
            else
                high = middle - 1;
        }
        return Keep(low);
    }
}
