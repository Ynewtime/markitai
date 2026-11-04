using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.System;

namespace MarkitaiApp.Views;

/// <summary>The web address field and its Add button (desktop/DESIGN.md 6.2, 6.3, 7.1).</summary>
public sealed partial class UrlBox : UserControl
{
    public UrlBox()
    {
        InitializeComponent();
    }

    /// <summary>The leading link glyph of the list column's field.</summary>
    public bool ShowLinkGlyph
    {
        get => LinkGlyph.Visibility == Visibility.Visible;
        set
        {
            LinkGlyph.Visibility = Ui.Show(value);
            Field.Padding = value ? new Thickness(38, 5, 6, 6) : new Thickness(10, 5, 6, 6);
        }
    }

    public void FocusField() => Field.Focus(FocusState.Programmatic);

    void Submit()
    {
        if (string.IsNullOrWhiteSpace(Field.Text))
            return;
        if (App.Bench.AddUrl(Field.Text))
            Field.Text = "";
        else
            Error.Visibility = Visibility.Visible;
    }

    void Add_Click(object sender, RoutedEventArgs e) => Submit();

    void Field_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key != VirtualKey.Enter)
            return;
        e.Handled = true;
        Submit();
    }

    void Field_TextChanged(object sender, TextChangedEventArgs e) => Error.Visibility = Visibility.Collapsed;
}
