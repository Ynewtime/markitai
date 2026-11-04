using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace MarkitaiApp;

/// <summary>The only window when markitai_app.dll is missing or its ABI differs (native/DESIGN.md 3.1, 6.7).</summary>
public sealed class FatalWindow : Window
{
    public FatalWindow()
    {
        var s = App.Strings;
        Title = s["app.name"];
        var quit = new Button { Content = s["quit.confirm"], Style = (Style)Application.Current.Resources["AccentButtonStyle"], HorizontalAlignment = HorizontalAlignment.Right };
        quit.Click += (_, _) => Close();
        Content = new Grid
        {
            Background = (Microsoft.UI.Xaml.Media.Brush)Application.Current.Resources["SolidBackgroundFillColorBaseBrush"],
            Children =
            {
                new StackPanel
                {
                    Spacing = 12,
                    MaxWidth = 480,
                    Padding = new Thickness(24),
                    VerticalAlignment = VerticalAlignment.Center,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    Children =
                    {
                        new TextBlock { Text = s["fatal.abi.title"], Style = (Style)Application.Current.Resources["SubtitleTextBlockStyle"], TextWrapping = TextWrapping.Wrap },
                        new TextBlock { Text = s["fatal.abi.body"], TextWrapping = TextWrapping.Wrap },
                        quit,
                    },
                },
            },
        };
        AppWindow.Resize(new Windows.Graphics.SizeInt32(640, 360));
    }
}
