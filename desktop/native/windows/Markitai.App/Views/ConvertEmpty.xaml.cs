using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace MarkitaiApp.Views;

/// <summary>Convert with no items: one large drop zone (desktop/DESIGN.md 6.2).</summary>
public sealed partial class ConvertEmpty : UserControl
{
    public ConvertEmpty()
    {
        InitializeComponent();
        var bench = App.Bench;
        ShowFolder();
        bench.SettingsChanged += ShowFolder;
        bench.NoticeChanged += () => Notices.Show(NoticeBar, bench.Notice);
        Notices.Show(NoticeBar, bench.Notice);
    }

    void ShowFolder() => Body.Text = App.Strings.Format("convert.empty.body", ("folder", App.Bench.OutputFolder));

    public void FocusChoose() => Choose.Focus(FocusState.Programmatic);

    public void FocusUrl() => Url.FocusField();

    public void SetDragging(bool dragging)
    {
        Zone.Visibility = Ui.Show(!dragging);
        ZoneActive.Visibility = Ui.Show(dragging);
        Heading.Text = App.Strings[dragging ? "convert.drop.release" : "convert.empty.title"];
    }

    async void Choose_Click(object sender, RoutedEventArgs e) => await MainWindow.Current.ChooseFilesAsync();

    void NoticeBar_Closed(InfoBar sender, InfoBarClosedEventArgs args) => App.Bench.SetNotice(null);
}

static class Notices
{
    public static void Show(InfoBar bar, Notice? notice)
    {
        bar.IsOpen = notice is not null;
        if (notice is null)
            return;
        bar.Message = notice.Text;
        bar.Severity = notice.Severity;
    }
}
