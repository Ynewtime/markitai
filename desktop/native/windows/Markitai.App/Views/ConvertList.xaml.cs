using Markitai.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;

namespace MarkitaiApp.Views;

/// <summary>The Convert list column (desktop/DESIGN.md 6.3).</summary>
public sealed partial class ConvertList : UserControl
{
    readonly Workbench bench = App.Bench;

    public ConvertList()
    {
        InitializeComponent();
        Url.ShowLinkGlyph = true;
        List.ItemsSource = bench.Queue.Items;
        if (!new Windows.UI.ViewManagement.UISettings().AnimationsEnabled)
            List.ItemContainerTransitions.Clear();
        bench.Queue.Items.CollectionChanged += (_, _) => Refresh();
        bench.Queue.StateChanged += Refresh;
        bench.SettingsChanged += Refresh;
        bench.NoticeChanged += () => Notices.Show(NoticeBar, bench.Notice);
        Notices.Show(NoticeBar, bench.Notice);
        Refresh();
    }

    /// <summary>The selection, and whether the user chose it (a click or Enter) rather than moved to it.</summary>
    public event Action<ConversionItem?, bool>? Selected;

    public ConversionItem? SelectedItem => List.SelectedItem as ConversionItem;

    public void Select(ConversionItem item)
    {
        List.SelectedItem = item;
        List.ScrollIntoView(item);
    }

    public void FocusList() => List.Focus(FocusState.Programmatic);

    public void FocusUrl() => Url.FocusField();

    /// <summary>Removes an item that is not converting; a selected item passes the selection to the next row, else the previous.</summary>
    public bool Remove(ConversionItem item)
    {
        var items = bench.Queue.Items;
        var index = items.IndexOf(item);
        var wasSelected = SelectedItem == item;
        if (!bench.Queue.Remove(item))
            return false;
        if (wasSelected && items.Count > 0)
            Select(items[Math.Min(index, items.Count - 1)]);
        return true;
    }

    public void SetDragging(bool dragging)
    {
        Bar.Visibility = Ui.Show(!dragging);
        BarActive.Visibility = Ui.Show(dragging);
        BarLabel.Text = App.Strings[dragging ? "convert.drop.release" : "convert.drop.bar"];
    }

    void Refresh()
    {
        ClearFinished.IsEnabled = bench.Queue.HasFinished;
        foreach (var item in bench.Queue.Items)
            if (List.ContainerFromItem(item) is ListViewItem container)
                Ui.SetName(container, Ui.RowName(item));
    }

    void List_ContainerContentChanging(ListViewBase sender, ContainerContentChangingEventArgs args)
    {
        if (args.Item is ConversionItem item)
            Ui.SetName(args.ItemContainer, Ui.RowName(item));
    }

    void List_SelectionChanged(object sender, SelectionChangedEventArgs e) => Selected?.Invoke(SelectedItem, false);

    void List_ItemClick(object sender, ItemClickEventArgs e) => Selected?.Invoke(e.ClickedItem as ConversionItem, true);

    void List_ContextRequested(UIElement sender, ContextRequestedEventArgs args)
    {
        if (Menus.Row(List, args) is not ({ } container, ConversionItem item))
            return;
        args.Handled = true;
        Menus.ShowAt(Menus.For(item), container, args);
    }

    void ClearFinished_Click(object sender, RoutedEventArgs e) => bench.Queue.ClearFinished();

    async void Choose_Click(object sender, RoutedEventArgs e) => await MainWindow.Current.ChooseFilesAsync();

    void NoticeBar_Closed(InfoBar sender, InfoBarClosedEventArgs args) => bench.SetNotice(null);
}

/// <summary>Row context menus (desktop/DESIGN.md 6.3, 6.5).</summary>
static class Menus
{
    public static (ListViewItem? Container, object? Item) Row(ListView list, ContextRequestedEventArgs args)
    {
        for (var at = args.OriginalSource as DependencyObject; at is not null; at = VisualTreeHelper.GetParent(at))
            if (at is ListViewItem container)
                return (container, list.ItemFromContainer(container));
        return (null, null);
    }

    public static void ShowAt(MenuFlyout menu, ListViewItem container, ContextRequestedEventArgs args)
    {
        if (args.TryGetPosition(container, out var point))
            menu.ShowAt(container, new Microsoft.UI.Xaml.Controls.Primitives.FlyoutShowOptions { Position = point });
        else
            menu.ShowAt(container);
    }

    public static MenuFlyout For(ConversionItem item)
    {
        var menu = new MenuFlyout();
        var window = MainWindow.Current;
        switch (item.State)
        {
            case Done done:
                AddFinished(menu, Finished.Of(item, done));
                menu.Items.Add(new MenuFlyoutSeparator());
                break;
            case Failed:
                menu.Items.Add(Entry("detail.retry", Ui.Retry, () => window.Retry(item)));
                menu.Items.Add(new MenuFlyoutSeparator());
                break;
        }
        var remove = Entry("convert.remove", Ui.Delete, () => window.Remove(item));
        menu.Items.Add(remove);
        if (item.State is Converting)
        {
            // WinUI shows no tooltip on a disabled item, so the reason is also a line of its own.
            var busy = App.Strings["convert.remove.busy"];
            remove.IsEnabled = false;
            ToolTipService.SetToolTip(remove, busy);
            AutomationProperties.SetHelpText(remove, busy);
            menu.Items.Add(new MenuFlyoutItem { Text = busy, IsEnabled = false });
        }
        return menu;
    }

    public static MenuFlyout For(HistoryEntry entry)
    {
        var menu = new MenuFlyout();
        var window = MainWindow.Current;
        if (File.Exists(entry.MarkdownPath))
        {
            AddFinished(menu, Finished.Of(entry));
            menu.Items.Add(new MenuFlyoutSeparator());
        }
        menu.Items.Add(Entry("history.remove", Ui.Delete, () => window.Remove(entry)));
        return menu;
    }

    static void AddFinished(MenuFlyout menu, Finished finished)
    {
        var window = MainWindow.Current;
        menu.Items.Add(Entry("detail.copy", Ui.Copy, async () => await window.CopyAsync(finished)));
        menu.Items.Add(Entry("detail.save_as", Ui.SaveAs, async () => await window.SaveAsAsync(finished)));
        menu.Items.Add(Entry("detail.show.windows", Ui.ShowGlyph, () => window.ShowInExplorer(finished.MarkdownPath)));
    }

    static MenuFlyoutItem Entry(string key, string glyph, Action action)
    {
        var entry = new MenuFlyoutItem { Text = App.Strings[key], Icon = new FontIcon { Glyph = glyph } };
        entry.Click += (_, _) => action();
        return entry;
    }
}
