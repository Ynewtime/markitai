using System.Collections.ObjectModel;
using System.Collections.Specialized;
using Markitai.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;

namespace MarkitaiApp.Views;

/// <summary>The History list column: search, rows, Clear History (native/DESIGN.md 6.5).</summary>
public sealed partial class HistoryList : UserControl
{
    readonly Workbench bench = App.Bench;
    readonly ObservableCollection<HistoryEntry> matches = [];
    string query = "";

    public HistoryList()
    {
        InitializeComponent();
        bench.History.Entries.CollectionChanged += Entries_CollectionChanged;
        ShowMatches();
    }

    /// <summary>The selection, and whether the user chose it (a click or Enter) rather than moved to it.</summary>
    public event Action<HistoryEntry?, bool>? Selected;

    public HistoryEntry? SelectedEntry => List.SelectedItem as HistoryEntry;

    IList<HistoryEntry> Shown => query.Length == 0 ? bench.History.Entries : matches;

    public void FocusSearch() => Search.Focus(FocusState.Programmatic);

    public void FocusList() => List.Focus(FocusState.Programmatic);

    /// <summary>Removes an entry (its files stay); a selected entry passes the selection to the next row, else the previous.</summary>
    public bool Remove(HistoryEntry entry)
    {
        var shown = Shown;
        var index = shown.IndexOf(entry);
        var wasSelected = SelectedEntry == entry;
        bench.RemoveFromHistory(entry);
        if (wasSelected && index >= 0 && shown.Count > 0)
            List.SelectedItem = shown[Math.Min(index, shown.Count - 1)];
        return true;
    }

    static bool Matches(HistoryEntry entry, string text) =>
        entry.Title.Contains(text, StringComparison.CurrentCultureIgnoreCase)
        || entry.Source.Contains(text, StringComparison.CurrentCultureIgnoreCase);

    /// <summary>Binds the list to History itself when there is no query (so rows animate in place), else to the matches.</summary>
    void ShowMatches()
    {
        var selected = SelectedEntry;
        if (query.Length > 0)
        {
            matches.Clear();
            foreach (var entry in bench.History.Entries)
                if (Matches(entry, query))
                    matches.Add(entry);
        }
        var source = Shown;
        if (!ReferenceEquals(List.ItemsSource, source))
            List.ItemsSource = source;
        if (selected is not null && source.Contains(selected))
            List.SelectedItem = selected;
        NoResults.Text = App.Strings.Format("history.no_results", ("query", query));
        NoResults.Visibility = Ui.Show(query.Length > 0 && matches.Count == 0 && bench.History.Entries.Count > 0);
        ClearHistory.IsEnabled = bench.History.Entries.Count > 0;
    }

    void Entries_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (query.Length == 0)
        {
            ClearHistory.IsEnabled = bench.History.Entries.Count > 0;
            return;
        }
        ShowMatches();
    }

    void Search_TextChanged(AutoSuggestBox sender, AutoSuggestBoxTextChangedEventArgs args)
    {
        query = sender.Text.Trim();
        ShowMatches();
    }

    async void ClearHistory_Click(object sender, RoutedEventArgs e)
    {
        var count = bench.History.Entries.Count;
        if (count == 0)
            return;
        var title = App.Strings.Plural("history.clear.title", count);
        if (await Ui.ConfirmAsync(XamlRoot, title, App.Strings["history.clear.body"], App.Strings["history.clear.confirm"]))
            bench.ClearHistory();
    }

    void List_ContainerContentChanging(ListViewBase sender, ContainerContentChangingEventArgs args)
    {
        if (args.Item is HistoryEntry entry)
            Ui.SetName(args.ItemContainer, $"{entry.Title}, {Ui.HistorySubtitle(entry.FinishedAt, entry.Warnings)}");
    }

    void List_SelectionChanged(object sender, SelectionChangedEventArgs e) => Selected?.Invoke(SelectedEntry, false);

    void List_ItemClick(object sender, ItemClickEventArgs e) => Selected?.Invoke(e.ClickedItem as HistoryEntry, true);

    void List_ContextRequested(UIElement sender, ContextRequestedEventArgs args)
    {
        if (Menus.Row(List, args) is not ({ } container, HistoryEntry entry))
            return;
        args.Handled = true;
        Menus.ShowAt(Menus.For(entry), container, args);
    }
}
