using Markitai.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;

namespace MarkitaiApp;

/// <summary>Segoe Fluent Icons glyphs (native/DESIGN.md 13.6) and the x:Bind helpers the list rows use.</summary>
public static class Ui
{
    public const string Convert = "";
    public const string History = "";
    public const string Settings = "";
    public const string AddFiles = "";
    public const string Link = "";
    public const string FileKind = "";
    public const string WebKind = "";
    public const string Queued = "";
    public const string DoneGlyph = "";
    public const string Warning = "";
    public const string FailedGlyph = "";
    public const string Copy = "";
    public const string SaveAs = "";
    public const string ShowGlyph = "";
    public const string Retry = "";
    public const string Clear = "";
    public const string Delete = "";
    public const string Accept = "";

    public static string KindGlyph(Source source) => KindGlyphOf(source.Kind);

    public static string KindGlyphOf(SourceKind kind) => kind == SourceKind.Url ? WebKind : FileKind;

    public static string Subtitle(ItemState state) => App.Bench.Text.Subtitle(state, App.Bench.OutputFolder);

    public static string RowName(ConversionItem item) => $"{item.Title}, {Subtitle(item.State)}";

    public static string HistorySubtitle(DateTimeOffset finishedAt, IReadOnlyList<string> warnings) =>
        App.Bench.Text.HistorySubtitle(finishedAt, warnings.Count, DateTimeOffset.Now);

    public static Visibility WarningsVisible(IReadOnlyList<string> warnings) => Show(warnings.Count > 0);

    public static Visibility QueuedVisible(ItemState state) => Show(state is Queued);

    public static Visibility SpinnerVisible(ItemState state) => Show(state is Converting);

    public static bool SpinnerActive(ItemState state) => state is Converting;

    public static Visibility WarningVisible(ItemState state) => Show(state is Done { Result.Warnings.Count: > 0 });

    public static Visibility FailedVisible(ItemState state) => Show(state is Failed);

    public static Visibility Show(bool visible) => visible ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>A polite announcement for screen readers (native/DESIGN.md 12).</summary>
    public static void Announce(UIElement element, string text)
    {
        var peer = FrameworkElementAutomationPeer.FromElement(element) ?? FrameworkElementAutomationPeer.CreatePeerForElement(element);
        peer?.RaiseNotificationEvent(AutomationNotificationKind.ActionCompleted, AutomationNotificationProcessing.ImportantAll, text, "markitai-status");
    }

    /// <summary>Whether keyboard focus is in a text field, where Ctrl+V and Delete edit text.</summary>
    public static bool FocusInText(XamlRoot root) =>
        Microsoft.UI.Xaml.Input.FocusManager.GetFocusedElement(root) is TextBox or AutoSuggestBox or PasswordBox or RichEditBox;

    /// <summary>A confirmation whose default button is Cancel (native/DESIGN.md 6.7). True when the user confirms.</summary>
    public static async Task<bool> ConfirmAsync(XamlRoot root, string title, string body, string confirm) =>
        await Dialogs.ShowAsync(Dialogs.Confirmation(root, title, body, confirm)) == ContentDialogResult.Primary;

    public static void SetName(DependencyObject element, string name) => AutomationProperties.SetName(element, name);
}

/// <summary>
/// The window's one ContentDialog: WinUI throws when a second opens, so a dialog asked
/// for while another shows is not shown, and the quit dialog first closes the other.
/// </summary>
public static class Dialogs
{
    static ContentDialog? open;
    static Task? closed;

    public static ContentDialog Confirmation(XamlRoot root, string title, string body, string confirm) => new()
    {
        Title = title,
        Content = body,
        PrimaryButtonText = confirm,
        CloseButtonText = App.Strings["common.cancel"],
        DefaultButton = ContentDialogButton.Close,
        XamlRoot = root,
    };

    /// <summary>The user's answer; null when another dialog is open.</summary>
    public static async Task<ContentDialogResult?> ShowAsync(ContentDialog dialog)
    {
        if (open is not null)
            return null;
        var done = new TaskCompletionSource();
        (open, closed) = (dialog, done.Task);
        try
        {
            return await dialog.ShowAsync();
        }
        finally
        {
            (open, closed) = (null, null);
            done.SetResult();
        }
    }

    /// <summary>Closes the open dialog as Cancel and waits until it is gone.</summary>
    public static async Task CloseAsync()
    {
        if (open is null || closed is null)
            return;
        var gone = closed;
        open.Hide();
        await gone;
    }
}
