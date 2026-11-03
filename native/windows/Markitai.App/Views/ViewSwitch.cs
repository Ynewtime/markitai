using CommunityToolkit.WinUI.Controls;
using Microsoft.UI.Xaml.Automation.Peers;

namespace MarkitaiApp.Views;

/// <summary>
/// The detail's Preview/Markdown switch: the toolkit's Segmented with a selection peer.
/// Segmented alone shows UI Automation a group of list items with no SelectionItem
/// pattern, so Narrator cannot say which view is on and UI Automation cannot pick one.
/// The list peer gives the items SelectionItem and the switch Selection; the role stays
/// the group native/DESIGN.md 12 names.
/// </summary>
public sealed partial class ViewSwitch : Segmented
{
    protected override AutomationPeer OnCreateAutomationPeer() => new Peer(this);

    sealed partial class Peer(ViewSwitch owner) : ListViewBaseAutomationPeer(owner)
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Group;

        protected override string GetClassNameCore() => nameof(ViewSwitch);
    }
}
