using CommunityToolkit.WinUI.Controls;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Controls;

namespace MarkitaiApp.Views;

/// <summary>
/// The detail's Preview/Markdown switch: the toolkit's Segmented with its own peers.
/// UI Automation sees a group named detail.view with Selection, and one list item per
/// view, named by its text, with SelectionItem, so Narrator says which view is on and
/// UI Automation can pick one (desktop/DESIGN.md 12). The list peers Segmented could
/// inherit either drop SelectionItem or name the items after their type.
/// </summary>
public sealed partial class ViewSwitch : Segmented
{
    public ViewSwitch()
    {
        SelectionChanged += (_, _) =>
        {
            if (ContainerFromIndex(SelectedIndex) is ViewSegment segment && FrameworkElementAutomationPeer.FromElement(segment) is { } peer)
                peer.RaiseAutomationEvent(AutomationEvents.SelectionItemPatternOnElementSelected);
        };
    }

    protected override AutomationPeer OnCreateAutomationPeer() => new Peer(this);

    sealed partial class Peer(ViewSwitch owner) : FrameworkElementAutomationPeer(owner), ISelectionProvider
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Group;

        protected override string GetClassNameCore() => nameof(ViewSwitch);

        protected override object GetPatternCore(PatternInterface patternInterface) =>
            patternInterface == PatternInterface.Selection ? this : base.GetPatternCore(patternInterface);

        public bool CanSelectMultiple => false;

        public bool IsSelectionRequired => true;

        public IRawElementProviderSimple[] GetSelection() =>
            owner.ContainerFromIndex(owner.SelectedIndex) is ViewSegment segment && FromElement(segment) is { } peer
                ? [ProviderFromPeer(peer)]
                : [];
    }
}

/// <summary>One view of the switch; its peer is a list item named by its text, with SelectionItem.</summary>
public sealed partial class ViewSegment : SegmentedItem
{
    protected override AutomationPeer OnCreateAutomationPeer() => new Peer(this);

    ViewSwitch? Switch => ItemsControl.ItemsControlFromItemContainer(this) as ViewSwitch;

    sealed partial class Peer(ViewSegment owner) : FrameworkElementAutomationPeer(owner), ISelectionItemProvider
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.ListItem;

        protected override string GetClassNameCore() => nameof(ViewSegment);

        protected override string GetNameCore() =>
            AutomationProperties.GetName(owner) is { Length: > 0 } name ? name : owner.Content as string ?? "";

        protected override object GetPatternCore(PatternInterface patternInterface) =>
            patternInterface == PatternInterface.SelectionItem ? this : base.GetPatternCore(patternInterface);

        public bool IsSelected => owner.IsSelected;

        public IRawElementProviderSimple? SelectionContainer =>
            owner.Switch is { } view && FromElement(view) is { } peer ? ProviderFromPeer(peer) : null;

        public void Select()
        {
            if (owner.Switch is { } view)
                view.SelectedItem = owner;
        }

        public void AddToSelection() => Select();

        public void RemoveFromSelection()
        {
        }
    }
}
