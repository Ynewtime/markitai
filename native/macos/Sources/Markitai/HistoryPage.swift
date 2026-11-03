import AppKit
import MarkitaiKit
import SwiftUI

struct HistoryPage: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var history: History
    @ViewState private var confirmingClear = false

    var body: some View {
        Group {
            if history.entries.isEmpty {
                VStack(spacing: 8) {
                    Text(L["history.empty.title"]).font(.title.weight(.semibold))
                    Text(L["history.empty.body"])
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 420)
                }
                .padding(40)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                SplitLayout(hasSelection: model.selectedEntry != nil) {
                    HistoryListColumn()
                } detail: {
                    if let entry = model.selectedEntry {
                        DocumentView(document: Document(entry: entry))
                    } else {
                        Text(L["detail.empty"])
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            }
        }
        .navigationTitle(L["nav.history"])
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button { confirmingClear = true } label: {
                    Label(L["history.clear"], systemImage: "trash")
                }
                .help(L["history.clear"])
                .disabled(history.entries.isEmpty)
            }
        }
        .documentToolbar()
        .confirmationDialog(
            L.plural("history.clear.title", count: history.entries.count),
            isPresented: $confirmingClear
        ) {
            Button(L["history.clear.confirm"], role: .destructive, action: model.clearHistory)
            Button(L["common.cancel"], role: .cancel) {}
        } message: {
            Text(L["history.clear.body"])
        }
    }
}

struct HistoryListColumn: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var history: History

    var body: some View {
        let entries = model.filteredEntries
        VStack(spacing: 8) {
            SearchField(text: $model.historyQuery, focusRequest: model.focusSearchField)
                .frame(height: 28)
            if entries.isEmpty {
                Text(L["history.no_results", ["query": model.historyQuery.trimmingCharacters(in: .whitespaces)]])
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.top, 24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            } else {
                List(selection: Binding(get: { model.historySelection }, set: { id in
                    model.historySelection = id
                    if id != nil && model.isStacked { model.showsDetailWhenStacked = true }
                })) {
                    ForEach(entries) { entry in
                        HistoryRow(entry: entry)
                            .tag(entry.id)
                            .simultaneousGesture(TapGesture().onEnded(model.showDetailIfStacked))
                            .contextMenu { HistoryMenu(entry: entry) }
                            .transition(.opacity)
                    }
                }
                .listStyle(.inset)
                .rowHeight()
                .scrollContentBackground(.hidden)
                .accessibilityLabel(L["nav.history"])
                .padding(.horizontal, -12)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        let subtitle = subtitle
        HStack(spacing: 10) {
            KindGlyph(kind: entry.kind)
            VStack(alignment: .leading, spacing: 1) {
                Text(entry.title)
                    .lineLimit(1)
                    .truncationMode(entry.kind == .file ? .middle : .tail)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if !entry.warnings.isEmpty {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        }
        .frame(height: 44)
        .rowName("\(entry.title), \(subtitle)")
    }

    private var subtitle: String {
        var text = entry.finishedDate.map { Format.finished($0) } ?? entry.finishedAt
        if !entry.warnings.isEmpty {
            text += " · " + L.plural("state.warnings", count: entry.warnings.count)
        }
        return text
    }
}

struct HistoryMenu: View {
    @EnvironmentObject private var model: AppModel
    let entry: HistoryEntry

    var body: some View {
        Button(L["detail.copy"]) { select(); model.copyMarkdown() }
        Button(L["detail.save_as"]) { select(); model.saveAs() }
        Button(L["detail.show.macos"]) { select(); model.showInFinder() }
        Divider()
        Button(L["history.remove"]) { model.removeFromHistory(entry.id) }
    }

    private func select() {
        model.historySelection = entry.id
        model.load(Document(entry: entry))
    }
}

/// `NSSearchField`, labeled `history.search`, filtering as you type.
struct SearchField: NSViewRepresentable {
    @Binding var text: String
    let focusRequest: Int

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    func makeNSView(context: Context) -> NSSearchField {
        let field = NSSearchField()
        field.placeholderString = L["history.search"]
        field.setAccessibilityLabel(L["history.search"])
        field.delegate = context.coordinator
        field.sendsSearchStringImmediately = true
        return field
    }

    func updateNSView(_ field: NSSearchField, context: Context) {
        context.coordinator.text = $text
        if field.stringValue != text { field.stringValue = text }
        if context.coordinator.focusRequest != focusRequest {
            context.coordinator.focusRequest = focusRequest
            DispatchQueue.main.async { field.window?.makeFirstResponder(field) }
        }
    }

    final class Coordinator: NSObject, NSSearchFieldDelegate {
        var text: Binding<String>
        var focusRequest = 0

        init(text: Binding<String>) { self.text = text }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSSearchField else { return }
            text.wrappedValue = field.stringValue
        }
    }
}
