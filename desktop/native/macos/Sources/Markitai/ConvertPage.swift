import AppKit
import MarkitaiKit
import SwiftUI

struct ConvertPage: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var workbench: Workbench

    var body: some View {
        Group {
            if workbench.items.isEmpty {
                EmptyConvert()
            } else {
                SplitLayout(hasSelection: workbench.selection != nil) {
                    ConvertListColumn()
                } detail: {
                    ItemDetail()
                }
            }
        }
        .navigationTitle(L["nav.convert"])
        .toolbar {
            if !workbench.items.isEmpty && !model.showsDetailWhenStacked {
                ToolbarItem(placement: .primaryAction) {
                    Button { withAnimation(Motion.list) { workbench.clearFinished() } } label: {
                        Label(L["convert.clear_finished"], systemImage: "xmark.circle")
                    }
                    .help(L["convert.clear_finished"])
                    .disabled(!workbench.anyFinished)
                }
            }
        }
        .documentToolbar()
    }
}

/// The dashed drop target's border: solid accent while files are over the window.
struct DropBorder: ViewModifier {
    @EnvironmentObject private var model: AppModel

    func body(content: Content) -> some View {
        content.overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(
                    model.dropTargeted ? Brand.tint : Color(nsColor: .tertiaryLabelColor),
                    style: StrokeStyle(lineWidth: 1.5, dash: model.dropTargeted ? [] : [5, 4])
                )
        )
    }
}

struct EmptyConvert: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var settings: UserSettings
    @FocusState private var chooseFocused: Bool

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 48, weight: .ultraLight))
                .foregroundStyle(.secondary)
            Text(model.dropTargeted ? L["convert.drop.release"] : L["convert.empty.title"])
                .font(.title.weight(.semibold))
                .padding(.top, 6)
            Text(L["convert.empty.body", ["folder": Format.path(settings.effectiveOutputFolder)]])
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
                .fixedSize(horizontal: false, vertical: true)
            Button(L["convert.choose"]) { model.chooseFiles() }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .focused($chooseFocused)
                .padding(.top, 8)
            HStack(spacing: 12) {
                line
                Text(L["convert.or"]).font(.subheadline).foregroundStyle(.secondary)
                line
            }
            .accessibilityHidden(true)
            URLRow(glyph: false)
            NoticeLines()
        }
        .padding(40)
        .frame(maxWidth: 560)
        .modifier(DropBorder())
        .accessibilityElement(children: .contain)
        .accessibilityLabel(L["a11y.drop_zone"])
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.bottom, 52)
        // Choose Files… first (DESIGN.md 6.2). A button takes focus only with
        // keyboard navigation on; asked for without it, SwiftUI hands focus to
        // the URL field instead, so then nothing has focus.
        .onAppear { if NSApp.isFullKeyboardAccessEnabled { chooseFocused = true } }
    }

    private var line: some View {
        Rectangle().fill(Color(nsColor: .separatorColor)).frame(height: 1)
    }
}

/// The URL field and Add. The empty page's field has no leading glyph
/// (DESIGN.md 6.2), the list's has the link glyph (6.3).
struct URLRow: View {
    @EnvironmentObject private var model: AppModel
    @FocusState private var focused: Bool
    let glyph: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    if glyph {
                        Image(systemName: "link").foregroundStyle(.secondary).accessibilityHidden(true)
                    }
                    TextField(L["convert.url.label"], text: $model.urlText, prompt: Text(L["convert.url.placeholder"]))
                        .textFieldStyle(.plain)
                        .labelsHidden()
                        .focused($focused)
                        .onSubmit(model.addURLFromField)
                        .accessibilityLabel(L["convert.url.label"])
                }
                .padding(.horizontal, 8)
                .frame(height: 28)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Color(nsColor: .textBackgroundColor))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .strokeBorder(Color(nsColor: .separatorColor))
                )
                Button(L["convert.url.add"], action: model.addURLFromField)
                    .untinted()
            }
            if model.urlInvalid {
                Label {
                    Text(L["convert.url.invalid"])
                } icon: {
                    Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
                }
                .font(.subheadline)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .onAppear { if model.takeFocus(.webAddress) { focused = true } }
        .onChange(of: model.focusRequest) { _ in if model.takeFocus(.webAddress) { focused = true } }
    }
}

struct NoticeLines: View {
    @EnvironmentObject private var workbench: Workbench

    var body: some View {
        if !workbench.notices.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(workbench.notices, id: \.self) { notice in
                    Label {
                        Text(notice).fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    }
                    .font(.subheadline)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
        }
    }
}

struct ConvertListColumn: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var workbench: Workbench

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "arrow.down.doc").foregroundStyle(.secondary).accessibilityHidden(true)
                Text(model.dropTargeted ? L["convert.drop.release"] : L["convert.drop.bar"])
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(L["convert.choose"]) { model.chooseFiles() }
                    .untinted()
            }
            .padding(.leading, 12)
            .padding(.trailing, 8)
            .frame(height: 48)
            .modifier(DropBorder())
            .accessibilityElement(children: .contain)
            .accessibilityLabel(L["a11y.drop_zone"])
            URLRow(glyph: true)
            NoticeLines()
            List(selection: $workbench.selection) {
                ForEach(workbench.items) { item in
                    ItemRow(item: item)
                        .tag(item.id)
                        .simultaneousGesture(TapGesture().onEnded(model.showDetailIfStacked))
                        .transition(.opacity)
                }
            }
            .contextMenu(forSelectionType: UUID.self) { ids in
                if let item = ids.first.flatMap(workbench.item) {
                    ItemMenu(item: item)
                }
            } primaryAction: { _ in
                model.showDetailIfStacked()
            }
            .focusedList()
            .listStyle(.inset)
            .rowHeight()
            .scrollContentBackground(.hidden)
            .accessibilityLabel(L["a11y.conversions"])
            .padding(.horizontal, -12)
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
    }
}

struct ItemRow: View {
    let item: Item

    var body: some View {
        let subtitle = item.subtitle
        HStack(spacing: 10) {
            KindGlyph(kind: item.source.kind)
            VStack(alignment: .leading, spacing: 1) {
                Text(item.title)
                    .lineLimit(1)
                    .truncationMode(item.source.kind == .file ? .middle : .tail)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(isFailed ? .primary : .secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 0)
            trailing
        }
        .frame(height: Row.content)
        .rowName("\(item.title), \(subtitle)")
    }

    private var isFailed: Bool {
        if case .failed = item.state { return true }
        return false
    }

    @ViewBuilder private var trailing: some View {
        switch item.state {
        case .queued:
            Image(systemName: "hourglass").foregroundStyle(.secondary)
        case .converting:
            ProgressView().controlSize(.small)
        case .done(let conversion):
            if !conversion.warnings.isEmpty {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        case .failed:
            Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        }
    }
}

/// DESIGN.md 13.4: rows are 44 pt apart, separator to separator: 36 pt of
/// content and the inset style's 4 pt above and below.
enum Row {
    static let pitch: CGFloat = 44
    static let content: CGFloat = 36
}

/// Focus moves back to the list after the stacked detail (DESIGN.md 14).
struct FocusedList: ViewModifier {
    @EnvironmentObject private var model: AppModel
    @FocusState private var focused: Bool

    func body(content: Content) -> some View {
        content
            .focused($focused)
            .onAppear { if model.takeFocus(.list) { focused = true } }
            .onChange(of: model.focusRequest) { _ in if model.takeFocus(.list) { focused = true } }
    }
}

extension View {
    /// The List measures a row it inserts without animation before the row
    /// has content, and keeps its estimate, this minimum (by default 24 pt):
    /// the estimate is the real height.
    func rowHeight() -> some View {
        environment(\.defaultMinListRowHeight, Row.pitch)
    }

    func focusedList() -> some View { modifier(FocusedList()) }

    /// A row is one element named `<title>, <subtitle>` (DESIGN.md 12), exposed
    /// as text, as the sidebar's rows are: with the name alone, its role is
    /// AXUnknown, without a role description.
    func rowName(_ name: String) -> some View {
        accessibilityElement(children: .ignore)
            .accessibilityLabel(name)
            .accessibilityAddTraits(.isStaticText)
    }
}

struct KindGlyph: View {
    let kind: SourceKind

    var body: some View {
        Image(systemName: kind == .file ? "doc" : "globe")
            .foregroundStyle(.secondary)
            .frame(width: 16)
            .accessibilityHidden(true)
    }
}

struct ItemMenu: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var workbench: Workbench
    let item: Item

    var body: some View {
        switch item.state {
        case .done:
            Button(L["detail.copy"]) { select(); model.copyMarkdown() }
            Button(L["detail.save_as"]) { select(); model.saveAs() }
            Button(L["detail.show.macos"]) { select(); model.showInFinder() }
            Divider()
            remove
        case .failed:
            Button(L["detail.retry"]) { workbench.retry(item.id) }
            Divider()
            remove
        case .queued:
            remove
        case .converting:
            Button(L["convert.remove"]) {}
                .disabled(true)
                .help(L["convert.remove.busy"])
        }
    }

    private var remove: some View {
        Button(L["convert.remove"]) { withAnimation(Motion.list) { workbench.remove(item.id) } }
    }

    private func select() {
        workbench.selection = item.id
        model.page = .convert
    }
}

struct ItemDetail: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var workbench: Workbench
    @EnvironmentObject private var settings: UserSettings

    var body: some View {
        if let item = model.selectedItem {
            switch item.state {
            case .done(let conversion):
                DocumentView(document: Document(item: item, conversion: conversion))
            case .queued:
                StateDetail(item: item) {
                    Text(L["detail.queued"]).foregroundStyle(.secondary)
                }
            case .converting:
                StateDetail(item: item) {
                    Text(L["detail.converting.note"]).foregroundStyle(.secondary)
                }
            case .failed(let failure):
                StateDetail(item: item) {
                    ErrorBlock(item: item, failure: failure)
                }
            }
        } else {
            Text(L["detail.empty"])
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// The detail of an item that is not Done: header, then its body centered.
struct StateDetail<Body: View>: View {
    let item: Item
    @ViewBuilder let content: Body

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            DetailHeader(title: item.title, source: item.source) {
                StatusLine(state: item.state)
            }
            Divider()
            content
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: isFailed ? .topLeading : .center)
                .padding(20)
        }
    }

    private var isFailed: Bool {
        if case .failed = item.state { return true }
        return false
    }
}

struct StatusLine: View {
    let state: ItemState

    var body: some View {
        switch state {
        case .queued:
            Label { Text(L["state.queued"]) } icon: { Image(systemName: "hourglass").foregroundStyle(.secondary) }
        case .converting(let started):
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                TimelineView(.periodic(from: started, by: 1)) { context in
                    Text(L["detail.converting", ["elapsed": Format.elapsed(context.date.timeIntervalSince(started))]])
                        .monospacedDigit()
                }
            }
        case .done(let conversion):
            Label { Text(L["state.done", ["duration": Format.duration(conversion.duration)]]) } icon: {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            }
        case .failed:
            Label { Text(L["state.failed"]) } icon: { Image(systemName: "xmark.octagon.fill").foregroundStyle(.red) }
        }
    }
}

struct ErrorBlock: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var workbench: Workbench
    let item: Item
    let failure: Failure

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(ErrorSentence.text(for: failure))
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            HStack(spacing: 8) {
                Button(L["detail.retry"]) { workbench.retry(item.id) }
                    .keyboardShortcut(.defaultAction)
                    .tint(Brand.tint)
                switch ErrorSentence.of(failure.code).action {
                case .remove:
                    Button(L["convert.remove"]) { withAnimation(Motion.list) { workbench.remove(item.id) } }
                case .openSettings:
                    OpenSettingsButton()
                case .openConfiguration:
                    Button(L["settings.config.open"]) { ConfigurationFile.open() }
                case nil:
                    EmptyView()
                }
            }
            .untinted()
            if !failure.message.isEmpty {
                Text(L["detail.details"]).font(.subheadline.weight(.semibold)).padding(.top, 8)
                ScrollView {
                    Text(failure.message)
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
                .frame(maxHeight: 220)
                .fixedSize(horizontal: false, vertical: true)
                .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.quaternary))
            }
        }
        .frame(maxWidth: 640, alignment: .leading)
    }
}
