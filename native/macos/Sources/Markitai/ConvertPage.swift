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
            if !workbench.items.isEmpty {
                ToolbarItem(placement: .navigation) {
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
            URLRow()
            NoticeLines()
        }
        .padding(40)
        .frame(width: 560)
        .modifier(DropBorder())
        .accessibilityElement(children: .contain)
        .accessibilityLabel(L["a11y.drop_zone"])
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.bottom, 52)
        .onAppear { chooseFocused = true }
    }

    private var line: some View {
        Rectangle().fill(Color(nsColor: .separatorColor)).frame(height: 1)
    }
}

struct URLRow: View {
    @EnvironmentObject private var model: AppModel
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "link").foregroundStyle(.secondary).accessibilityHidden(true)
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
        .onChange(of: model.focusURLField) { _ in focused = true }
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
            URLRow()
            NoticeLines()
            List(selection: Binding(get: { workbench.selection }, set: { id in
                workbench.selection = id
                if id != nil && model.isStacked { model.showsDetailWhenStacked = true }
            })) {
                ForEach(workbench.items) { item in
                    ItemRow(item: item)
                        .tag(item.id)
                        .simultaneousGesture(TapGesture().onEnded(model.showDetailIfStacked))
                        .contextMenu { ItemMenu(item: item) }
                        .transition(.opacity)
                }
            }
            .listStyle(.inset)
            .scrollContentBackground(.hidden)
            .accessibilityLabel(L["a11y.conversions"])
            .padding(.horizontal, -12)
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
    }
}

/// The state text of a list row (DESIGN.md 4.1).
func subtitle(of item: Item, outputFolder: String) -> String {
    switch item.state {
    case .queued: return L["state.queued"]
    case .converting: return L["state.converting"]
    case .done(let conversion):
        var text = L["state.done", ["duration": Format.duration(conversion.duration)]]
        if !conversion.warnings.isEmpty {
            text += " · " + L.plural("state.warnings", count: conversion.warnings.count)
        }
        return text
    case .failed(let failure):
        return L["row.failed", ["reason": ErrorSentence.text(for: failure, outputFolder: outputFolder)]]
    }
}

struct ItemRow: View {
    @EnvironmentObject private var settings: UserSettings
    let item: Item

    var body: some View {
        let subtitle = subtitle(of: item, outputFolder: settings.effectiveOutputFolder)
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
        .frame(height: 44)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.title), \(subtitle)")
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
        case .converting(let started, _):
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
    @EnvironmentObject private var settings: UserSettings
    let item: Item
    let failure: Failure

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(ErrorSentence.text(for: failure, outputFolder: settings.effectiveOutputFolder))
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
