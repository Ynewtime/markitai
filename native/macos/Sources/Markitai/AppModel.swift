import AppKit
import Combine
import MarkitaiKit
import SwiftUI
import UniformTypeIdentifiers

enum Page: Hashable {
    case convert
    case history
}

enum DetailMode: Hashable {
    case preview
    case markdown
}

/// A finished conversion the detail shows: a Done item or a History entry.
struct Document: Equatable {
    let id: String
    let title: String
    let source: Source
    let folder: String
    let markdownPath: String
    let duration: Double
    let warnings: [String]
    let usage: Usage?
    let finished: Date?

    init(item: Item, conversion: Conversion) {
        id = item.id.uuidString
        title = item.title
        source = item.source
        folder = conversion.folder
        markdownPath = conversion.markdownPath
        duration = conversion.duration
        warnings = conversion.warnings
        usage = conversion.usage
        finished = nil
    }

    init(entry: HistoryEntry) {
        id = entry.id
        title = entry.title
        source = entry.sourceValue
        folder = entry.folder
        markdownPath = entry.markdownPath
        duration = entry.duration
        warnings = entry.warnings
        usage = entry.usage
        finished = entry.finishedDate
    }
}

enum Loaded: Equatable {
    case loading
    case text(String)
    case missing
}

/// The window's state and every user action, shared by the views, the menu
/// bar and the application delegate. The app has one window.
@MainActor
final class AppModel: ObservableObject {
    static let shared = AppModel()
    static let windowID = "main"

    let abiMatches = Core.abiMatches
    let settings: UserSettings
    let history: History
    let workbench: Workbench

    @Published var page: Page = .convert
    @Published var historySelection: String?
    @Published var historyQuery = ""
    @Published var mode: DetailMode = .preview
    /// Stacked layout only: the detail has replaced the list (DESIGN.md 14).
    @Published var showsDetailWhenStacked = false
    @Published var isStacked = false
    @Published var dropTargeted = false
    @Published var copiedID: String?
    @Published var urlText = "" { didSet { if urlText != oldValue { urlInvalid = false } } }
    @Published var urlInvalid = false
    @Published var focusURLField = 0
    @Published var focusSearchField = 0
    @Published private(set) var markdownByPath: [String: Loaded] = [:]

    var openWindow: (() -> Void)?
    var openSettings: (() -> Void)?
    private var forwarding: [AnyCancellable] = []

    private init() {
        let data = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dev.markitai.Markitai")
        settings = UserSettings(store: UserDefaults.standard)
        history = History(file: data.appendingPathComponent("history.json"))
        workbench = Workbench(settings: settings, history: history)
        workbench.announce = { text in
            NSAccessibility.post(
                element: NSApp.mainWindow ?? NSApp as Any,
                notification: .announcementRequested,
                userInfo: [.announcement: text, .priority: NSAccessibilityPriorityLevel.medium.rawValue]
            )
        }
        // What this model derives (the shown document, what Retry and Remove
        // act on) follows the queue, History and Settings: the menu bar and
        // toolbar that observe only this model see them change.
        let children = [workbench.objectWillChange, history.objectWillChange, settings.objectWillChange]
        forwarding = children.map { $0.sink { [weak self] _ in self?.objectWillChange.send() } }
    }

    // MARK: Adding

    func add(_ inputs: [Input]) {
        guard !inputs.isEmpty else { return }
        page = .convert
        withAnimation(Motion.list) {
            _ = workbench.add(inputs)
        }
    }

    func addURLFromField() {
        guard let address = WebAddress.parse(urlText) else {
            urlInvalid = true
            return
        }
        add([.web(address)])
        urlText = ""
    }

    func focusWebAddress() {
        page = .convert
        showsDetailWhenStacked = false
        focusURLField += 1
    }

    func chooseFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.prompt = L["convert.choose"].replacingOccurrences(of: "…", with: "")
        guard panel.runModal() == .OK else { return }
        add(panel.urls.map(Input.file))
    }

    func pasteFromClipboard() {
        let board = NSPasteboard.general
        if let files = board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL], !files.isEmpty {
            add(files.map(Input.file))
            return
        }
        guard let text = board.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines),
              text.lowercased().hasPrefix("http://") || text.lowercased().hasPrefix("https://"),
              let address = WebAddress.parse(text)
        else { return }
        add([.web(address)])
    }

    /// Files and web links dropped on the window, in the order given.
    func drop(_ providers: [NSItemProvider]) -> Bool {
        let dropped = DroppedInputs(count: providers.count)
        let group = DispatchGroup()
        for (index, provider) in providers.enumerated() {
            let isFile = provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
            guard isFile || provider.canLoadObject(ofClass: URL.self) else { continue }
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                Task { @MainActor in
                    if let url {
                        if isFile, url.isFileURL {
                            dropped.inputs[index] = .file(url)
                        } else if !isFile, ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
                                  let address = WebAddress.parse(url.absoluteString) {
                            dropped.inputs[index] = .web(address)
                        }
                    }
                    group.leave()
                }
            }
        }
        group.notify(queue: .main) { [self] in
            add(dropped.inputs.compactMap { $0 })
        }
        return true
    }

    // MARK: The selected document

    var selectedItem: Item? { workbench.item(workbench.selection) }

    var selectedEntry: HistoryEntry? {
        guard let id = historySelection else { return nil }
        return history.entries.first { $0.id == id }
    }

    /// The document the detail shows, on either page.
    var document: Document? {
        switch page {
        case .convert:
            guard let item = selectedItem, case .done(let conversion) = item.state else { return nil }
            return Document(item: item, conversion: conversion)
        case .history:
            return selectedEntry.map(Document.init(entry:))
        }
    }

    /// The Markdown of the shown document, read off the main thread for History.
    func markdown(for document: Document) -> Loaded {
        if case .done(let conversion) = selectedItem?.state, page == .convert, conversion.markdownPath == document.markdownPath {
            return .text(conversion.markdown)
        }
        return markdownByPath[document.markdownPath] ?? .loading
    }

    func load(_ document: Document) {
        let path = document.markdownPath
        guard page == .history || markdownByPath[path] == nil else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            let loaded: Loaded = (try? String(contentsOfFile: path, encoding: .utf8)).map(Loaded.text) ?? .missing
            Task { @MainActor in self.markdownByPath[path] = loaded }
        }
    }

    /// False when a History entry's Markdown file is gone: then only Remove works.
    var canShowDocument: Bool {
        guard let document else { return false }
        return markdown(for: document) != .missing
    }

    var shownMarkdown: String? {
        guard let document, case .text(let text) = markdown(for: document) else { return nil }
        return text
    }

    func copyMarkdown() {
        guard let document, let text = shownMarkdown else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedID = document.id
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [self] in
            if copiedID == document.id { copiedID = nil }
        }
    }

    /// DESIGN.md 7.2: the shown Markdown, plus the conversion's `.markitai`
    /// folder next to it when the text points into it, never replacing a file.
    func saveAs() {
        guard let document, let text = shownMarkdown else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = (document.folder as NSString).lastPathComponent + ".md"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            try text.write(to: target, atomically: true, encoding: .utf8)
            if text.contains(".markitai/") {
                let assets = URL(fileURLWithPath: document.folder).appendingPathComponent(".markitai")
                try Self.merge(assets, into: target.deletingLastPathComponent().appendingPathComponent(".markitai"))
            }
        } catch {
            let alert = NSAlert()
            alert.messageText = L["error.save_failed"]
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    static func merge(_ source: URL, into target: URL) throws {
        let files = FileManager.default
        guard let walker = files.enumerator(at: source, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        try files.createDirectory(at: target, withIntermediateDirectories: true)
        for case let file as URL in walker {
            let relative = String(file.standardizedFileURL.path.dropFirst(source.standardizedFileURL.path.count + 1))
            let destination = target.appendingPathComponent(relative)
            if (try? file.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                try files.createDirectory(at: destination, withIntermediateDirectories: true)
            } else if !files.fileExists(atPath: destination.path) {
                try files.copyItem(at: file, to: destination)
            }
        }
    }

    func showInFinder() {
        guard canShowDocument, let document else { return }
        let file = URL(fileURLWithPath: document.markdownPath)
        if FileManager.default.fileExists(atPath: file.path) {
            NSWorkspace.shared.activateFileViewerSelecting([file])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: document.folder)])
        }
    }

    /// A click on a row in the stacked list shows its detail, also when the
    /// row is still selected from before Back (DESIGN.md 14), which changes
    /// no selection.
    func showDetailIfStacked() {
        if isStacked { showsDetailWhenStacked = true }
    }

    // MARK: Item commands

    var canRetry: Bool {
        guard page == .convert, case .failed = selectedItem?.state else { return false }
        return true
    }

    func retrySelected() {
        guard canRetry, let id = workbench.selection else { return }
        workbench.retry(id)
    }

    var canRemove: Bool {
        switch page {
        case .convert: return workbench.selection.map(workbench.canRemove) ?? false
        case .history: return selectedEntry != nil
        }
    }

    func removeSelected() {
        switch page {
        case .convert:
            guard let id = workbench.selection else { return }
            withAnimation(Motion.list) { workbench.remove(id) }
        case .history:
            guard let entry = selectedEntry else { return }
            removeFromHistory(entry.id)
        }
    }

    func removeFromHistory(_ id: String) {
        let entries = filteredEntries
        let index = entries.firstIndex { $0.id == id }
        withAnimation(Motion.list) { history.remove(id: id) }
        guard historySelection == id, let index else { return }
        let remaining = filteredEntries
        historySelection = remaining.indices.contains(index) ? remaining[index].id : remaining.last?.id
    }

    func clearHistory() {
        withAnimation(Motion.list) { history.clear() }
        historySelection = nil
    }

    var filteredEntries: [HistoryEntry] {
        let query = historyQuery.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return history.entries }
        return history.entries.filter {
            $0.title.localizedCaseInsensitiveContains(query) || $0.source.localizedCaseInsensitiveContains(query)
        }
    }

    // MARK: View

    func go(to page: Page) {
        self.page = page
        showsDetailWhenStacked = false
    }

    func toggleMode() {
        mode = mode == .preview ? .markdown : .preview
    }

    func searchHistory() {
        go(to: .history)
        focusSearchField += 1
    }

    func back() {
        showsDetailWhenStacked = false
    }

    func showWindow() {
        if let window = NSApp.windows.first(where: { $0.identifier?.rawValue.hasPrefix(Self.windowID) == true && $0.canBecomeMain }) {
            window.makeKeyAndOrderFront(nil)
        } else {
            openWindow?()
        }
        NSApp.activate(ignoringOtherApps: true)
    }
}

@MainActor
private final class DroppedInputs {
    var inputs: [Input?]
    init(count: Int) { inputs = Array(repeating: nil, count: count) }
}

enum Motion {
    /// Insertions and removals fade over 150 ms, not at all with Reduce Motion.
    static var list: Animation? {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? nil : .easeInOut(duration: 0.15)
    }
}
