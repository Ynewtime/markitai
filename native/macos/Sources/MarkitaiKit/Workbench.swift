import Foundation

/// Something the user asked to convert, before it becomes an item.
public enum Input: Equatable, Sendable {
    case file(URL)
    case web(String)
}

public enum WebAddress {
    /// DESIGN.md 7.1: trim; without a scheme but with a dot, prefix `https://`;
    /// accept only http and https URLs with a host.
    public static func parse(_ text: String) -> String? {
        var text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !text.contains(where: \.isWhitespace) else { return nil }
        if !text.contains("://") {
            guard text.contains(".") else { return nil }
            text = "https://" + text
        }
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = components.host, !host.isEmpty
        else { return nil }
        return text
    }
}

/// The Convert list and its queue (DESIGN.md 4.1): newest first, two
/// conversions at a time, oldest queued first.
@MainActor
public final class Workbench: ObservableObject {
    public static let concurrency = 2

    @Published public private(set) var items: [Item] = []
    @Published public var selection: UUID?
    @Published public private(set) var notices: [String] = []
    /// Warnings disclosures the user expanded, by item or History entry id.
    @Published public var expandedWarnings: Set<String> = []

    public let settings: UserSettings
    public let history: History
    /// Says a state change aloud (`a11y.done`, `a11y.failed`).
    public var announce: (String) -> Void = { _ in }

    private let run: Job.Run
    private let claims = Claims()
    private var nextTurn = 0
    private var noticeExpiry: Task<Void, Never>?
    private let strings: Strings

    public init(settings: UserSettings, history: History, strings: Strings = L, run: @escaping Job.Run = Job.core) {
        self.settings = settings
        self.history = history
        self.strings = strings
        self.run = run
        if let name = history.resetName {
            show([strings["history.reset", ["name": name]]])
        }
    }

    public var anyConverting: Bool { items.contains { if case .converting = $0.state { return true } else { return false } } }
    public var convertingCount: Int { items.filter { if case .converting = $0.state { return true } else { return false } }.count }
    public var queuedCount: Int { items.filter { $0.state == .queued }.count }
    public var anyFinished: Bool { items.contains { !$0.state.isActive } }

    public func item(_ id: UUID?) -> Item? {
        guard let id else { return nil }
        return items.first { $0.id == id }
    }

    /// Adds files and web addresses in the order given. Folders are refused
    /// (a `.numbers` package is a document), and a source already queued or
    /// converting is not added twice.
    @discardableResult
    public func add(_ inputs: [Input], now: Date = Date()) -> [UUID] {
        var folders: [String] = []
        var duplicates: [String] = []
        var added: [Item] = []
        let options = settings.options
        for input in inputs {
            let source: Source
            switch input {
            case .file(let url):
                let path = url.path
                var isDirectory: ObjCBool = false
                if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue,
                   !url.lastPathComponent.lowercased().hasSuffix(".numbers") {
                    folders.append(url.lastPathComponent)
                    continue
                }
                source = .file(path)
            case .web(let address):
                source = .url(address)
            }
            let waiting = items.contains { $0.source == source && $0.state.isActive }
                || added.contains { $0.source == source }
            if waiting {
                duplicates.append(source.defaultTitle)
                continue
            }
            added.append(Item(source: source, options: options, turn: nextTurn, now: now))
            nextTurn += 1
        }
        var lines: [String] = []
        if folders.count == 1 {
            lines.append(strings["notice.folder", ["name": folders[0]]])
        } else if folders.count > 1 {
            lines.append(strings["notice.folders", ["count": Format.integer(folders.count, language: strings.language)]])
        }
        lines += duplicates.map { strings["notice.duplicate", ["name": $0]] }
        show(lines)
        guard !added.isEmpty else { return [] }
        if items.isEmpty {
            selection = added[0].id
        }
        items.insert(contentsOf: added, at: 0)
        pump()
        return added.map(\.id)
    }

    public func retry(_ id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }), case .failed = items[index].state else { return }
        items[index].options = settings.options
        items[index].state = .queued
        items[index].turn = nextTurn
        nextTurn += 1
        pump()
    }

    public func canRemove(_ id: UUID) -> Bool {
        guard let item = item(id) else { return false }
        if case .converting = item.state { return false }
        return true
    }

    public func remove(_ id: UUID) {
        guard canRemove(id) else { return }
        remove { $0.id == id }
    }

    public func clearFinished() {
        remove { !$0.state.isActive }
    }

    /// Removes items; a removed selection moves to the next row, else the previous.
    private func remove(where condition: (Item) -> Bool) {
        let before = items
        items.removeAll(where: condition)
        guard let selected = selection, let index = before.firstIndex(where: { $0.id == selected }),
              !items.contains(where: { $0.id == selected })
        else { return }
        let remaining = Set(items.map(\.id))
        let next = before[index...].first { remaining.contains($0.id) }
            ?? before[..<index].last { remaining.contains($0.id) }
        selection = next?.id
    }

    private func pump() {
        while !claims.isQuitting, convertingCount < Workbench.concurrency,
              let index = items.indices.filter({ items[$0].state == .queued }).min(by: { items[$0].turn < items[$1].turn }) {
            start(index)
        }
    }

    private func start(_ index: Int) {
        let item = items[index]
        items[index].state = .converting(startedAt: Date())
        let run = self.run
        let claims = self.claims
        let outputFolder = settings.effectiveOutputFolder
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            guard let result = run(item.source, item.options, outputFolder, { claims.claim(item.id, $0) }) else { return }
            Task { @MainActor in self.finish(item.id, result) }
        }
    }

    private func finish(_ id: UUID, _ result: Result<Conversion, Failure>) {
        guard claims.finish(id), let index = items.firstIndex(where: { $0.id == id }) else { return }
        switch result {
        case .success(let conversion):
            if case .url = items[index].source, let title = conversion.urlTitle {
                items[index].title = title
            }
            items[index].state = .done(conversion)
            history.add(HistoryEntry(item: items[index], conversion: conversion, finished: Date()))
            announce(strings["a11y.done", ["title": items[index].title]])
        case .failure(let failure):
            items[index].state = .failed(failure)
            announce(strings["a11y.failed", ["title": items[index].title]])
        }
        pump()
    }

    /// The user confirmed Quit (DESIGN.md 7.3): deletes the folders claimed for
    /// items still Converting, whose conversions end with the process. After
    /// this nothing is claimed or started, and results change nothing.
    public func discardConverting() {
        for folder in claims.quit() {
            try? FileManager.default.removeItem(atPath: folder)
        }
    }

    /// Notices stay 8 s, or until the next add.
    private func show(_ lines: [String]) {
        noticeExpiry?.cancel()
        notices = lines
        guard !lines.isEmpty else { return }
        noticeExpiry = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            guard !Task.isCancelled else { return }
            self?.notices = []
        }
    }

    public func clearNotices() {
        show([])
    }
}

/// The folders claimed for items still Converting. Workers claim under its
/// lock and Quit takes the same lock, so every folder is either claimed before
/// Quit, which deletes it, or never claimed.
private final class Claims: @unchecked Sendable {
    private let lock = NSLock()
    private var quitting = false
    private var folders: [UUID: String] = [:]

    var isQuitting: Bool { lock.withLock { quitting } }

    func claim(_ id: UUID, _ make: () -> Result<String, Failure>) -> Result<String, Failure>? {
        lock.withLock {
            guard !quitting else { return nil }
            let result = make()
            if case .success(let folder) = result {
                folders[id] = folder
            }
            return result
        }
    }

    /// The item's result reached the list; false once quitting.
    func finish(_ id: UUID) -> Bool {
        lock.withLock {
            folders[id] = nil
            return !quitting
        }
    }

    /// The folders to delete. Nothing is claimed after this.
    func quit() -> [String] {
        lock.withLock {
            quitting = true
            return Array(folders.values)
        }
    }
}
