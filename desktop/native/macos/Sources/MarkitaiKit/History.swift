import Foundation

/// One finished conversion in `history.json` (DESIGN.md 4.4).
public struct HistoryEntry: Codable, Identifiable, Equatable {
    public var id: String
    public var source: String
    public var kind: SourceKind
    public var title: String
    public var folder: String
    public var markdownPath: String
    public var finishedAt: String
    public var duration: Double
    public var warnings: [String]
    public var usage: Usage?
    public var options: Options
    public var markitaiVersion: String

    enum CodingKeys: String, CodingKey {
        case id, source, kind, title, folder
        case markdownPath = "markdown_path"
        case finishedAt = "finished_at"
        case duration, warnings, usage, options
        case markitaiVersion = "markitai_version"
    }

    public var sourceValue: Source { Source(raw: source, kind: kind) }

    public var finishedDate: Date? { try? HistoryEntry.timestamp.parse(finishedAt) }

    /// UTC, RFC 3339, seconds (DESIGN.md 4.4).
    static let timestamp = Date.ISO8601FormatStyle()

    public init(item: Item, conversion: Conversion, finished: Date, version: String = Core.version) {
        id = item.id.uuidString.lowercased()
        source = item.source.raw
        kind = item.source.kind
        title = item.title
        folder = conversion.folder
        markdownPath = conversion.markdownPath
        finishedAt = HistoryEntry.timestamp.format(finished)
        duration = conversion.duration
        warnings = conversion.warnings
        usage = conversion.usage
        options = item.options
        markitaiVersion = version
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(source, forKey: .source)
        try container.encode(kind, forKey: .kind)
        try container.encode(title, forKey: .title)
        try container.encode(folder, forKey: .folder)
        try container.encode(markdownPath, forKey: .markdownPath)
        try container.encode(finishedAt, forKey: .finishedAt)
        try container.encode(duration, forKey: .duration)
        try container.encode(warnings, forKey: .warnings)
        try container.encode(usage, forKey: .usage)
        try container.encode(options, forKey: .options)
        try container.encode(markitaiVersion, forKey: .markitaiVersion)
    }
}

private struct HistoryFile: Codable {
    var version: Int
    var entries: [HistoryEntry]
}

/// `history.json`: newest first, at most 500, written whole after every change.
@MainActor
public final class History: ObservableObject {
    public static let limit = 500

    @Published public private(set) var entries: [HistoryEntry] = []
    /// The name the unreadable file was kept as, reported once.
    public private(set) var resetName: String?
    public let file: URL

    public init(file: URL, now: Date = Date()) {
        self.file = file
        guard let data = try? Data(contentsOf: file) else { return }
        if let stored = try? JSONDecoder().decode(HistoryFile.self, from: data), stored.version == 1 {
            entries = Array(stored.entries.prefix(History.limit))
            return
        }
        let stamp = DateFormatter()
        stamp.locale = Locale(identifier: "en_US_POSIX")
        stamp.timeZone = TimeZone(identifier: "UTC")
        stamp.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        let name = "history-unreadable-\(stamp.string(from: now)).json"
        let kept = file.deletingLastPathComponent().appendingPathComponent(name)
        if (try? FileManager.default.moveItem(at: file, to: kept)) != nil {
            resetName = name
        }
    }

    /// An entry's Markdown as its Convert item showed it: the file without its
    /// front matter (DESIGN.md 3.3). Nil when the file is gone or not UTF-8.
    nonisolated public static func markdown(at path: String) -> String? {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        return Core.markdownBody(text) ?? text
    }

    public func add(_ entry: HistoryEntry) {
        entries.insert(entry, at: 0)
        if entries.count > History.limit {
            entries.removeLast(entries.count - History.limit)
        }
        write()
    }

    public func remove(id: String) {
        entries.removeAll { $0.id == id }
        write()
    }

    public func clear() {
        entries = []
        write()
    }

    private func write() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(HistoryFile(version: 1, entries: entries)) else { return }
        let folder = file.deletingLastPathComponent()
        let temporary = folder.appendingPathComponent(file.lastPathComponent + ".tmp")
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let handle = try FileHandle(forWritingTo: create(temporary))
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
            if rename(temporary.path, file.path) != 0 {
                throw CocoaError(.fileWriteUnknown)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
        }
    }

    private func create(_ url: URL) throws -> URL {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw CocoaError(.fileWriteNoPermission)
        }
        return url
    }
}
