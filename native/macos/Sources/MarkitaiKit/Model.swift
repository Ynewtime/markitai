import Foundation

public enum SourceKind: String, Codable, Sendable {
    case file
    case url
}

/// What one conversion reads: an absolute file path or an http(s) URL.
public enum Source: Hashable, Sendable {
    case file(String)
    case url(String)

    public var raw: String {
        switch self {
        case .file(let path): return path
        case .url(let url): return url
        }
    }

    public var kind: SourceKind {
        switch self {
        case .file: return .file
        case .url: return .url
        }
    }

    /// A file's name, or the URL without its scheme.
    public var defaultTitle: String {
        switch self {
        case .file(let path): return (path as NSString).lastPathComponent
        case .url(let url):
            guard let range = url.range(of: "://") else { return url }
            return String(url[range.upperBound...])
        }
    }

    /// The source line under a detail title: a path with `~`, or the full URL.
    public var display: String {
        switch self {
        case .file(let path): return Format.path(path)
        case .url(let url): return url
        }
    }

    public init(raw: String, kind: SourceKind) {
        self = kind == .file ? .file(raw) : .url(raw)
    }
}

public enum ConfigSource: String, Codable, Sendable {
    case user
    case builtin
}

/// The settings a conversion captured when it was added or retried.
public struct Options: Codable, Equatable, Sendable {
    public var llm = false
    public var ocr = false
    public var screenshot = false
    public var alt = false
    public var desc = false
    public var config: ConfigSource = .user

    public init(llm: Bool = false, ocr: Bool = false, screenshot: Bool = false, alt: Bool = false, desc: Bool = false, config: ConfigSource = .user) {
        self.llm = llm
        self.ocr = ocr
        self.screenshot = screenshot
        self.alt = alt
        self.desc = desc
        self.config = config
    }
}

public enum CostStatus: String, Codable, Sendable {
    case complete
    case partial
    case unknown
}

/// AI usage of one conversion; absent when no request ran.
public struct Usage: Codable, Equatable, Sendable {
    public var requests: Int
    public var inputTokens: Int
    public var outputTokens: Int
    public var costUSD: Double
    public var cost: CostStatus

    public var tokens: Int { inputTokens + outputTokens }

    enum CodingKeys: String, CodingKey {
        case requests
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case costUSD = "cost_usd"
        case cost
    }

    public init(requests: Int, inputTokens: Int, outputTokens: Int, costUSD: Double, cost: CostStatus) {
        self.requests = requests
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.costUSD = costUSD
        self.cost = cost
    }
}

/// What a finished conversion shows (DESIGN.md 4.2).
public struct Conversion: Equatable, Sendable {
    public var folder: String
    public var markdown: String
    public var markdownPath: String
    public var duration: Double
    public var warnings: [String]
    public var usage: Usage?
    public var urlTitle: String?

    public init(folder: String, markdown: String, markdownPath: String, duration: Double, warnings: [String], usage: Usage?, urlTitle: String?) {
        self.folder = folder
        self.markdown = markdown
        self.markdownPath = markdownPath
        self.duration = duration
        self.warnings = warnings
        self.usage = usage
        self.urlTitle = urlTitle
    }
}

public struct Failure: Error, Equatable, Sendable {
    public var code: String
    public var message: String
    public var folderWasClaimed: Bool

    public init(code: String, message: String, folderWasClaimed: Bool) {
        self.code = code
        self.message = message
        self.folderWasClaimed = folderWasClaimed
    }
}

public enum ItemState: Equatable, Sendable {
    case queued
    case converting(startedAt: Date, folder: String)
    case done(Conversion)
    case failed(Failure)

    public var isActive: Bool {
        switch self {
        case .queued, .converting: return true
        case .done, .failed: return false
        }
    }
}

/// One source in the Convert list (DESIGN.md 4.1). The list lives in memory.
public struct Item: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let source: Source
    public var title: String
    public let addedAt: Date
    public var options: Options
    public var state: ItemState
    /// Processing order: lower starts first.
    var turn: Int

    init(source: Source, options: Options, turn: Int, now: Date) {
        self.id = UUID()
        self.source = source
        self.title = source.defaultTitle
        self.addedAt = now
        self.options = options
        self.state = .queued
        self.turn = turn
    }
}

/// The second button of a Failed item's error block (DESIGN.md section 9).
public enum ErrorAction: Equatable {
    case remove
    case openSettings
    case openConfiguration
}

public struct ErrorSentence: Equatable {
    public let key: String
    public let action: ErrorAction?

    /// The mapped sentence for an error code.
    public static func of(_ code: String) -> ErrorSentence {
        switch code {
        case "not_found": return .init(key: "error.not_found", action: .remove)
        case "is_directory": return .init(key: "error.is_directory", action: .remove)
        case "no_model_configured": return .init(key: "error.no_model_configured", action: .openSettings)
        case "invalid_input": return .init(key: "error.invalid_input", action: nil)
        case "unsupported": return .init(key: "error.unsupported", action: .remove)
        case "conversion_error": return .init(key: "error.conversion_error", action: nil)
        case "fetch_error": return .init(key: "error.fetch_error", action: nil)
        case "config_error": return .init(key: "error.config_error", action: .openConfiguration)
        case "io_error": return .init(key: "error.io_error", action: .openSettings)
        case "output_unavailable": return .init(key: "error.output_unavailable", action: .openSettings)
        case "image_only": return .init(key: "error.image_only", action: .openSettings)
        case "skipped": return .init(key: "error.skipped", action: nil)
        case "invalid_json", "internal_error": return .init(key: "error.internal", action: nil)
        default: return .init(key: "error.unknown", action: nil)
        }
    }

    /// The sentence with its placeholders filled.
    public static func text(for failure: Failure, outputFolder: String, strings: Strings = L) -> String {
        strings[of(failure.code).key, [
            "settings": strings["nav.settings"],
            "folder": Format.path(outputFolder),
            "reason": failure.message,
        ]]
    }
}
