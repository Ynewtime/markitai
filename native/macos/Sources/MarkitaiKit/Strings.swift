import Foundation

public enum AppLanguage: String, Sendable {
    case english = "en"
    case simplifiedChinese = "zh-Hans"

    /// DESIGN.md 7.5: zh-Hans when the first preferred language starts with `zh`.
    public static func choose(from preferred: [String]) -> AppLanguage {
        preferred.first?.lowercased().hasPrefix("zh") == true ? .simplifiedChinese : .english
    }

    /// The locale dates and numbers are written in: the app's language with
    /// the user's region, so a 24-hour region keeps its clock.
    public var locale: Locale {
        let region = Locale.current.region?.identifier
        switch self {
        case .english: return Locale(identifier: region.map { "en_\($0)" } ?? "en_US")
        case .simplifiedChinese: return Locale(identifier: "zh-Hans" + (region.map { "_\($0)" } ?? "_CN"))
        }
    }
}

/// Every user-visible string, from `native/strings.json` (DESIGN.md section 10),
/// embedded in the binary.
public struct Strings: Sendable {
    public let language: AppLanguage
    private let table: [String: [String: String]]

    public static let table: [String: [String: String]] = {
        let data = Data(PackageResources.strings_json)
        return (try? JSONDecoder().decode([String: [String: String]].self, from: data)) ?? [:]
    }()

    public init(language: AppLanguage) {
        self.language = language
        self.table = Strings.table
    }

    public subscript(_ key: String, _ arguments: [String: String] = [:]) -> String {
        guard let text = table[key]?[language.rawValue] else {
            assertionFailure("strings.json has no \(key)")
            return key
        }
        return arguments.reduce(text) { text, argument in
            text.replacingOccurrences(of: "{\(argument.key)}", with: argument.value)
        }
    }

    /// A `.one` / `.other` pair, chosen by `count == 1`, with `{count}` filled.
    public func plural(_ key: String, count: Int, _ arguments: [String: String] = [:]) -> String {
        var arguments = arguments
        arguments["count"] = Format.integer(count, language: language)
        return self[key + (count == 1 ? ".one" : ".other"), arguments]
    }
}

/// The app's strings in the language chosen at launch.
public let L = Strings(language: AppLanguage.choose(from: Locale.preferredLanguages))
