import Foundation
import Testing
@testable import MarkitaiKit

struct CopyTests {
    let english = Strings(language: .english)
    let chinese = Strings(language: .simplifiedChinese)

    /// The keys the app's sources look up, as `L["key"…]` or `L.plural("key"…)`.
    func usedKeys() throws -> Set<String> {
        let pattern = try NSRegularExpression(pattern: #"L(?:\[|\.plural\()"([a-z0-9_.]+)""#)
        var keys = Set<String>()
        for file in Repository.sources("Markitai") + Repository.sources("MarkitaiKit") {
            let text = try String(contentsOf: file, encoding: .utf8)
            for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                let key = String(text[Range(match.range(at: 1), in: text)!])
                keys.insert(text[Range(match.range, in: text)!].hasPrefix("L.plural") ? key + ".one" : key)
                if text[Range(match.range, in: text)!].hasPrefix("L.plural") { keys.insert(key + ".other") }
            }
        }
        return keys
    }

    @Test func everyKeyTheAppUsesIsInStringsJSONInBothLanguages() throws {
        let keys = try usedKeys()
        #expect(keys.count > 80)
        for key in keys.sorted() {
            #expect(Strings.table[key]?["en"]?.isEmpty == false, "\(key) has no English text")
            #expect(Strings.table[key]?["zh-Hans"]?.isEmpty == false, "\(key) has no zh-Hans text")
        }
    }

    /// Views take text only from strings.json: no literal reaches a label.
    @Test func noLiteralTextInTheViews() throws {
        let pattern = try NSRegularExpression(pattern: #"\b(Text|Button|Label|Toggle|Picker|Section|TextField|LabeledContent|Menu|help|accessibilityLabel)\("[^"\\]|(messageText|informativeText|placeholderString|prompt) = "[^"\\]"#)
        for file in Repository.sources("Markitai") {
            let text = try String(contentsOf: file, encoding: .utf8)
            let hits = pattern.matches(in: text, range: NSRange(text.startIndex..., in: text))
            #expect(hits.isEmpty, "\(file.lastPathComponent) shows literal text")
        }
    }

    @Test func languageChoice() {
        #expect(AppLanguage.choose(from: ["zh-Hant-TW", "en"]) == .simplifiedChinese)
        #expect(AppLanguage.choose(from: ["zh-Hans-CN"]) == .simplifiedChinese)
        #expect(AppLanguage.choose(from: ["en-GB", "zh-Hans"]) == .english)
        #expect(AppLanguage.choose(from: ["ja-JP"]) == .english)
        #expect(AppLanguage.choose(from: []) == .english)
    }

    @Test func formats() {
        #expect(Format.duration(18.6, strings: english) == "18.6 s")
        #expect(Format.duration(18.6, strings: chinese) == "18.6 秒")
        #expect(Format.duration(125, strings: english) == "2 min 5 s")
        #expect(Format.duration(125, strings: chinese) == "2 分 5 秒")
        #expect(Format.elapsed(65) == "1:05")
        #expect(Format.usd(0.0412) == "$0.0412")
        #expect(Format.usd(1.5) == "$1.50")
        let usage = Usage(requests: 6, inputTokens: 19870, outputTokens: 4440, costUSD: 0.0412, cost: .complete)
        #expect(Format.footer(duration: 18.6, usage: usage, strings: english) == "Converted in 18.6 s · 24,310 tokens · $0.0412")
        #expect(Format.footer(duration: 18.6, usage: nil, strings: english) == "Converted in 18.6 s")
        var partial = usage
        partial.cost = .partial
        #expect(Format.cost(partial, strings: english) == "at least $0.0412")
        partial.cost = .unknown
        #expect(Format.cost(partial, strings: chinese) == "费用未知")
        #expect(english.plural("state.warnings", count: 1) == "1 warning")
        #expect(english.plural("state.warnings", count: 2) == "2 warnings")
        #expect(Format.path("/Users/example/Documents/Markitai", home: "/Users/example") == "~/Documents/Markitai")
        #expect(Format.path("/Volumes/Data/out", home: "/Users/example") == "/Volumes/Data/out")
    }

    @Test func historyDates() {
        let calendar = Calendar.current
        let now = calendar.date(from: DateComponents(year: 2026, month: 11, day: 4, hour: 12))!
        let today = calendar.date(from: DateComponents(year: 2026, month: 11, day: 4, hour: 10, minute: 42))!
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!
        #expect(Format.finished(today, now: now, strings: english).hasPrefix("Today, "))
        #expect(Format.finished(yesterday, now: now, strings: chinese).hasPrefix("昨天 "))
        let older = Format.finished(calendar.date(byAdding: .day, value: -7, to: today)!, now: now, strings: english)
        #expect(older.contains("2026") && !older.hasPrefix("Today"))
    }
}
