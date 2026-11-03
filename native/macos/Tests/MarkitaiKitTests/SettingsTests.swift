import Foundation
import Testing
@testable import MarkitaiKit

@MainActor
struct SettingsTests {
    /// `-conversion.llm YES` on the command line (the argument domain) and a
    /// hand-edited plist hold flags and numbers as text.
    @Test func flagsAndNumbersGivenAsTextAreRead() throws {
        let defaults = try #require(UserDefaults(suiteName: "dev.markitai.Markitai.tests.\(UUID().uuidString)"))
        defaults.register(defaults: [
            "conversion.llm": "YES",
            "conversion.ocr": "1",
            "conversion.alt": "NO",
            "view.zoom": "1.5",
        ])
        let settings = UserSettings(store: defaults)
        #expect(settings.llm, "YES turns Enhance with AI on")
        #expect(settings.ocr)
        #expect(!settings.alt && !settings.desc && !settings.screenshot)
        #expect(settings.zoom == 1.5)
        #expect(settings.options.llm, "an item added now converts with AI")
    }

    @Test func outOfRangeZoomIsActualSize() throws {
        let defaults = try #require(UserDefaults(suiteName: "dev.markitai.Markitai.tests.\(UUID().uuidString)"))
        defaults.register(defaults: ["view.zoom": 9.0])
        #expect(UserSettings(store: defaults).zoom == 1)
        #expect(UserSettings(store: MemoryStore()).zoom == 1, "no zoom stored is actual size")
    }
}
