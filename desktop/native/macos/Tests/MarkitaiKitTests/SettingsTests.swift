import Foundation
import Testing
@testable import MarkitaiKit

@MainActor
struct SettingsTests {
    /// `-conversion.llm YES` on the command line (the argument domain) and a
    /// hand-edited plist hold flags and numbers as text.
    @Test func flagsAndNumbersGivenAsTextAreRead() throws {
        let defaults = try #require(UserDefaults(suiteName: "dev.markitai.native.tests.\(UUID().uuidString)"))
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

    /// DESIGN.md 4.5: a missing domain, or one holding only some keys, reads
    /// every missing key as its default. The store answers a missing key as
    /// `UserDefaults` does (nil, false, 0); a suite would write the account's
    /// preferences folder, which cfprefsd keeps outside the test's HOME.
    @Test func missingKeysReadAsTheirDefaults() {
        let partial = MemoryStore()
        partial.values["conversion.ocr"] = true
        for (name, store) in [("empty", MemoryStore()), ("partial", partial)] {
            let settings = UserSettings(store: store)
            #expect(!settings.llm && !settings.screenshot && !settings.desc && !settings.alt, "\(name): flags off")
            #expect(settings.ocr == (name == "partial"), "\(name): a stored key is kept")
            #expect(settings.outputFolder == nil, "\(name): the platform's output folder")
            #expect(settings.effectiveOutputFolder == UserSettings.defaultOutputFolder)
            #expect(settings.configSource == .user, "\(name): the user's configuration")
            #expect(settings.zoom == 1, "\(name): actual size, not zero")
        }
        #expect(partial.values.count == 1, "reading writes nothing back")
    }

    @Test func outOfRangeZoomIsActualSize() throws {
        let defaults = try #require(UserDefaults(suiteName: "dev.markitai.native.tests.\(UUID().uuidString)"))
        defaults.register(defaults: ["view.zoom": 9.0])
        #expect(UserSettings(store: defaults).zoom == 1)
        #expect(UserSettings(store: MemoryStore()).zoom == 1, "no zoom stored is actual size")
    }
}
