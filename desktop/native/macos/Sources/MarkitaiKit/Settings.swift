import Foundation

/// Where settings persist: `UserDefaults` in the app, a dictionary in tests.
/// Flags and numbers are read with `UserDefaults`' conversions: the argument
/// domain (`-conversion.llm YES`) holds them as text.
public protocol SettingsStorage: AnyObject {
    func object(forKey key: String) -> Any?
    func bool(forKey key: String) -> Bool
    func double(forKey key: String) -> Double
    func set(_ value: Any?, forKey key: String)
}

extension UserDefaults: SettingsStorage {}

/// The settings of DESIGN.md 4.5, under their documented keys.
@MainActor
public final class UserSettings: ObservableObject {
    @Published public var llm: Bool { didSet { store.set(llm, forKey: "conversion.llm") } }
    @Published public var ocr: Bool { didSet { store.set(ocr, forKey: "conversion.ocr") } }
    @Published public var screenshot: Bool { didSet { store.set(screenshot, forKey: "conversion.screenshot") } }
    @Published public var desc: Bool { didSet { store.set(desc, forKey: "conversion.desc") } }
    @Published public var alt: Bool { didSet { store.set(alt, forKey: "conversion.alt") } }
    @Published public var outputFolder: String? { didSet { store.set(outputFolder, forKey: "output.folder") } }
    @Published public var configSource: ConfigSource { didSet { store.set(configSource.rawValue, forKey: "configuration.source") } }
    @Published public var zoom: Double { didSet { store.set(zoom, forKey: "view.zoom") } }

    public static let zoomRange = 0.5...3.0
    private let store: SettingsStorage

    public init(store: SettingsStorage) {
        self.store = store
        llm = store.bool(forKey: "conversion.llm")
        ocr = store.bool(forKey: "conversion.ocr")
        screenshot = store.bool(forKey: "conversion.screenshot")
        desc = store.bool(forKey: "conversion.desc")
        alt = store.bool(forKey: "conversion.alt")
        let folder = store.object(forKey: "output.folder") as? String
        outputFolder = folder.flatMap { $0.hasPrefix("/") ? $0 : nil }
        configSource = (store.object(forKey: "configuration.source") as? String).flatMap(ConfigSource.init(rawValue:)) ?? .user
        let zoom = store.double(forKey: "view.zoom")
        self.zoom = UserSettings.zoomRange.contains(zoom) ? zoom : 1
    }

    public var options: Options {
        Options(llm: llm, ocr: ocr, screenshot: screenshot, alt: alt, desc: desc, config: configSource)
    }

    /// The output folder, or the default `Documents/Markitai` (DESIGN.md 4.3).
    public var effectiveOutputFolder: String {
        outputFolder ?? UserSettings.defaultOutputFolder
    }

    public static var defaultOutputFolder: String {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Markitai").path
    }

    /// Whether a test file can be created and deleted in `folder`.
    public static func canWrite(_ folder: String) -> Bool {
        let probe = (folder as NSString).appendingPathComponent(".markitai-write-test-\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: probe, contents: Data()) else { return false }
        return (try? FileManager.default.removeItem(atPath: probe)) != nil
    }

    public func zoom(by step: Double) {
        zoom = min(max((zoom + step) * 100, 50), 300).rounded() / 100
    }
}
