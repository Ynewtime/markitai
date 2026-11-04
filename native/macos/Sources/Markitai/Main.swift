import AppKit
import MarkitaiKit
import SwiftUI

@main
@MainActor
enum Main {
    static func main() {
        localizeTheSystemMenus()
        if Core.abiMatches {
            let data = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("dev.markitai.Markitai")
            Core.prepareEnvironment(dataDirectory: data)
        }
        MarkitaiApp.main()
    }

    /// DESIGN.md 7.5: any Chinese first language shows zh-Hans, and the bundle
    /// has no zh-Hant, so AppKit's own items (Hide, Quit, Services, the
    /// panels) would fall back to English. The argument domain lasts for this
    /// process only and keeps the arguments it was launched with.
    private static func localizeTheSystemMenus() {
        let defaults = UserDefaults.standard
        let preferred = defaults.stringArray(forKey: "AppleLanguages") ?? Locale.preferredLanguages
        guard let languages = AppLanguage.bundleLanguages(preferred) else { return }
        var arguments = defaults.volatileDomain(forName: UserDefaults.argumentDomain)
        arguments["AppleLanguages"] = languages
        defaults.setVolatileDomain(arguments, forName: UserDefaults.argumentDomain)
    }
}

struct MarkitaiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @ObservedObject private var model = AppModel.shared

    var body: some Scene {
        Window(L["app.name"], id: AppModel.windowID) {
            Group {
                if model.abiMatches {
                    RootView()
                } else {
                    FatalView()
                }
            }
            .environmentObject(model)
            .environmentObject(model.workbench)
            .environmentObject(model.settings)
            .environmentObject(model.history)
            .tint(Brand.tint)
        }
        .defaultSize(width: 1200, height: 780)
        .windowResizability(.contentMinSize)
        .commands { AppCommands(model: model) }

        Settings {
            SettingsView()
                .environmentObject(model)
                .environmentObject(model.settings)
                .tint(Brand.tint)
        }
    }
}

enum Brand {
    /// Ink 900 in light, 500 in dark (DESIGN.md 13.2).
    static let tint = Color(nsColor: NSColor(name: "MarkitaiTint") { appearance in
        let dark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        return dark
            ? NSColor(srgbRed: 0x71 / 255, green: 0x71 / 255, blue: 0x7a / 255, alpha: 1)
            : NSColor(srgbRed: 0x18 / 255, green: 0x18 / 255, blue: 0x1b / 255, alpha: 1)
    })
}

extension View {
    /// A push button in the system's colors. Under a tint, dark mode draws a
    /// push button's title in that tint, and Ink 500 on a dark window is below
    /// 4.5:1 (DESIGN.md 13.2: never draw text in the tint). Prominent and
    /// default buttons, switches and radios keep the tint.
    func untinted() -> some View { tint(Color?.none) }
}

struct FatalView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 64, height: 64)
            Text(L["fatal.abi.title"]).font(.title2.weight(.semibold))
            Text(L["fatal.abi.body"]).multilineTextAlignment(.center)
            Button(L["quit.confirm"]) { NSApp.terminate(nil) }
                .keyboardShortcut(.defaultAction)
        }
        .padding(32)
        .frame(minWidth: 640, maxWidth: .infinity, minHeight: 480, maxHeight: .infinity)
    }
}

/// SwiftUI's `State` property wrapper. In this SDK `@State` names a macro whose
/// plugin ships only with Xcode, so the views spell the wrapper through this alias.
typealias ViewState<Value> = SwiftUI.State<Value>
