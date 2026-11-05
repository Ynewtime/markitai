import AppKit
import MarkitaiKit
import SwiftUI

/// The menu bar (DESIGN.md section 11).
struct AppCommands: Commands {
    @ObservedObject var model: AppModel

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button(L["menu.about.native"]) {
                // Empty `.version`: AppKit otherwise repeats the build
                // number, the same 2.3.2, in brackets after the version.
                NSApp.orderFrontStandardAboutPanel(options: [
                    .applicationName: L["app.name.native"],
                    .version: "",
                    .credits: NSAttributedString(string: "\(L["settings.about.core"]) \(Core.version)"),
                ])
            }
        }
        // No help book: the default item would say Help isn't available.
        CommandGroup(replacing: .help) {}
        CommandGroup(replacing: .newItem) {
            Button(L["convert.choose"]) { model.chooseFiles() }
                .keyboardShortcut("o")
            Button(L["menu.add_url"]) { model.focusWebAddress() }
                .keyboardShortcut("l")
        }
        // After, not replacing: the save group holds Close (⌘W).
        CommandGroup(after: .saveItem) {
            Button(L["detail.save_as"]) { model.saveAs() }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                .disabled(model.shownMarkdown == nil)
            Button(L["detail.show.macos"]) { model.showInFinder() }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(!model.canShowDocument)
            Divider()
            Button(L["detail.retry"]) { model.retrySelected() }
                .keyboardShortcut("r")
                .disabled(!model.canRetry)
            Button(L[model.page == .history ? "history.remove" : "convert.remove"]) { model.removeSelected() }
                .keyboardShortcut(.delete)
                .disabled(!model.canRemove)
        }
        CommandGroup(after: .pasteboard) {
            Divider()
            Button(L["detail.copy"]) { model.copyMarkdown() }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(model.shownMarkdown == nil)
        }
        CommandGroup(after: .textEditing) {
            Button(L["history.search"]) { model.searchHistory() }
                .keyboardShortcut("f")
        }
        CommandGroup(before: .toolbar) {
            Button(L["nav.convert"]) { model.go(to: .convert) }
                .keyboardShortcut("1")
            Button(L["nav.history"]) { model.go(to: .history) }
                .keyboardShortcut("2")
            Divider()
            Button(L["menu.toggle_view"]) { model.toggleMode() }
                .keyboardShortcut("m", modifiers: [.command, .shift])
            Divider()
            Button(L["menu.zoom_in"]) { model.settings.zoom(by: 0.1) }
                .keyboardShortcut("=")
            Button(L["menu.zoom_out"]) { model.settings.zoom(by: -0.1) }
                .keyboardShortcut("-")
            Button(L["menu.actual_size"]) { model.settings.zoom = 1 }
                .keyboardShortcut("0")
            Divider()
        }
    }
}

/// `action.open_settings`: `SettingsLink` from macOS 14, the
/// `showSettingsWindow:` action before. The app menu's Settings… (⌘,) is the
/// `Settings` scene's own item.
struct OpenSettingsButton: View {
    var body: some View {
        if #available(macOS 14, *) {
            SettingsLink { Text(L["action.open_settings", ["settings": L["nav.settings"]]]) }
        } else {
            Button(L["action.open_settings", ["settings": L["nav.settings"]]]) {
                NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
            }
        }
    }
}
