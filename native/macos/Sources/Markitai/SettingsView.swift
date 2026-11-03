import AppKit
import MarkitaiKit
import SwiftUI

/// The markitai configuration file: open it, or create it as `{}`.
enum ConfigurationFile {
    static func open() {
        guard let file = Core.configFile() else { return }
        let url = URL(fileURLWithPath: file.path)
        if !NSWorkspace.shared.open(url) {
            failed(url.lastPathComponent)
        }
    }

    static func create() {
        guard let file = Core.configFile() else { return }
        let url = URL(fileURLWithPath: file.path)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                try Data("{}\n".utf8).write(to: url, options: .withoutOverwriting)
            }
            open()
        } catch {
            failed(url.lastPathComponent)
        }
    }

    private static func failed(_ name: String) {
        let alert = NSAlert()
        alert.messageText = L["error.open_failed", ["name": name]]
        alert.runModal()
    }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: UserSettings
    @ViewState private var configuration = Core.configFile()
    @ViewState private var unwritable = false

    var body: some View {
        Form {
            Section(L["settings.conversion"]) {
                toggle("settings.llm", $settings.llm)
                toggle("settings.ocr", $settings.ocr)
                toggle("settings.screenshot", $settings.screenshot)
                toggle("settings.desc", $settings.desc).disabled(!settings.llm)
                toggle("settings.alt", $settings.alt).disabled(!settings.llm)
            }
            Section(L["settings.output"]) {
                LabeledContent {
                    HStack(spacing: 8) {
                        Button(L["settings.output.change"], action: chooseFolder)
                        Button(L["settings.output.show"]) {
                            let folder = URL(fileURLWithPath: settings.effectiveOutputFolder)
                            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                            NSWorkspace.shared.activateFileViewerSelecting([folder])
                        }
                    }
                    .untinted()
                } label: {
                    Text(L["settings.output.folder"])
                    Text(L["settings.output.note"])
                    Text(Format.path(settings.effectiveOutputFolder))
                        .textSelection(.enabled)
                        .truncationMode(.middle)
                }
                if unwritable {
                    Label {
                        Text(L["settings.output.unwritable"])
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    }
                }
            }
            Section(L["settings.config"]) {
                Picker(L["settings.config"], selection: $settings.configSource) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L["settings.config.user"])
                        Text(configuration.map { $0.exists ? Format.path($0.path) : L["settings.config.user.missing"] } ?? "")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .tag(ConfigSource.user)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L["settings.config.builtin"])
                        Text(L["settings.config.builtin.note"])
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .tag(ConfigSource.builtin)
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
                Group {
                    if configuration?.exists == true {
                        Button(L["settings.config.open"]) { ConfigurationFile.open() }
                    } else {
                        Button(L["settings.config.create"]) {
                            ConfigurationFile.create()
                            configuration = Core.configFile()
                        }
                    }
                }
                .untinted()
            }
            Section(L["settings.about"]) {
                HStack(spacing: 12) {
                    Image(nsImage: NSApp.applicationIconImage)
                        .resizable()
                        .frame(width: 48, height: 48)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L["app.name"]).font(.headline)
                        Text(L["settings.about.copyright"]).font(.subheadline).foregroundStyle(.secondary)
                    }
                }
                LabeledContent(L["settings.about.app"], value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? Core.version)
                LabeledContent(L["settings.about.core"], value: Core.version)
            }
        }
        .formStyle(.grouped)
        // The form scrolls: at its full height (about 990 pt) the window would
        // not fit a 13-inch screen.
        .frame(width: 520, height: 640)
        .navigationTitle(L["nav.settings"])
        .onAppear { configuration = Core.configFile() }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didBecomeKeyNotification)) { _ in
            configuration = Core.configFile()
        }
    }

    private func toggle(_ key: String, _ value: Binding<Bool>) -> some View {
        Toggle(isOn: value) {
            Text(L[key])
            Text(L[key + ".note"])
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: settings.effectiveOutputFolder)
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        if UserSettings.canWrite(folder.path) {
            unwritable = false
            settings.outputFolder = folder.path
        } else {
            unwritable = true
        }
    }
}
