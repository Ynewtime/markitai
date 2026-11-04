import AppKit
import Combine
import MarkitaiKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var pendingFiles: [URL] = []
    private var launched = false
    private var keyMonitor: Any?
    private var quitDialogFollowing: AnyCancellable?

    func applicationDidFinishLaunching(_ notification: Notification) {
        launched = true
        Task { _ = await PreviewNetworkRules.compiled() }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            MainActor.assumeIsolated { Self.deletesLineInTextField(event) } ? nil : event
        }
        if !pendingFiles.isEmpty {
            AppModel.shared.add(pendingFiles.map(Input.file))
            pendingFiles = []
        }
    }

    /// Files dropped on the Dock icon or opened with the app.
    func application(_ application: NSApplication, open urls: [URL]) {
        let files = urls.filter(\.isFileURL)
        guard launched else {
            pendingFiles += files
            return
        }
        AppModel.shared.add(files.map(Input.file))
        AppModel.shared.showWindow()
    }

    /// Paste with no text field focused adds the clipboard's files or web
    /// address (DESIGN.md 11). The application delegate is the last responder
    /// that sees `paste:`; a focused field takes it before.
    @objc func paste(_ sender: Any?) {
        AppModel.shared.pasteFromClipboard()
    }

    /// ⌘⌫ is Remove (DESIGN.md 11) except while a text field is being edited,
    /// where it deletes to the start of the line, as in Finder. Menu key
    /// equivalents run before the field sees a key, so the field gets this one
    /// here first.
    private static func deletesLineInTextField(_ event: NSEvent) -> Bool {
        guard event.keyCode == 51,
              event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command,
              let editor = event.window?.firstResponder as? NSTextView, editor.isEditable
        else { return false }
        editor.deleteToBeginningOfLine(nil)
        return true
    }

    /// Closing the window keeps conversions running; the Dock icon reopens it.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows {
            AppModel.shared.showWindow()
        }
        return true
    }

    /// Every quit path asks first while a conversion runs (DESIGN.md 6.7, 7.3).
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let workbench = AppModel.shared.workbench
        let running = workbench.convertingCount
        guard running > 0 else { return .terminateNow }
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = L["quit.title"]
        alert.informativeText = QuitDialog.body(converting: running, queued: workbench.queuedCount)
        let quit = alert.addButton(withTitle: L["quit.confirm"])
        let cancel = alert.addButton(withTitle: L["common.cancel"])
        quit.keyEquivalent = ""
        cancel.keyEquivalent = "\r"
        quit.hasDestructiveAction = true
        let sheetParent = NSApp.windows.first(where: { $0.isVisible && $0.canBecomeMain })
        // The count follows the queue while the dialog is up. Once nothing
        // runs, nothing would be lost: the dialog closes and the app quits.
        quitDialogFollowing = workbench.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { _ in
                let running = workbench.convertingCount
                guard running == 0 else {
                    alert.informativeText = QuitDialog.body(converting: running, queued: workbench.queuedCount)
                    return
                }
                if let sheetParent {
                    sheetParent.endSheet(alert.window, returnCode: .alertFirstButtonReturn)
                } else {
                    NSApp.stopModal(withCode: .alertFirstButtonReturn)
                }
            }
        let answer = { (response: NSApplication.ModalResponse) in
            self.quitDialogFollowing = nil
            let quit = response == .alertFirstButtonReturn
            if quit {
                workbench.discardConverting()
            }
            sender.reply(toApplicationShouldTerminate: quit)
        }
        if let sheetParent {
            alert.beginSheetModal(for: sheetParent, completionHandler: answer)
        } else {
            answer(alert.runModal())
        }
        return .terminateLater
    }
}
