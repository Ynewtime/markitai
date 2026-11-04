import AppKit
import MarkitaiKit
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        NavigationSplitView {
            List(selection: Binding<Page?>(get: { model.page }, set: { if let page = $0 { model.go(to: page) } })) {
                Label(L["nav.convert"], systemImage: "doc.text").tag(Page.convert)
                Label(L["nav.history"], systemImage: "clock.arrow.circlepath").tag(Page.history)
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 260)
        } detail: {
            Group {
                switch model.page {
                case .convert: ConvertPage()
                case .history: HistoryPage()
                }
            }
            .frame(minWidth: 420)
        }
        .onDrop(of: [.fileURL, .url], isTargeted: $model.dropTargeted) { providers in
            model.drop(providers)
        }
        .frame(minWidth: 640, minHeight: 480)
        .onAppear {
            model.openWindow = { openWindow(id: AppModel.windowID) }
        }
    }
}

/// The list column beside the detail, or, narrower than 760, one of them at a
/// time with a back button (DESIGN.md section 14).
struct SplitLayout<ListColumn: View, Detail: View>: View {
    @EnvironmentObject private var model: AppModel
    let hasSelection: Bool
    @ViewBuilder let list: ListColumn
    @ViewBuilder let detail: Detail

    var body: some View {
        GeometryReader { geometry in
            let stacked = geometry.size.width < 760
            Group {
                if stacked {
                    if model.showsDetailWhenStacked && hasSelection {
                        detail
                            .toolbar {
                                ToolbarItem(placement: .navigation) {
                                    Button(action: model.back) {
                                        Label(L["common.back"], systemImage: "chevron.left")
                                    }
                                    .help(L["common.back"])
                                    .keyboardShortcut("[")
                                }
                            }
                            // Esc goes back wherever focus is in the detail
                            // (`onExitCommand` needs focus inside the view).
                            .background {
                                Button(L["common.back"], action: model.back)
                                    .keyboardShortcut(.cancelAction)
                                    .opacity(0)
                                    .frame(width: 0, height: 0)
                                    .accessibilityHidden(true)
                            }
                    } else {
                        list
                    }
                } else {
                    HSplitView {
                        list.frame(minWidth: 260, idealWidth: 320, maxWidth: 420)
                        detail.frame(minWidth: 320, maxWidth: .infinity)
                    }
                }
            }
            .onAppear { model.isStacked = stacked }
            .onChange(of: stacked) { model.isStacked = $0 }
        }
    }
}
