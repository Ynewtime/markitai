import AppKit
import MarkitaiKit
import SwiftUI
import WebKit

struct DetailHeader<Status: View>: View {
    let title: String
    let source: Source
    @ViewBuilder let status: Status

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.title3.weight(.semibold))
                .lineLimit(2)
                .textSelection(.enabled)
            HStack(spacing: 6) {
                KindGlyph(kind: source.kind)
                Text(source.display)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
            status.font(.subheadline)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 20)
        .padding(.top, 4)
        .padding(.bottom, 12)
    }
}

/// A finished conversion: header, warnings, Preview or Markdown, footer.
struct DocumentView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var settings: UserSettings
    let document: Document

    var body: some View {
        let loaded = model.markdown(for: document)
        VStack(alignment: .leading, spacing: 0) {
            DetailHeader(title: document.title, source: document.source) {
                Label {
                    Text(status)
                } icon: {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                }
            }
            if !document.warnings.isEmpty {
                WarningsBox(id: document.id, warnings: document.warnings)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 10)
            }
            Divider()
            switch loaded {
            case .loading:
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            case .missing:
                VStack(spacing: 8) {
                    Text(L["detail.missing.title"]).font(.title3.weight(.semibold))
                    Text(L["detail.missing.body", ["path": Format.path(document.markdownPath)]])
                        .multilineTextAlignment(.center)
                        .textSelection(.enabled)
                }
                .padding(20)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .text(let markdown):
                Group {
                    switch model.mode {
                    case .preview:
                        PreviewPane(markdown: markdown, baseDirectory: document.folder, zoom: settings.zoom)
                    case .markdown:
                        MarkdownText(text: markdown, zoom: settings.zoom)
                            .accessibilityLabel(L["a11y.markdown_text"])
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                Text(Format.footer(duration: document.duration, usage: document.usage))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .textSelection(.enabled)
                    .padding(.horizontal, 20)
                    .frame(height: 30)
            }
        }
        .onAppear { if document.finished != nil { model.load(document) } }
        .onChange(of: document.id) { _ in if document.finished != nil { model.load(document) } }
    }

    private var status: String {
        let done = L["state.done", ["duration": Format.duration(document.duration)]]
        guard let finished = document.finished else { return done }
        return done + " · " + Format.finished(finished)
    }
}

struct WarningsBox: View {
    @EnvironmentObject private var workbench: Workbench
    let id: String
    let warnings: [String]

    var body: some View {
        DisclosureGroup(isExpanded: Binding(
            get: { workbench.expandedWarnings.contains(id) },
            set: { open in
                if open { workbench.expandedWarnings.insert(id) } else { workbench.expandedWarnings.remove(id) }
            }
        )) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(warnings.enumerated()), id: \.offset) { _, warning in
                    Text(warning)
                        .font(.subheadline)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.top, 6)
        } label: {
            Label {
                Text(L.plural("state.warnings", count: warnings.count)).font(.subheadline.weight(.medium))
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.quaternary))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(L["a11y.warnings"])
    }
}

/// The detail bar: view switch centered, document actions trailing.
struct DocumentToolbar: ViewModifier {
    @EnvironmentObject private var model: AppModel

    func body(content: Content) -> some View {
        content.toolbar {
            if model.document != nil && !(model.isStacked && !model.showsDetailWhenStacked) {
                ToolbarItem(placement: .principal) {
                    Picker(L["detail.view"], selection: $model.mode) {
                        Text(L["detail.preview"]).tag(DetailMode.preview)
                        Text(L["detail.markdown"]).tag(DetailMode.markdown)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .accessibilityLabel(L["detail.view"])
                    .help(L["menu.toggle_view"])
                }
                ToolbarItemGroup(placement: .primaryAction) {
                    Button(action: model.copyMarkdown) {
                        Label(L["detail.copy"], systemImage: model.copiedID == model.document?.id ? "checkmark" : "doc.on.doc")
                    }
                    .help(L["detail.copy"])
                    .disabled(model.shownMarkdown == nil)
                    Button(action: model.saveAs) {
                        Label(L["detail.save_as"], systemImage: "square.and.arrow.down")
                    }
                    .help(L["detail.save_as"])
                    .disabled(model.shownMarkdown == nil)
                    Button(action: model.showInFinder) {
                        Label(L["detail.show.macos"], systemImage: "folder")
                    }
                    .help(L["detail.show.macos"])
                    .disabled(!model.canShowDocument)
                }
            }
        }
    }
}

extension View {
    func documentToolbar() -> some View { modifier(DocumentToolbar()) }
}

/// The core's preview document in a `WKWebView` (DESIGN.md section 8),
/// rendered off the main thread whenever the Markdown or appearance changes.
struct PreviewPane: View {
    @Environment(\.colorScheme) private var colorScheme
    let markdown: String
    let baseDirectory: String
    let zoom: Double
    @ViewState private var html: String?

    var body: some View {
        PreviewWebView(html: html, baseDirectory: baseDirectory, zoom: zoom)
            .task(id: RenderKey(markdown: markdown, dark: colorScheme == .dark, base: baseDirectory)) {
                let dark = colorScheme == .dark
                let markdown = markdown
                let base = baseDirectory
                html = await Task.detached(priority: .userInitiated) {
                    Core.previewHTML(markdown: markdown, dark: dark, baseDirectory: base)
                }.value
            }
    }

    private struct RenderKey: Hashable {
        let markdown: String
        let dark: Bool
        let base: String
    }
}

struct PreviewWebView: NSViewRepresentable {
    let html: String?
    let baseDirectory: String
    let zoom: Double

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> PreviewWKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = false
        PreviewNetworkRules.apply(to: configuration.userContentController)
        let view = PreviewWKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.setValue(false, forKey: "drawsBackground")
        view.allowsMagnification = false
        view.setAccessibilityLabel(L["a11y.preview"])
        return view
    }

    func updateNSView(_ view: PreviewWKWebView, context: Context) {
        if view.pageZoom != zoom { view.pageZoom = zoom }
        guard let html, context.coordinator.loaded != html else { return }
        context.coordinator.loaded = html
        context.coordinator.loading = true
        view.loadHTMLString(html, baseURL: URL(fileURLWithPath: baseDirectory, isDirectory: true))
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var loaded: String?
        var loading = false

        /// Allows the document's own load and same-document fragment links;
        /// opens http, https and mailto links in their apps; cancels the rest.
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            if loading && action.navigationType == .other && action.targetFrame?.isMainFrame == true {
                loading = false
                return decisionHandler(.allow)
            }
            guard let url = action.request.url else { return decisionHandler(.cancel) }
            if url.fragment != nil, let current = webView.url, Self.withoutFragment(url) == Self.withoutFragment(current) {
                return decisionHandler(.allow)
            }
            if ["http", "https", "mailto"].contains(url.scheme?.lowercased() ?? "") {
                NSWorkspace.shared.open(url)
            }
            decisionHandler(.cancel)
        }

        private static func withoutFragment(_ url: URL) -> String {
            url.absoluteString.components(separatedBy: "#")[0]
        }
    }
}

/// A content rule list blocking every http(s) and ws(s) request, compiled
/// once and added to each preview as soon as it is ready.
@MainActor
enum PreviewNetworkRules {
    private static var list: WKContentRuleList?
    private static var waiting: [WKUserContentController] = []
    private static var compiling = false

    static func apply(to controller: WKUserContentController) {
        if let list {
            controller.add(list)
            return
        }
        waiting.append(controller)
        guard !compiling else { return }
        compiling = true
        let json = #"[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},{"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}}]"#
        WKContentRuleListStore.default().compileContentRuleList(forIdentifier: "MarkitaiPreviewNetwork", encodedContentRuleList: json) { compiled, _ in
            Task { @MainActor in
                list = compiled
                if let compiled { waiting.forEach { $0.add(compiled) } }
                waiting = []
            }
        }
    }
}

/// Keeps only Copy in the preview's context menu.
final class PreviewWKWebView: WKWebView {
    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        for item in menu.items where item.identifier?.rawValue != "WKMenuItemIdentifierCopy" {
            menu.removeItem(item)
        }
        super.willOpenMenu(menu, with: event)
    }
}

/// The Markdown text: not editable, selectable, monospaced, wrapped, inset 20.
struct MarkdownText: NSViewRepresentable {
    let text: String
    let zoom: Double

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        guard let view = scroll.documentView as? NSTextView else { return scroll }
        view.isEditable = false
        view.isSelectable = true
        view.isRichText = false
        view.drawsBackground = false
        scroll.drawsBackground = false
        view.textContainerInset = NSSize(width: 20, height: 20)
        view.textContainer?.widthTracksTextView = true
        view.isAutomaticLinkDetectionEnabled = false
        view.setAccessibilityLabel(L["a11y.markdown_text"])
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let view = scroll.documentView as? NSTextView else { return }
        let font = NSFont.monospacedSystemFont(ofSize: 12 * zoom, weight: .regular)
        if view.string != text || view.font != font {
            view.string = text
            view.font = font
            view.textColor = .labelColor
        }
    }
}
