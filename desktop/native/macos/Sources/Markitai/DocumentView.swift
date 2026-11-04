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
                        PreviewPane(markdown: markdown, file: document.markdownPath, baseDirectory: document.folder, zoom: settings.zoom)
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
        .task(id: document.id) { if document.finished != nil { await model.load(document) } }
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
/// The web view exists only once the network rule list is compiled, so no
/// document loads without it. A failed render shows `preview.failed` and is
/// not kept: the pane renders again the next time it shows.
struct PreviewPane: View {
    @Environment(\.colorScheme) private var colorScheme
    let markdown: String
    /// The Markdown file in `baseDirectory`, loaded first for read access.
    let file: String
    let baseDirectory: String
    let zoom: Double
    @ViewState private var rendered: Rendered?
    @ViewState private var rules: PreviewNetworkRules.Compiled?

    var body: some View {
        Group {
            if case .failed = rendered {
                Text(L["preview.failed"])
                    .multilineTextAlignment(.center)
                    .padding(20)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let rules {
                PreviewWebView(html: rendered?.html, file: file, baseDirectory: baseDirectory, zoom: zoom, rules: rules)
            } else {
                Color.clear
            }
        }
        .task { rules = await PreviewNetworkRules.compiled() }
        .task(id: RenderKey(markdown: markdown, dark: colorScheme == .dark, base: baseDirectory)) {
            let dark = colorScheme == .dark
            let markdown = markdown
            let base = baseDirectory
            let html = await Task.detached(priority: .userInitiated) {
                Core.previewHTML(markdown: markdown, dark: dark, baseDirectory: base)
            }.value
            rendered = html.map(Rendered.html) ?? .failed
        }
    }

    private enum Rendered {
        case html(String)
        case failed

        var html: String? {
            guard case .html(let html) = self else { return nil }
            return html
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
    let file: String
    let baseDirectory: String
    let zoom: Double
    let rules: PreviewNetworkRules.Compiled

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> PreviewWKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = false
        if let list = rules.list {
            configuration.userContentController.add(list)
        }
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
        context.coordinator.show(html, file: file, folder: URL(fileURLWithPath: baseDirectory, isDirectory: true), in: view)
    }

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate {
        var loaded: String?
        var loading = false
        /// The folder the page may read, and the document waiting for that
        /// grant while the file load that gives it runs.
        private var granted: URL?
        private var grant: WKNavigation?
        private var waiting: String?

        /// `loadHTMLString` gives the page no read access to the result
        /// folder, so its images would not load. Loading the Markdown file with
        /// `allowingReadAccessTo:` its folder grants it, and WebKit keeps it
        /// for later documents whose base is that folder. The view stays
        /// hidden until the document replaces the file.
        func show(_ html: String, file: String, folder: URL, in view: WKWebView) {
            loaded = html
            if granted == folder, grant != nil {
                waiting = html
                return
            }
            loading = true
            if granted == folder {
                view.loadHTMLString(html, baseURL: folder)
                return
            }
            granted = folder
            waiting = html
            view.isHidden = true
            grant = view.loadFileURL(URL(fileURLWithPath: file), allowingReadAccessTo: folder)
        }

        private func ended(_ navigation: WKNavigation!, in webView: WKWebView) {
            guard navigation == grant, let html = waiting, let folder = granted else {
                // A load that ends while another folder's grant runs, such as
                // the grant it replaced, leaves the view hidden for that one.
                if grant == nil { webView.isHidden = false }
                return
            }
            grant = nil
            waiting = nil
            loading = true
            webView.loadHTMLString(html, baseURL: folder)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            ended(navigation, in: webView)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            ended(navigation, in: webView)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            ended(navigation, in: webView)
        }

        /// Allows the document's own load and same-document fragment links;
        /// opens http, https and mailto links in their apps; cancels the rest.
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void) {
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
/// once (the app starts it at launch) and added to every preview.
@MainActor
enum PreviewNetworkRules {
    /// The compiled list; nil only if WebKit refused it, and then the
    /// document's own CSP still blocks the network.
    struct Compiled {
        let list: WKContentRuleList?
    }

    private static var compiling: Task<Compiled, Never>?

    static func compiled() async -> Compiled {
        if let compiling { return await compiling.value }
        let task = Task {
            let json = #"[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},{"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}}]"#
            let list = try? await WKContentRuleListStore.default().compileContentRuleList(forIdentifier: "MarkitaiPreviewNetwork", encodedContentRuleList: json)
            return Compiled(list: list)
        }
        compiling = task
        return await task.value
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
