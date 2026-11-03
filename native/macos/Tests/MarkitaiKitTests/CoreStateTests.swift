import Foundation
import Testing
@testable import MarkitaiKit

/// Every state of DESIGN.md 4.1 and every row of section 9, reached through
/// the real core with repository fixtures and authored files.
@MainActor
@Suite(.serialized)
struct CoreStateTests {
    let markitaiHome: URL
    let output: URL
    let inputs: URL

    init() throws {
        markitaiHome = try Isolation.markitaiHome()
        try? FileManager.default.removeItem(at: markitaiHome.appendingPathComponent("config.json"))
        output = try temporaryFolder("output").appendingPathComponent("Markitai")
        inputs = try temporaryFolder("inputs")
    }

    func authored(_ name: String, _ text: String = "# Native\n\nA paragraph with **bold** text.\n") throws -> URL {
        let url = inputs.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
        return url
    }

    /// Converts through a fresh Workbench and waits until nothing is waiting.
    func convert(_ inputs: [Input], configure: (UserSettings) -> Void = { _ in }) async throws -> (Workbench, [Item]) {
        let settings = UserSettings(store: MemoryStore())
        settings.outputFolder = output.path
        configure(settings)
        let history = History(file: output.deletingLastPathComponent().appendingPathComponent("history-\(UUID().uuidString).json"))
        let workbench = Workbench(settings: settings, history: history)
        workbench.add(inputs)
        try await eventually(180, "conversions finish") { !workbench.items.contains { $0.state.isActive } }
        return (workbench, workbench.items)
    }

    func folders() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: output.path)) ?? []).filter { !$0.hasPrefix(".") }.sorted()
    }

    @Test func doneWithoutWarningsOrAI() async throws {
        let source = try authored("Quarterly Report.md")
        let (workbench, items) = try await convert([.file(source)]) { $0.configSource = .builtin }
        let conversion = try #require(items.first?.conversion, "\(String(describing: items.first?.failure))")
        #expect(conversion.folder == output.appendingPathComponent("Quarterly Report").path)
        #expect(FileManager.default.fileExists(atPath: conversion.markdownPath))
        #expect(conversion.markdownPath.hasPrefix(conversion.folder + "/"))
        #expect(conversion.markdown.contains("**bold**"))
        #expect(conversion.warnings.isEmpty)
        #expect(conversion.usage == nil)
        #expect(Format.footer(duration: conversion.duration, usage: conversion.usage).hasPrefix("Converted in "))
        let entry = try #require(workbench.history.entries.first)
        #expect(entry.source == source.path, "\(entry.source)")
        #expect(entry.kind == .file && entry.title == "Quarterly Report.md")
        #expect(entry.folder == conversion.folder && entry.markdownPath == conversion.markdownPath)
        #expect(entry.options.config == .builtin && entry.usage == nil)
        #expect(entry.markitaiVersion == Core.version)
    }

    @Test func doneWithWarnings() async throws {
        let source = Repository.fixture("crates/markitai-core/src/formats/native/fixtures/textedit-word97-picture.doc")
        let (_, items) = try await convert([.file(URL(fileURLWithPath: source))]) { $0.configSource = .builtin }
        let conversion = try #require(items.first?.conversion, "\(String(describing: items.first?.failure))")
        #expect(!conversion.warnings.isEmpty)
        let subtitle = L["state.done", ["duration": Format.duration(conversion.duration)]] + " · " + L.plural("state.warnings", count: conversion.warnings.count)
        #expect(subtitle.contains("warning"))
    }

    @Test func doneWithAIUsage() async throws {
        let model = try LoopbackModel()
        try Data(model.configuration.utf8).write(to: markitaiHome.appendingPathComponent("config.json"))
        defer { try? FileManager.default.removeItem(at: markitaiHome.appendingPathComponent("config.json")) }
        let source = try authored("With AI.md", "# With AI\n\nA paragraph the loopback model returns as written.\n")
        let (workbench, items) = try await convert([.file(source)]) { $0.llm = true }
        let conversion = try #require(items.first?.conversion, "\(String(describing: items.first?.failure))")
        let usage = try #require(conversion.usage, "AI ran")
        #expect(usage.requests == model.requests && usage.requests > 0)
        #expect(usage.tokens == 1540 * usage.requests)
        #expect(conversion.markdownPath.hasSuffix(".llm.md"))
        #expect(Format.footer(duration: conversion.duration, usage: usage).contains(L["footer.tokens", ["count": Format.integer(usage.tokens, language: L.language)]]))
        let entry = try #require(workbench.history.entries.first)
        #expect(entry.usage == usage && entry.options.llm)
    }

    @Test func twoSourcesWithOneLabelGetTheirOwnFolders() async throws {
        let first = try authored("a/report.md")
        let second = try authored("b/report.md")
        let (_, items) = try await convert([.file(first), .file(second)]) { $0.configSource = .builtin }
        #expect(items.allSatisfy { $0.conversion != nil })
        #expect(folders() == ["report", "report (2)"])
    }

    enum Case: String, CaseIterable {
        case notFound, unsupported, conversionError, fetchError, imageOnly, noModel, configError
    }

    @Test(arguments: Case.allCases)
    func failedItemsLeaveNoFolder(_ failure: Case) async throws {
        var input: Input
        var configure: (UserSettings) -> Void = { $0.configSource = .builtin }
        let expected: String
        switch failure {
        case .notFound:
            input = .file(inputs.appendingPathComponent("gone.pdf"))
            expected = "not_found"
        case .unsupported:
            input = .file(try authored("blob.bin", "\u{0}\u{1}\u{2}\u{3}\u{7f}binary"))
            expected = "unsupported"
        case .conversionError:
            input = .file(try authored("broken.pdf", "%PDF-1.7\nnot really a pdf"))
            expected = "conversion_error"
        case .fetchError:
            input = .web("http://127.0.0.1:9/page")
            expected = "fetch_error"
        case .imageOnly:
            input = .file(URL(fileURLWithPath: Repository.fixture("crates/markitai-core/src/images/fixtures/heif/quadrants-121x81-orientation6.avif")))
            expected = "image_only"
        case .noModel:
            input = .file(try authored("needs-model.md"))
            configure = { $0.configSource = .builtin; $0.llm = true }
            expected = "no_model_configured"
        case .configError:
            try Data(#"{"llm": {"enabled": "sometimes"}, "output": 7}"#.utf8).write(to: markitaiHome.appendingPathComponent("config.json"))
            input = .file(try authored("config.md"))
            configure = { $0.configSource = .user }
            expected = "config_error"
        }
        defer { try? FileManager.default.removeItem(at: markitaiHome.appendingPathComponent("config.json")) }
        let (_, items) = try await convert([input], configure: configure)
        let item = try #require(items.first)
        #expect(item.failure?.code == expected, "\(String(describing: item.state))")
        #expect(item.failure?.folderWasClaimed == true)
        #expect(folders().isEmpty, "a failed conversion leaves no folder")
        #expect(!ErrorSentence.text(for: item.failure!).isEmpty)
    }

    @Test func anUnwritableOutputFolderIsOutputUnavailable() async throws {
        let blocker = try authored("not-a-folder")
        let unwritable = blocker.appendingPathComponent("Markitai").path
        let settings = UserSettings(store: MemoryStore())
        settings.outputFolder = unwritable
        let workbench = Workbench(settings: settings, history: History(file: inputs.appendingPathComponent("h.json")))
        workbench.add([.file(try authored("x.md"))])
        try await eventually(60, "failed") { workbench.items[0].failure != nil }
        #expect(workbench.items[0].failure?.code == "output_unavailable")
        #expect(workbench.items[0].failure?.folderWasClaimed == false)
        #expect(ErrorSentence.of("output_unavailable").action == .openSettings)

        settings.outputFolder = output.path
        let sentence = L["error.output_unavailable", ["folder": Format.path(unwritable), "settings": L["nav.settings"]]]
        #expect(ErrorSentence.text(for: try #require(workbench.items[0].failure)) == sentence, "the sentence names the folder the run used")
        #expect(workbench.items[0].subtitle == L["row.failed", ["reason": sentence]])
    }

    @Test func aFolderThatCannotBeWrittenIsIOError() throws {
        let source = Source.file(try authored("io.md").path)
        let result = Job.run(source: source, options: Options(config: .builtin), outputFolder: output.path, claim: { make in
            let claimed = make()
            if case .success(let folder) = claimed { chmod(folder, 0o500) }
            return claimed
        }, convert: Core.convert)
        guard case .failure(let failure) = result else { Issue.record("converted into a read-only folder"); return }
        #expect(failure.code == "io_error", "\(failure)")
        #expect(folders().isEmpty)
    }

    @Test func aDirectorySourceIsIsDirectory() throws {
        let directory = inputs.appendingPathComponent("Plain Folder")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let result = Job.run(source: .file(directory.path), options: Options(config: .builtin), outputFolder: output.path, claim: { $0() }, convert: Core.convert)
        guard case .failure(let failure) = result else { Issue.record("a folder converted"); return }
        #expect(failure.code == "is_directory", "\(failure)")
        #expect(folders().isEmpty)
    }

    @Test func anEmptySourceIsInvalidInput() throws {
        let result = Job.run(source: .file(""), options: Options(config: .builtin), outputFolder: output.path, claim: { $0() }, convert: Core.convert)
        guard case .failure(let failure) = result else { Issue.record("an empty source converted"); return }
        #expect(failure.code == "invalid_input", "\(failure)")
        #expect(ErrorSentence.of(failure.code).key == "error.invalid_input")
        #expect(folders().isEmpty)
    }

    /// Envelopes the core answers only in rare cases still map as section 9 says.
    @Test func rareEnvelopesMap() {
        let folder = output.path
        let skipped = Job.parse(.ok(["skip_reason": "exists", "markdown": ""]), folder: folder + "/x", outputFolder: folder)
        #expect(skipped == .failure(Failure(code: "skipped", message: "exists", folderWasClaimed: true, outputFolder: folder)))
        let imageSkip = Job.parse(.ok(["skip_reason": "image_only", "markdown": ""]), folder: folder + "/x", outputFolder: folder)
        #expect(imageSkip == .failure(Failure(code: "skipped", message: "image_only", folderWasClaimed: true, outputFolder: folder)), "any skip_reason is skipped")
        #expect(ErrorSentence.of("image_only") == ErrorSentence(key: "error.image_only", action: .openSettings))
        #expect(ErrorSentence.text(for: Failure(code: "skipped", message: "exists", folderWasClaimed: true, outputFolder: folder)) == L["error.skipped", ["reason": "exists"]])
        #expect(ErrorSentence.of("invalid_json").key == "error.internal")
        #expect(ErrorSentence.of("internal_error").key == "error.internal")
        #expect(ErrorSentence.of("something_new").key == "error.unknown")
        #expect(ErrorSentence.of("config_error").action == .openConfiguration)
        #expect(ErrorSentence.of("not_found").action == .remove)
        #expect(ErrorSentence.of("conversion_error").action == nil)
    }

    @Test func costStatusFollowsTheModelRows() {
        let complete = Job.usage(["requests": 2, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.5,
                                  "by_model": ["a": ["cost_status": "complete"]]])
        #expect(complete?.cost == .complete && complete?.tokens == 15)
        let partial = Job.usage(["requests": 2, "cost_usd": 0.5, "by_model": ["a": ["cost_status": "complete"], "b": ["cost_status": "unknown"]]])
        #expect(partial?.cost == .partial)
        let unknown = Job.usage(["requests": 2, "cost_usd": 0, "by_model": ["b": ["cost_status": "unknown"]]])
        #expect(unknown?.cost == .unknown)
        #expect(Job.usage(["requests": 0]) == nil, "no AI usage without requests")
    }

    @Test func previewIsOneDeterministicCoreDocument() throws {
        let markdown = "---\ntitle: Hidden\n---\n# Table\n\n| a | b |\n|---|--:|\n| 1 | 2 |\n\n- [x] done\n\nNote[^1]\n\n[^1]: Foot.\n\n<script>alert(1)</script>\n\n![remote](https://example.com/x.png) ![local](img.png)\n"
        let light = try #require(Core.previewHTML(markdown: markdown, dark: false, baseDirectory: inputs.path))
        let dark = try #require(Core.previewHTML(markdown: markdown, dark: true, baseDirectory: inputs.path))
        #expect(light.hasPrefix("<!doctype html>") && light.contains(#"data-theme="light""#))
        #expect(dark.contains(#"data-theme="dark""#))
        #expect(light == Core.previewHTML(markdown: markdown, dark: false, baseDirectory: inputs.path), "same request, same bytes")
        #expect(!light.contains("<script>") && light.contains("&lt;script&gt;"), "raw HTML shows escaped")
        #expect(!light.contains("Hidden"), "front matter is left out")
        #expect(light.contains(#"class="img-remote""#))
    }
}
