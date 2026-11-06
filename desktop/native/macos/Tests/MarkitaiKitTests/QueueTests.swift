import Foundation
import Testing
@testable import MarkitaiKit

/// A conversion runner that claims real folders through the core and holds
/// each conversion until the test finishes it, recording what it was given.
final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private let holdsClaims: Bool
    private var permits: [String: DispatchSemaphore] = [:]
    private var waiting: [String: DispatchSemaphore] = [:]
    private var outcomes: [String: Envelope] = [:]
    private var converting: [String] = []
    private var ended: [String] = []
    private(set) var started: [String] = []
    private(set) var options: [String: Options] = [:]

    /// `holdingClaims`: each run waits for `permitClaim` before it claims its folder.
    init(holdingClaims: Bool = false) {
        holdsClaims = holdingClaims
    }

    var run: Job.Run {
        { source, options, outputFolder, claim in
            let permit = DispatchSemaphore(value: 0)
            let semaphore = DispatchSemaphore(value: 0)
            self.lock.withLock {
                self.permits[source.raw] = permit
                self.waiting[source.raw] = semaphore
                self.started.append(source.raw)
                self.options[source.raw] = options
            }
            if self.holdsClaims { permit.wait() }
            let result = Job.run(source: source, options: options, outputFolder: outputFolder, claim: claim) { _ in
                self.lock.withLock { self.converting.append(source.raw) }
                semaphore.wait()
                return self.lock.withLock { self.outcomes.removeValue(forKey: source.raw)! }
            }
            self.lock.withLock { self.ended.append(source.raw) }
            return result
        }
    }

    func startedSources() -> [String] { lock.withLock { started } }
    /// The run started for the file `name`. Runs start on a concurrent queue,
    /// so which of two starts first is not promised.
    func source(named name: String) -> String {
        lock.withLock { started.first { ($0 as NSString).lastPathComponent == name }! }
    }
    /// Sources whose folder is claimed and whose conversion is held.
    func convertingSources() -> [String] { lock.withLock { converting } }
    func endedSources() -> [String] { lock.withLock { ended } }

    func permitClaim(_ raw: String) {
        lock.withLock { _ = permits[raw]?.signal() }
    }

    func finish(_ raw: String, _ outcome: Envelope) {
        lock.withLock {
            outcomes[raw] = outcome
            waiting.removeValue(forKey: raw)?.signal()
        }
    }

    static func done(_ raw: String, warnings: [String] = []) -> Envelope {
        .ok(["markdown": "# \(raw)", "output_path": "", "duration": 1.5, "warnings": warnings])
    }

    static func failed(_ code: String, _ message: String = "") -> Envelope {
        .failure(code: code, message: message)
    }
}

@MainActor
struct QueueTests {
    let history: History
    let settings = UserSettings(store: MemoryStore())
    let output: URL

    init() throws {
        history = History(file: try temporaryFolder("queue").appendingPathComponent("history.json"))
        output = try temporaryFolder("output")
        settings.outputFolder = output.path
    }

    func folders() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: output.path)) ?? []).filter { !$0.hasPrefix(".") }.sorted()
    }

    func files(_ names: String...) throws -> [Input] {
        let folder = try temporaryFolder("inputs")
        return try names.map { name in
            let url = folder.appendingPathComponent(name)
            try Data("# \(name)\n".utf8).write(to: url)
            return .file(url)
        }
    }

    @Test func twoConvertAtOnceOldestFirstNewestShownFirst() async throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        let inputs = try files("a.md", "b.md", "c.md")
        workbench.add(inputs)
        #expect(workbench.items.map(\.title) == ["a.md", "b.md", "c.md"], "a batch keeps its order")
        try await eventually(5, "two started") { gate.startedSources().count == 2 }
        #expect(Set(gate.startedSources().map { ($0 as NSString).lastPathComponent }) == ["a.md", "b.md"])
        #expect(workbench.convertingCount == 2)
        #expect(workbench.items[2].state == .queued)

        workbench.add(try files("d.md"))
        #expect(workbench.items.first?.title == "d.md", "newest first")
        gate.finish(gate.startedSources()[0], Gate.done("a"))
        try await eventually(5, "c starts before d") { gate.startedSources().count == 3 }
        #expect((gate.startedSources()[2] as NSString).lastPathComponent == "c.md")
        #expect(workbench.items.first { $0.title == "d.md" }?.state == .queued)
        #expect(history.entries.count == 1, "a Done item is added to History")
    }

    @Test func duplicatesWhileWaitingAreRefusedFinishedOnesConvertAgain() async throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        let input = try files("same.md")
        workbench.add(input + input)
        #expect(workbench.items.count == 1)
        #expect(workbench.notices == [L["notice.duplicate", ["name": "same.md"]]])
        workbench.add(input)
        #expect(workbench.items.count == 1)
        try await eventually(5, "started") { gate.startedSources().count == 1 }
        gate.finish(gate.startedSources()[0], Gate.done("same"))
        try await eventually(5, "done") { workbench.items[0].conversion != nil }
        workbench.add(input)
        #expect(workbench.items.count == 2, "a Done source converts again as a new item")
        #expect(workbench.notices.isEmpty, "the next add clears the notice")
    }

    @Test func foldersAreRefusedNumbersPackagesAreDocuments() throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        let folder = try temporaryFolder("folders")
        let plain = folder.appendingPathComponent("Plain")
        let other = folder.appendingPathComponent("Other")
        let numbers = folder.appendingPathComponent("Budget.NUMBERS")
        for url in [plain, other, numbers] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        workbench.add([.file(plain)])
        #expect(workbench.items.isEmpty)
        #expect(workbench.notices == [L["notice.folder", ["name": "Plain"]]])
        workbench.add([.file(plain), .file(other), .file(numbers)])
        #expect(workbench.notices == [L["notice.folders", ["count": "2"]]])
        #expect(workbench.items.map(\.title) == ["Budget.NUMBERS"])
    }

    @Test func selectionFollowsTheRules() async throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        let first = try files("1.md", "2.md")
        workbench.add(first)
        #expect(workbench.selection == workbench.items[0].id, "the first item added to an empty list is selected")
        let selected = workbench.selection
        workbench.add(try files("3.md", "4.md", "5.md"))
        #expect(workbench.selection == selected, "adding to a list does not move the selection")
        let titles = workbench.items.map(\.title)
        #expect(titles == ["3.md", "4.md", "5.md", "1.md", "2.md"])

        let four = workbench.items[1]
        workbench.selection = four.id
        #expect(!workbench.canRemove(workbench.items[3].id), "a converting item cannot be removed")
        workbench.remove(four.id)
        #expect(workbench.item(workbench.selection)?.title == "5.md", "removal selects the next row")
        workbench.remove(workbench.items.first { $0.title == "5.md" }!.id)
        #expect(workbench.item(workbench.selection)?.title == "1.md")
        for source in gate.startedSources() { gate.finish(source, Gate.done(source)) }
        try await eventually(5, "both done") { workbench.items.filter { $0.conversion != nil }.count == 2 }
        workbench.selection = workbench.items.last?.id
        workbench.clearFinished()
        #expect(workbench.items.map(\.title) == ["3.md"])
        #expect(workbench.item(workbench.selection)?.title == "3.md", "a removed last row selects the previous one")
        try await eventually(5, "3 started") { gate.startedSources().count == 3 }
        gate.finish(gate.startedSources()[2], Gate.done("3"))
    }

    @Test func retryCapturesTheCurrentSettings() async throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        settings.llm = false
        settings.alt = true
        let input = try files("retry.md")
        workbench.add(input)
        try await eventually(5, "started") { gate.startedSources().count == 1 }
        let raw = gate.startedSources()[0]
        #expect(gate.options[raw]?.llm == false)
        settings.llm = true
        gate.finish(raw, Gate.failed("fetch_error", "offline"))
        try await eventually(5, "failed") { workbench.items[0].failure != nil }
        #expect(workbench.items[0].options.llm == false, "an item keeps what it captured")
        workbench.retry(workbench.items[0].id)
        try await eventually(5, "restarted") { gate.startedSources().count == 2 }
        #expect(gate.options[raw]?.llm == true, "Retry captures the current settings")
        gate.finish(raw, Gate.done(raw))
        try await eventually(5, "done") { workbench.items[0].conversion != nil }
    }

    @Test func announcementsSayDoneAndFailed() async throws {
        let gate = Gate()
        var spoken: [String] = []
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        workbench.announce = { spoken.append($0) }
        workbench.add(try files("ok.md", "bad.md"))
        try await eventually(5, "both started") { gate.startedSources().count == 2 }
        gate.finish(gate.source(named: "ok.md"), Gate.done("ok"))
        gate.finish(gate.source(named: "bad.md"), Gate.failed("unsupported"))
        try await eventually(5, "both finished") { !workbench.anyConverting }
        #expect(Set(spoken) == [L["a11y.done", ["title": "ok.md"]], L["a11y.failed", ["title": "bad.md"]]])
    }

    /// DESIGN.md 7.3: Quit deletes what the Converting items claimed, keeps
    /// what finished before it, and nothing changes after it.
    @Test func quitDeletesTheFoldersOfConvertingItemsAndKeepsFinishedOnes() async throws {
        let gate = Gate()
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        workbench.add(try files("a.md", "b.md", "c.md", "d.md"))
        try await eventually(5, "a and b claimed") { gate.convertingSources().count == 2 }
        let a = gate.source(named: "a.md")
        gate.finish(a, Gate.done("a"))
        try await eventually(5, "a done") { workbench.items.first { $0.title == "a.md" }?.conversion != nil }
        try await eventually(5, "c claimed") { gate.convertingSources().count == 3 }
        #expect(folders() == ["a", "b", "c"])

        workbench.discardConverting()
        #expect(folders() == ["a"], "the folders of b and c are gone, a's is kept")
        #expect(history.entries.map(\.title) == ["a.md"])

        for raw in gate.convertingSources() where raw != a { gate.finish(raw, Gate.done(raw)) }
        try await eventually(5, "b and c ended") { gate.endedSources().count == 3 }
        workbench.add(try files("e.md"))
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(workbench.items.filter(\.isConverting).map(\.title).sorted() == ["b.md", "c.md"], "results after Quit change nothing")
        #expect(workbench.items.filter { $0.state == .queued }.map(\.title).sorted() == ["d.md", "e.md"], "nothing starts after Quit")
        #expect(gate.startedSources().count == 3)
        #expect(history.entries.map(\.title) == ["a.md"])
        #expect(folders() == ["a"])
    }

    /// A run that reaches its claim after Quit makes no folder.
    @Test func aClaimAfterQuitMakesNoFolder() async throws {
        let gate = Gate(holdingClaims: true)
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        workbench.add(try files("late.md"))
        try await eventually(5, "started") { gate.startedSources().count == 1 }
        workbench.discardConverting()
        let raw = gate.startedSources()[0]
        gate.permitClaim(raw)
        try await eventually(5, "the run ended or claimed") { gate.endedSources().count == 1 || gate.convertingSources().count == 1 }
        #expect(gate.convertingSources().isEmpty, "no conversion ran")
        #expect(folders().isEmpty)
        gate.finish(raw, Gate.done(raw))
    }

    /// Without Quit the same held claim converts as usual.
    @Test func aHeldClaimConvertsWithoutQuit() async throws {
        let gate = Gate(holdingClaims: true)
        let workbench = Workbench(settings: settings, history: history, run: gate.run)
        workbench.add(try files("held.md"))
        try await eventually(5, "started") { gate.startedSources().count == 1 }
        gate.permitClaim(gate.startedSources()[0])
        try await eventually(5, "claimed") { gate.convertingSources().count == 1 }
        #expect(folders() == ["held"])
        gate.finish(gate.startedSources()[0], Gate.done("held"))
        try await eventually(5, "done") { workbench.items[0].conversion != nil }
        #expect(workbench.items[0].conversion?.folder == output.appendingPathComponent("held").path)
    }

    @Test(arguments: [
        ("https://example.com/article", "https://example.com/article"),
        ("  example.com/blog/post  ", "https://example.com/blog/post"),
        ("HTTP://Example.com", "HTTP://Example.com"),
        ("ftp://example.com/file", nil),
        ("not a url", nil),
        ("localhost", nil),
        ("https://", nil),
    ] as [(String, String?)])
    func webAddresses(_ text: String, _ expected: String?) {
        #expect(WebAddress.parse(text) == expected)
    }
}
