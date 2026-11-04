import Foundation
import Testing
@testable import MarkitaiKit

/// A run whose folder claim takes `delay` on a worker, or, holding claims,
/// waits for `permitClaim`, then creates the folder, and whose conversion
/// waits for `finish`.
final class SlowClaim: @unchecked Sendable {
    private let lock = NSLock()
    private let delay: TimeInterval
    private let holdsClaims: Bool
    private let permit = DispatchSemaphore(value: 0)
    private let release = DispatchSemaphore(value: 0)
    private var claiming = false
    private var ended = false

    init(delay: TimeInterval = 0, holdingClaims: Bool = false) {
        self.delay = delay
        holdsClaims = holdingClaims
    }

    var run: Job.Run {
        { source, options, outputFolder, claim in
            Job.run(source: source, options: options, outputFolder: outputFolder, claim: { make in
                withoutActuallyEscaping(make) { make in
                    claim {
                        self.lock.withLock { self.claiming = true }
                        Thread.sleep(forTimeInterval: self.delay)
                        if self.holdsClaims { self.permit.wait() }
                        return make()
                    }
                }
            }) { _ in
                self.release.wait()
                return Gate.done(source.raw)
            }
            .map { result in
                self.lock.withLock { self.ended = true }
                return result
            }
        }
    }

    var isClaiming: Bool { lock.withLock { claiming } }
    var hasEnded: Bool { lock.withLock { ended } }
    func finish() { release.signal() }
    func permitClaim() { permit.signal() }
}

@MainActor
struct QuitTests {
    let history: History
    let settings = UserSettings(store: MemoryStore())
    let output: URL

    init() throws {
        history = History(file: try temporaryFolder("quit").appendingPathComponent("history.json"))
        output = try temporaryFolder("output")
        settings.outputFolder = output.path
    }

    func folders() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: output.path)) ?? []).filter { !$0.hasPrefix(".") }.sorted()
    }

    func file(_ name: String) throws -> Input {
        let url = try temporaryFolder("inputs").appendingPathComponent(name)
        try Data("# \(name)\n".utf8).write(to: url)
        return .file(url)
    }

    /// The main actor never waits for another item's folder claim.
    @Test func addingDoesNotWaitForAFolderClaimInFlight() async throws {
        let slow = SlowClaim(delay: 2)
        let workbench = Workbench(settings: settings, history: history, run: slow.run)
        workbench.add([try file("slow.md")])
        try await eventually(5, "the claim is in flight") { slow.isClaiming }
        let start = Date()
        workbench.add([try file("next.md")])
        let blocked = Date().timeIntervalSince(start)
        #expect(blocked < 0.5, "add() waited \(blocked) s for another item's folder claim")
        try await eventually(5, "claimed") { folders() == ["next", "slow"] }
        slow.finish()
        slow.finish()
    }

    /// DESIGN.md 7.3: Quit during a claim deletes the folder that claim makes.
    @Test func quitDeletesTheFolderOfAClaimInFlight() async throws {
        let slow = SlowClaim(holdingClaims: true)
        let workbench = Workbench(settings: settings, history: history, run: slow.run)
        workbench.add([try file("inflight.md")])
        try await eventually(5, "the claim is in flight") { slow.isClaiming }
        #expect(folders().isEmpty)
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { slow.permitClaim() }
        workbench.discardConverting()
        #expect(folders().isEmpty, "the folder made after Quit began is gone when Quit returns")
        slow.finish()
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(folders().isEmpty)
        #expect(history.entries.isEmpty)
    }

    /// DESIGN.md 7.3: a conversion that finished before Quit is kept and
    /// reaches History, also when its result has not reached the list yet.
    @Test func quitKeepsAConversionThatFinishedBeforeItsResultReachedTheList() async throws {
        let slow = SlowClaim()
        let workbench = Workbench(settings: settings, history: history, run: slow.run)
        workbench.add([try file("finished.md")])
        try await eventually(5, "claimed") { folders() == ["finished"] }
        slow.finish()
        blockMainActor(until: slow.hasEnded)
        #expect(workbench.anyConverting, "the result has not reached the list")
        workbench.discardConverting()
        #expect(folders() == ["finished"], "a finished conversion keeps its folder")
        try await eventually(5, "the result reached History") { history.entries.count == 1 }
    }

    /// Waits without yielding, so no result reaches the list meanwhile, then
    /// lets the worker that ended reach its last step.
    func blockMainActor(until condition: @autoclosure () -> Bool) {
        let deadline = Date().addingTimeInterval(5)
        while !condition() && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
        Thread.sleep(forTimeInterval: 0.2)
    }

    @Test func theQuitDialogSaysHowManyRunAndWhetherAnyWait() {
        let english = Strings(language: .english)
        let chinese = Strings(language: .simplifiedChinese)
        #expect(QuitDialog.body(converting: 1, queued: 0, strings: english) == english["quit.body.one"])
        #expect(QuitDialog.body(converting: 2, queued: 3, strings: english)
            == english["quit.body.other", ["count": "2"]] + " " + english["quit.body.queued"])
        #expect(QuitDialog.body(converting: 2, queued: 1, strings: chinese)
            == chinese["quit.body.other", ["count": "2"]] + chinese["quit.body.queued"],
            "Chinese sentences follow each other without a space")
    }
}
