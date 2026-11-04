import Foundation
import Testing
@testable import MarkitaiKit

@MainActor
struct HistoryTests {
    func entry(_ index: Int, folder: URL) throws -> HistoryEntry {
        let markdown = folder.appendingPathComponent("doc-\(index).md")
        try Data("# \(index)\n".utf8).write(to: markdown)
        let item = Item(source: .file("/fixtures/doc-\(index).pdf"), options: Options(llm: true), turn: index, now: Date())
        let conversion = Conversion(folder: folder.path, markdown: "# \(index)", markdownPath: markdown.path, duration: 18.6, warnings: index == 0 ? ["PDF page 7: a warning"] : [], usage: nil, urlTitle: nil)
        return HistoryEntry(item: item, conversion: conversion, finished: Date(timeIntervalSince1970: 1_793_760_138), version: "2.3.2")
    }

    func stored(_ file: URL) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
    }

    @Test func fiveHundredAndOneAdditionsKeepTheNewestFiveHundred() throws {
        let folder = try temporaryFolder("history")
        let file = folder.appendingPathComponent("history.json")
        let history = History(file: file)
        for index in 0...500 {
            history.add(try entry(index, folder: folder))
        }
        let json = try stored(file)
        let entries = try #require(json["entries"] as? [[String: Any]])
        #expect(json["version"] as? Int == 1)
        #expect(entries.count == 500)
        #expect((entries.first?["source"] as? String) == "/fixtures/doc-500.pdf", "newest first")
        #expect((entries.last?["source"] as? String) == "/fixtures/doc-1.pdf", "the oldest is dropped")
        #expect(entries.first?["finished_at"] as? String == "2026-11-04T02:42:18Z")
        #expect(entries.first?["usage"] is NSNull, "usage is null when no AI request ran")
        #expect((entries.first?["options"] as? [String: Any])?["config"] as? String == "user")
        #expect(History(file: file).entries.count == 500, "the file reads back")
        #expect(!FileManager.default.fileExists(atPath: file.path + ".tmp"))
    }

    @Test func removeAndClearLeaveTheFiles() throws {
        let folder = try temporaryFolder("history")
        let history = History(file: folder.appendingPathComponent("history.json"))
        let entries = try (0..<3).map { try entry($0, folder: folder) }
        entries.forEach(history.add)
        history.remove(id: entries[1].id)
        #expect(history.entries.map(\.id) == [entries[2].id, entries[0].id])
        history.clear()
        #expect(history.entries.isEmpty)
        #expect(try (stored(history.file)["entries"] as? [Any])?.isEmpty == true)
        for entry in entries {
            #expect(FileManager.default.fileExists(atPath: entry.markdownPath), "History never deletes converted files")
        }
    }

    @Test(arguments: ["{not json", #"{"version": 2, "entries": []}"#])
    func anUnreadableFileIsKeptAsideAndReportedOnce(_ contents: String) throws {
        let folder = try temporaryFolder("history")
        let file = folder.appendingPathComponent("history.json")
        try Data(contents.utf8).write(to: file)
        let now = Date(timeIntervalSince1970: 1_793_760_138)
        let history = History(file: file, now: now)
        #expect(history.entries.isEmpty)
        #expect(history.resetName == "history-unreadable-20261104T024218Z.json")
        let kept = folder.appendingPathComponent("history-unreadable-20261104T024218Z.json")
        #expect(try String(contentsOf: kept, encoding: .utf8) == contents, "the old file is kept, unchanged")
        let workbench = Workbench(settings: UserSettings(store: MemoryStore()), history: history)
        #expect(workbench.notices == [L["history.reset", ["name": "history-unreadable-20261104T024218Z.json"]]])
        #expect(History(file: file).resetName == nil, "reported once: the next launch starts clean")
    }
}
