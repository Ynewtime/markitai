import Foundation
import Testing
@testable import MarkitaiKit

enum Repository {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("../../../..")
        .standardizedFileURL

    static func fixture(_ relative: String) -> String {
        root.appendingPathComponent(relative).path
    }

    static func sources(_ target: String) -> [URL] {
        let folder = root.appendingPathComponent("native/macos/Sources/\(target)")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        return names.filter { $0.hasSuffix(".swift") }.map { folder.appendingPathComponent($0) }
    }
}

/// Every test that reaches the core runs with its own HOME and MARKITAI_HOME
/// (`build.py --check` sets them); this refuses to run otherwise.
enum Isolation {
    static func markitaiHome() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        let real = String(cString: getpwuid(getuid()).pointee.pw_dir)
        let home = try #require(environment["HOME"], "HOME is not set")
        let markitai = try #require(environment["MARKITAI_HOME"], "MARKITAI_HOME is not set: run the tests through build.py --check")
        try #require(home != real && NSHomeDirectory() != real, "HOME and CFFIXED_USER_HOME must not be the account's home")
        try #require(!markitai.hasPrefix(real + "/.markitai"), "MARKITAI_HOME must be isolated")
        return URL(fileURLWithPath: markitai)
    }
}

/// A fresh folder under the test's TMPDIR (Foundation's temporary directory
/// ignores TMPDIR).
func temporaryFolder(_ name: String = "case") throws -> URL {
    let base = ProcessInfo.processInfo.environment["TMPDIR"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
    let folder = base
        .appendingPathComponent("markitai-tests")
        .appendingPathComponent("\(name)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
}

final class MemoryStore: SettingsStorage {
    var values: [String: Any] = [:]
    func object(forKey key: String) -> Any? { values[key] }
    func bool(forKey key: String) -> Bool { values[key] as? Bool ?? false }
    func double(forKey key: String) -> Double { (values[key] as? NSNumber)?.doubleValue ?? 0 }
    func set(_ value: Any?, forKey key: String) { values[key] = value }
}

/// Waits on the main actor until `condition` holds, or fails after `seconds`.
@MainActor
func eventually(_ seconds: Double = 60, _ what: String, _ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(seconds)
    while !condition() {
        if Date() > deadline {
            Issue.record("timed out waiting for \(what)")
            throw CancellationError()
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
}

extension Item {
    var failure: Failure? {
        if case .failed(let failure) = state { return failure }
        return nil
    }

    var conversion: Conversion? {
        if case .done(let conversion) = state { return conversion }
        return nil
    }

    var isConverting: Bool {
        if case .converting = state { return true }
        return false
    }
}

/// An OpenAI-compatible chat endpoint on 127.0.0.1 that echoes the document
/// back and reports token usage, so "Enhance with AI" runs without a provider.
final class LoopbackModel: @unchecked Sendable {
    let port: UInt16
    private let socket: Int32
    private let lock = NSLock()
    private var served = 0

    var requests: Int { lock.withLock { served } }

    init() throws {
        let listener = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listener, $0, length) }
        }
        try #require(bound == 0 && listen(listener, 8) == 0, "loopback model cannot listen")
        _ = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
        }
        socket = listener
        port = UInt16(bigEndian: address.sin_port)
        Thread.detachNewThread { [self] in
            while true {
                let connection = accept(listener, nil, nil)
                guard connection >= 0 else { return }
                self.answer(connection)
            }
        }
    }

    deinit { close(socket) }

    /// `config.json` routing the default model here.
    var configuration: String {
        """
        {"llm": {"model_list": [{"model_name": "default", "litellm_params": {"model": "openai/loopback-fixture", "api_key": "LOOPBACK_TEST_KEY", "api_base": "http://127.0.0.1:\(port)/v1"}}], "router_settings": {"num_retries": 0, "timeout": 30}}}
        """
    }

    private func answer(_ connection: Int32) {
        defer { close(connection) }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)
        var bodyStart: Int?
        var length = 0
        while true {
            let count = read(connection, &buffer, buffer.count)
            guard count > 0 else { return }
            data.append(buffer, count: count)
            if bodyStart == nil, let end = data.range(of: Data("\r\n\r\n".utf8)) {
                bodyStart = end.upperBound
                let head = String(decoding: data[..<end.lowerBound], as: UTF8.self)
                for line in head.split(separator: "\r\n") where line.lowercased().hasPrefix("content-length:") {
                    length = Int(line.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)) ?? 0
                }
            }
            if let start = bodyStart, data.count - start >= length { break }
        }
        guard let start = bodyStart,
              let request = try? JSONSerialization.jsonObject(with: data[start...]) as? [String: Any],
              let messages = request["messages"] as? [[String: Any]]
        else { return }
        let user = messages.last { $0["role"] as? String == "user" }?["content"]
        let text = (user as? String) ?? "# Loopback"
        let content = (try? JSONSerialization.data(withJSONObject: [
            "cleaned_markdown": text,
            "frontmatter": ["description": "Loopback fixture", "tags": ["fixture"]],
        ])).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        let body = (try? JSONSerialization.data(withJSONObject: [
            "model": "loopback-fixture",
            "choices": [["message": ["role": "assistant", "content": content], "finish_reason": "stop"]],
            "usage": ["prompt_tokens": 1200, "completion_tokens": 340],
        ])) ?? Data()
        lock.withLock { served += 1 }
        let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        let response = Data(head.utf8) + body
        _ = response.withUnsafeBytes { write(connection, $0.baseAddress, response.count) }
    }
}
