import CMarkitaiApp
import Foundation

/// The response every `markitai_app_*` call answers.
public enum Envelope {
    case ok([String: Any])
    case failure(code: String, message: String)
}

/// The in-process markitai core (`crates/markitai-app`, DESIGN.md section 3).
/// Each call takes and answers JSON; this file is the only place that crosses
/// the C boundary.
public enum Core {
    public static let abiVersion: UInt32 = 2

    public static var abiMatches: Bool { markitai_app_abi_version() == abiVersion }

    public static var version: String { String(cString: markitai_app_version()) }

    /// Must run first in `main`, before any other thread starts.
    @discardableResult
    public static func prepareEnvironment(dataDirectory: URL) -> Envelope {
        call(markitai_app_prepare_environment, ["data_dir": dataDirectory.path])
    }

    /// Blocks until the conversion ends: call it off the main thread.
    public static func convert(_ request: [String: Any]) -> Envelope {
        call(markitai_app_convert_json, request)
    }

    public static func previewHTML(markdown: String, dark: Bool, baseDirectory: String?) -> String? {
        let request: [String: Any] = [
            "markdown": markdown,
            "theme": dark ? "dark" : "light",
            "base_dir": baseDirectory ?? NSNull(),
        ]
        guard case .ok(let result) = call(markitai_app_preview_html, request) else { return nil }
        return result["html"] as? String
    }

    /// The text of a Markdown file the core wrote without its front matter,
    /// as the conversion's `markdown` field had it (DESIGN.md 3.3).
    public static func markdownBody(_ text: String) -> String? {
        guard case .ok(let result) = call(markitai_app_markdown_body, ["markdown": text]) else { return nil }
        return result["markdown"] as? String
    }

    /// Creates the conversion's own folder under `parent`.
    public static func claimOutputFolder(parent: String, source: String) -> Result<String, CoreError> {
        switch call(markitai_app_claim_output_folder, ["parent": parent, "source": source]) {
        case .ok(let result):
            guard let path = result["path"] as? String else {
                return .failure(CoreError(code: "internal_error", message: "claim_output_folder answered no path"))
            }
            return .success(path)
        case .failure(let code, let message):
            return .failure(CoreError(code: code, message: message))
        }
    }

    public struct ConfigFile: Equatable {
        public let path: String
        public let exists: Bool
    }

    public static func configFile() -> ConfigFile? {
        var buffer = markitai_app_config_file()
        guard case .ok(let result) = take(&buffer), let path = result["path"] as? String else { return nil }
        return ConfigFile(path: path, exists: result["exists"] as? Bool ?? false)
    }

    private static func call(
        _ function: (UnsafePointer<UInt8>?, Int) -> MarkitaiAppBuffer,
        _ request: [String: Any]
    ) -> Envelope {
        guard let data = try? JSONSerialization.data(withJSONObject: request) else {
            return .failure(code: "invalid_input", message: "The request is not JSON")
        }
        var buffer = data.withUnsafeBytes { bytes in
            function(bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count)
        }
        return take(&buffer)
    }

    private static func take(_ buffer: inout MarkitaiAppBuffer) -> Envelope {
        defer { markitai_app_buffer_free(&buffer) }
        guard let bytes = buffer.data else {
            return .failure(code: "internal_error", message: "The core answered nothing")
        }
        let data = Data(bytes: bytes, count: buffer.len)
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .failure(code: "invalid_json", message: "The core answered something other than JSON")
        }
        if object["ok"] as? Bool == true {
            return .ok(object["result"] as? [String: Any] ?? [:])
        }
        let error = object["error"] as? [String: Any] ?? [:]
        return .failure(
            code: error["code"] as? String ?? "internal_error",
            message: error["message"] as? String ?? ""
        )
    }
}

public struct CoreError: Error, Equatable {
    public let code: String
    public let message: String
}
