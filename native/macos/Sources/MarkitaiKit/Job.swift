import Foundation

/// One run of an item, DESIGN.md 4.1 steps 1 to 5: claim the folder, convert,
/// map the envelope, and delete the folder of a failed run.
public enum Job {
    public typealias Run = @Sendable (_ source: Source, _ options: Options, _ outputFolder: String, _ claimed: @Sendable (String) -> Void) -> Result<Conversion, Failure>

    public static let core: Run = { source, options, outputFolder, claimed in
        run(source: source, options: options, outputFolder: outputFolder, claimed: claimed, convert: Core.convert)
    }

    static func run(
        source: Source,
        options: Options,
        outputFolder: String,
        claimed: (String) -> Void,
        convert: ([String: Any]) -> Envelope
    ) -> Result<Conversion, Failure> {
        let folder: String
        switch Core.claimOutputFolder(parent: outputFolder, source: source.raw) {
        case .success(let path): folder = path
        case .failure(let error):
            return .failure(Failure(code: "output_unavailable", message: error.message, folderWasClaimed: false))
        }
        claimed(folder)
        let outcome = parse(convert(request(source: source, options: options, folder: folder)), folder: folder)
        if case .failure = outcome {
            try? FileManager.default.removeItem(atPath: folder)
        }
        return outcome
    }

    /// The `convert_json` request (DESIGN.md 4.1 step 2).
    public static func request(source: Source, options: Options, folder: String) -> [String: Any] {
        var fields: [String: Any] = [
            "output_dir": folder,
            "llm": options.llm,
            "ocr": options.ocr,
            "screenshot": options.screenshot,
            "alt": options.llm && options.alt,
            "desc": options.llm && options.desc,
        ]
        if options.config == .builtin {
            fields["config"] = [String: Any]()
        }
        return ["source": source.raw, "options": fields]
    }

    /// Step 4: the envelope as Done or Failed. Any `skip_reason` is `skipped`;
    /// an error envelope keeps its code (the core answers `image_only` itself).
    public static func parse(_ envelope: Envelope, folder: String) -> Result<Conversion, Failure> {
        switch envelope {
        case .failure(let code, let message):
            return .failure(Failure(code: code, message: message, folderWasClaimed: true))
        case .ok(let result):
            if let skip = result["skip_reason"] as? String {
                return .failure(Failure(code: "skipped", message: skip, folderWasClaimed: true))
            }
            let markdown = nonEmpty(result["llm_markdown"]) ?? (result["markdown"] as? String ?? "")
            let path = nonEmpty(result["llm_output_path"]) ?? (result["output_path"] as? String ?? "")
            return .success(Conversion(
                folder: folder,
                markdown: markdown,
                markdownPath: path,
                duration: (result["duration"] as? NSNumber)?.doubleValue ?? 0,
                warnings: result["warnings"] as? [String] ?? [],
                usage: usage(result["usage"]),
                urlTitle: title(result["frontmatter"])
            ))
        }
    }

    private static func nonEmpty(_ value: Any?) -> String? {
        guard let text = value as? String, !text.isEmpty else { return nil }
        return text
    }

    /// DESIGN.md 4.2: AI ran when `requests > 0`; the cost is complete only
    /// when every model row is.
    static func usage(_ value: Any?) -> Usage? {
        guard let usage = value as? [String: Any] else { return nil }
        let int = { (key: String) in (usage[key] as? NSNumber)?.intValue ?? 0 }
        let requests = int("requests")
        guard requests > 0 else { return nil }
        let cost = (usage["cost_usd"] as? NSNumber)?.doubleValue ?? 0
        let rows = (usage["by_model"] as? [String: Any] ?? [:]).values.compactMap { $0 as? [String: Any] }
        let status: CostStatus
        if rows.allSatisfy({ $0["cost_status"] as? String == "complete" }) {
            status = .complete
        } else if cost > 0 {
            status = .partial
        } else {
            status = .unknown
        }
        return Usage(requests: requests, inputTokens: int("input_tokens"), outputTokens: int("output_tokens"), costUSD: cost, cost: status)
    }

    /// A URL item's title: the front matter's, trimmed, at most 200 characters.
    static func title(_ frontmatter: Any?) -> String? {
        guard let title = (frontmatter as? [String: Any])?["title"] as? String else { return nil }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : String(trimmed.prefix(200))
    }
}
