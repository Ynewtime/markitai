import Foundation

/// The formats of DESIGN.md 4.2 and 6.5.
public enum Format {
    public static func duration(_ seconds: Double, strings: Strings = L) -> String {
        if seconds < 59.95 {
            return strings["unit.seconds", ["value": String(format: "%.1f", max(seconds, 0))]]
        }
        let whole = Int(seconds.rounded())
        return strings["unit.minutes", ["minutes": String(whole / 60), "seconds": String(whole % 60)]]
    }

    /// `m:ss`, for a conversion still running.
    public static func elapsed(_ seconds: TimeInterval) -> String {
        let whole = max(Int(seconds), 0)
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }

    public static func integer(_ value: Int, language: AppLanguage) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = language.locale
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }

    /// USD: four decimals below $1, two from $1.
    public static func usd(_ value: Double) -> String {
        String(format: value < 1 ? "$%.4f" : "$%.2f", value)
    }

    public static func cost(_ usage: Usage, strings: Strings = L) -> String {
        switch usage.cost {
        case .complete: return usd(usage.costUSD)
        case .partial: return strings["footer.cost.at_least", ["cost": usd(usage.costUSD)]]
        case .unknown: return strings["footer.cost.unknown"]
        }
    }

    /// `Converted in 18.6 s · 24,310 tokens · $0.0412`.
    public static func footer(duration seconds: Double, usage: Usage?, strings: Strings = L) -> String {
        var parts = [strings["footer.duration", ["duration": duration(seconds, strings: strings)]]]
        if let usage, usage.requests > 0 {
            parts.append(strings["footer.tokens", ["count": integer(usage.tokens, language: strings.language)]])
            parts.append(cost(usage, strings: strings))
        }
        return parts.joined(separator: " · ")
    }

    /// `Today, 10:42`, `Yesterday, 16:05` or the medium date and short time.
    public static func finished(_ date: Date, now: Date = Date(), strings: Strings = L) -> String {
        let locale = strings.language.locale
        var calendar = Calendar.current
        calendar.locale = locale
        let time = DateFormatter()
        time.locale = locale
        time.dateStyle = .none
        time.timeStyle = .short
        if calendar.isDate(date, inSameDayAs: now) {
            return strings["date.today", ["time": time.string(from: date)]]
        }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now), calendar.isDate(date, inSameDayAs: yesterday) {
            return strings["date.yesterday", ["time": time.string(from: date)]]
        }
        let full = DateFormatter()
        full.locale = locale
        full.dateStyle = .medium
        full.timeStyle = .short
        return full.string(from: date)
    }

    /// A path with the home folder written as `~`.
    public static func path(_ path: String, home: String = NSHomeDirectory()) -> String {
        if path == home { return "~" }
        if path.hasPrefix(home + "/") { return "~" + path.dropFirst(home.count) }
        return path
    }
}
