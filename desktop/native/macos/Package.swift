// swift-tools-version: 5.10
import PackageDescription

let repository = Context.packageDirectory + "/../../.."
let coreLibrary = repository + "/target/release/libmarkitai_app.a"
// Complete concurrency checking; `build.py --check` turns its warnings into errors.
let strict: [SwiftSetting] = [.enableExperimentalFeature("StrictConcurrency")]

let package = Package(
    name: "Markitai",
    platforms: [.macOS(.v13)],
    targets: [
        .systemLibrary(name: "CMarkitaiApp", path: "Sources/CMarkitaiApp"),
        .target(
            name: "MarkitaiKit",
            dependencies: ["CMarkitaiApp"],
            resources: [.embedInCode("strings.json")],
            swiftSettings: strict,
            linkerSettings: [
                .unsafeFlags([coreLibrary]),
                .linkedFramework("Vision"),
                .linkedFramework("Foundation"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreFoundation"),
                .linkedLibrary("objc"),
                .linkedLibrary("iconv"),
            ]
        ),
        .executableTarget(name: "Markitai", dependencies: ["MarkitaiKit"], swiftSettings: strict),
        .testTarget(name: "MarkitaiKitTests", dependencies: ["MarkitaiKit"], swiftSettings: strict),
    ]
)
