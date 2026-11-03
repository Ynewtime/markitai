// swift-tools-version: 5.10
import PackageDescription

let repository = Context.packageDirectory + "/../.."
let coreLibrary = repository + "/target/release/libmarkitai_app.a"

let package = Package(
    name: "Markitai",
    platforms: [.macOS(.v13)],
    targets: [
        .systemLibrary(name: "CMarkitaiApp", path: "Sources/CMarkitaiApp"),
        .target(
            name: "MarkitaiKit",
            dependencies: ["CMarkitaiApp"],
            resources: [.embedInCode("strings.json")],
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
        .executableTarget(name: "Markitai", dependencies: ["MarkitaiKit"]),
        .testTarget(name: "MarkitaiKitTests", dependencies: ["MarkitaiKit"]),
    ]
)
