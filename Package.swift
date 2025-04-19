// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LLMCLI",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.13"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        // Shared library that contains ModelPipeline, KVCacheProcessor, etc.
        .target(
            name: "LLMKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Kit"
        ),

        // Original one‑shot benchmarking CLI
        .executableTarget(
            name: "LLMCLI",
            dependencies: [
                "LLMKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/LLMCLI"
        ),

        // Interactive chat CLI
        .executableTarget(
            name: "ChatCLI",
            dependencies: [
                "LLMKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/ChatCLI"
        ),
    ]
)