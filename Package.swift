// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "PageDewarp",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "PageDewarp",
            targets: ["PageDewarp"]
        ),
    ],
    dependencies: [],
    targets: [
        // Main Swift library
        .target(
            name: "PageDewarp",
            dependencies: ["OpenCVBridge", "CLBFGSB"],
            path: "Sources/PageDewarp"
        ),

        // ObjC++ bridge to OpenCV
        .target(
            name: "OpenCVBridge",
            dependencies: ["opencv2"],
            path: "Sources/OpenCVBridge",
            exclude: ["module.modulemap"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("."),
            ],
            linkerSettings: [
                .linkedFramework("UIKit"),
                .linkedFramework("Accelerate"),
            ]
        ),

        // Vendored L-BFGS-B C library
        .target(
            name: "CLBFGSB",
            path: "Sources/CLBFGSB",
            exclude: ["LICENSE"],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
            ]
        ),

        // OpenCV binary XCFramework
        .binaryTarget(
            name: "opencv2",
            url: "https://github.com/erykpiast/page-dewarp-swift/releases/download/opencv-4.10.0-minimal/opencv2.xcframework.zip",
            checksum: "cf41a1e06b0243dec59bc62dc98c9d21d5165fac193315bb1cc9431063643ec1"
        ),

        // Tests
        .testTarget(
            name: "PageDewarpTests",
            dependencies: ["PageDewarp", "OpenCVBridge"],
            path: "Tests",
            resources: [
                .copy("GoldenFiles"),
            ]
        ),
    ]
)
