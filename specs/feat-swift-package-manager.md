# Feature: Add Swift Package Manager Support

- **Status**: Draft
- **Authors**: Claude Code, 2026-04-04
- **Type**: Feature (distribution / developer experience)

---

## Overview

Add a `Package.swift` manifest so that `PageDewarp` can be consumed as a Swift Package dependency via SPM. This enables projects like `erykpiast/fasola` to `import PageDewarp` without CocoaPods or manual Xcode project embedding.

## Background / Problem Statement

The PageDewarp library is currently distributed only as an Xcode framework (via XcodeGen's `project.yml`) with OpenCV pulled in through CocoaPods. There is no `Package.swift`, so any consumer project must either:

1. Use CocoaPods and reference the framework via a podspec (which doesn't exist yet either), or
2. Embed the Xcode project as a submodule/workspace dependency.

SPM is the de-facto standard for Swift library distribution. Most modern iOS projects (including `fasola`) use SPM for dependency management. Without SPM support, adoption requires significant manual integration work from every consumer.

### Core challenges

The project has two non-Swift dependencies that complicate SPM integration:

1. **OpenCV** (~200MB binary framework) — used via an ObjC++ bridge (`Sources/OpenCVBridge/`). OpenCV does not ship an official SPM package. Community-maintained binary XCFramework packages exist on GitHub.

2. **L-BFGS-B C code** (`Sources/LBFGSB/`) — 7 vendored C source files with a header. This is straightforward to package as an SPM C target.

## Goals

- Add a `Package.swift` that builds `PageDewarp` as a library product
- Package the vendored LBFGSB C code as an SPM C-language target
- Integrate OpenCV as a binary XCFramework dependency
- Maintain backward compatibility with the existing XcodeGen/CocoaPods build
- Enable `import PageDewarp` from any SPM-based project
- Ensure the package works for iOS 16.0+ (matching current deployment target)

## Non-Goals

- Dropping the existing XcodeGen + CocoaPods build (both should coexist)
- Supporting macOS, tvOS, watchOS, or visionOS (iOS only for now)
- Publishing to a package registry
- Vendoring OpenCV source code into the repo
- Creating a CocoaPods podspec (separate effort)

## Technical Dependencies

### OpenCV XCFramework for SPM

OpenCV must be provided as a pre-built binary XCFramework. Options:

| Package | URL | Maintained | Notes |
|---------|-----|------------|-------|
| `opencv-spm` (AliMuqworWorya) | github.com/AliMuqworWorya/opencv-spm | Active | Binary XCFramework, ~4.10 |
| Build our own | N/A | Full control | Use OpenCV's CMake to build XCFramework, host as GitHub release |
| `OpenCV-SPM` (yeatse) | github.com/yeatse/OpenCV-SPM | Active | Wraps opencv2.xcframework |

**Recommended approach**: Build a **stripped-down** `opencv2.xcframework` containing only the modules we use, and host it as a GitHub release asset on `erykpiast/page-dewarp-swift`. This avoids a dependency on third-party SPM wrappers that may break or disappear. The binary is pinned to the exact version we test against (4.10.0).

#### Module dependency analysis

`OpenCVWrapper.mm` imports exactly three OpenCV headers:
```cpp
#import <opencv2/core.hpp>      // Mat, Scalar, Point, SVDecomp, PCACompute, ...
#import <opencv2/imgproc.hpp>   // findContours, adaptiveThreshold, remap, resize, ...
#import <opencv2/calib3d.hpp>   // solvePnP, projectPoints, Rodrigues
```

However, `calib3d` has transitive dependencies (via `ocv_define_module()` in its CMakeLists.txt):

```
calib3d → features2d → flann → core
                     → imgproc → core
```

**Minimum required module set (5 modules)**:
| Module | Why |
|--------|-----|
| `core` | Fundamental types (Mat, Scalar, Point), SVDecomp, PCACompute |
| `imgproc` | findContours, adaptiveThreshold, remap, resize, drawContours, morphology |
| `flann` | Transitive dep of features2d → calib3d |
| `features2d` | Transitive dep of calib3d |
| `calib3d` | solvePnP, projectPoints, Rodrigues |

#### Expected size reduction

| Build | Approximate size |
|-------|-----------------|
| Full OpenCV 4.10 XCFramework | ~88–90 MB |
| Stripped (5 modules, no objc bindings) | ~15–25 MB |
| opencv-mobile (extreme stripping) | ~4 MB (but lacks calib3d) |

### L-BFGS-B C library

Already vendored at `Sources/LBFGSB/`. No external dependency needed — SPM can compile C targets directly.

## Detailed Design

### Source tree restructuring

SPM enforces a convention-based directory layout. The current structure needs adjustment:

```
Sources/
├── PageDewarp/           # Swift library target (renamed from Core)
│   ├── AlgorithmCore.swift
│   ├── AnalyticalGradient.swift
│   ├── ... (all .swift files from Core/)
│   └── DewarpPipeline.swift
├── OpenCVBridge/         # ObjC++ target (mixed-language via clang)
│   ├── include/
│   │   └── OpenCVWrapper.h    # public header
│   ├── OpenCVWrapper.mm
│   └── module.modulemap       # kept for XcodeGen build only; SPM auto-generates its own
└── CLBFGSB/              # C target (renamed from LBFGSB for SPM naming)
    ├── include/
    │   └── lbfgsb.h       # public header
    ├── lbfgsb.c
    ├── linesearch.c
    ├── linpack.c
    ├── miniCBLAS.c
    ├── print.c
    ├── subalgorithms.c
    ├── timer.c
    └── LICENSE
```

Key changes:
- `Sources/Core/` → `Sources/PageDewarp/` (SPM convention: target directory name matches target name)
- `Sources/LBFGSB/` → `Sources/CLBFGSB/` (prefix with C to avoid conflicts, SPM requires `include/` subdirectory for public headers)
- `Sources/OpenCVBridge/` headers moved into `include/` subdirectory
- The existing `module.modulemap` is replaced by SPM's auto-generated one

### Package.swift

```swift
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
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
            ]
        ),

        // OpenCV binary XCFramework
        .binaryTarget(
            name: "opencv2",
            url: "https://github.com/erykpiast/page-dewarp-swift/releases/download/opencv-4.10.0-minimal/opencv2.xcframework.zip",
            checksum: "<sha256-checksum-here>"
        ),

        // Tests
        .testTarget(
            name: "PageDewarpTests",
            dependencies: ["PageDewarp"],
            path: "Tests",
            exclude: ["GoldenFiles"],
            resources: [
                .copy("GoldenFiles"),
            ]
        ),
    ]
)
```

### OpenCV XCFramework preparation

Build a stripped XCFramework containing only the 5 required modules, then upload as a GitHub release.

#### Build script (`scripts/build-opencv-xcframework.sh`)

```bash
#!/bin/bash
set -euo pipefail

OPENCV_VERSION="4.10.0"
WORKDIR="/tmp/opencv-spm-build"
OUTPUT_DIR="$(pwd)/opencv2.xcframework"

# 1. Clone OpenCV source
rm -rf "$WORKDIR"
git clone --depth 1 --branch "$OPENCV_VERSION" \
    https://github.com/opencv/opencv.git "$WORKDIR/opencv"

# 2. Build stripped XCFramework using build_xcframework.py
#    The script uses --without to exclude unwanted modules.
#    We keep: core, imgproc, flann, features2d, calib3d
#    We exclude everything else.
python3 "$WORKDIR/opencv/platforms/apple/build_xcframework.py" \
    --out "$WORKDIR/build" \
    --iphoneos_archs arm64 \
    --iphonesimulator_archs arm64 \
    --iphoneos_deployment_target 16.0 \
    --build_only_specified_archs \
    --without dnn \
    --without highgui \
    --without imgcodecs \
    --without ml \
    --without objdetect \
    --without photo \
    --without stitching \
    --without video \
    --without videoio \
    --without gapi \
    --without ts \
    --without objc \
    --without java \
    --without python \
    # DO NOT exclude flann or features2d — calib3d depends on both

# 3. Copy the XCFramework
cp -R "$WORKDIR/build/opencv2.xcframework" "$OUTPUT_DIR"

# 4. Zip and compute checksum for SPM
cd "$(dirname "$OUTPUT_DIR")"
zip -r opencv2.xcframework.zip opencv2.xcframework
swift package compute-checksum opencv2.xcframework.zip

echo ""
echo "XCFramework ready at: $OUTPUT_DIR"
echo "Upload opencv2.xcframework.zip as a GitHub release asset."
```

#### Alternative: CMake whitelist (full control, manual multi-arch)

If `build_xcframework.py` does not support fine-grained exclusion for a given OpenCV version, use CMake directly with `BUILD_LIST`:

```bash
# Per-platform build (repeat for iphoneos-arm64 and iphonesimulator-arm64)
cmake ../opencv \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../opencv/platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake \
    -DIPHONEOS_DEPLOYMENT_TARGET=16.0 \
    -DBUILD_LIST=core,imgproc,flann,features2d,calib3d \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_opencv_objc=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_IPP=OFF
cmake --build . --config Release

# Then manually create XCFramework from the two platform builds:
xcodebuild -create-xcframework \
    -framework build-iphoneos/opencv2.framework \
    -framework build-iphonesimulator/opencv2.framework \
    -output opencv2.xcframework
```

#### Upload

```bash
# Zip and compute checksum
zip -r opencv2.xcframework.zip opencv2.xcframework
CHECKSUM=$(swift package compute-checksum opencv2.xcframework.zip)
echo "Checksum: $CHECKSUM"

# Create GitHub release
gh release create opencv-4.10.0-minimal opencv2.xcframework.zip \
    --title "OpenCV 4.10.0 Minimal XCFramework (core+imgproc+calib3d)" \
    --notes "Stripped OpenCV 4.10.0 with only core, imgproc, flann, features2d, calib3d. ~15-25MB."
```

#### System frameworks to link

The stripped OpenCV still requires these system frameworks at link time (declared in `Package.swift` linkerSettings):
- `Accelerate` (BLAS/LAPACK used by core)
- `UIKit` (our bridge code uses UIImage)

### Maintaining XcodeGen compatibility

The `project.yml` build must continue to work. Two approaches:

**Option A: Symlinks** — Keep the old directory names and add symlinks for SPM. Fragile and confusing.

**Option B: Update project.yml paths** — Rename directories to the SPM layout and update `project.yml` source paths accordingly. The XcodeGen build references paths explicitly, so this is a clean change:

```yaml
# project.yml changes
targets:
  PageDewarp:
    sources:
      - path: Sources/PageDewarp         # Swift sources (was: Sources/Core via implicit Sources)
      - path: Sources/OpenCVBridge       # ObjC++ bridge (.mm + .h)
        excludes:
          - "module.modulemap"           # not a source file
      - path: Sources/OpenCVBridge/include/OpenCVWrapper.h
        type: header
        headerVisibility: public
      - path: Sources/CLBFGSB            # L-BFGS-B C sources
        excludes:
          - "include"                    # headers handled separately
          - "LICENSE"
      - path: Sources/CLBFGSB/include/lbfgsb.h
        type: header
        headerVisibility: public
    settings:
      base:
        MODULEMAP_FILE: Sources/OpenCVBridge/module.modulemap
        HEADER_SEARCH_PATHS: ["$(SRCROOT)/Sources/CLBFGSB/include"]
```

**Recommendation**: Option B. One source layout, two build systems reading from the same directories.

### ObjC++ target in SPM

SPM supports Objective-C and Objective-C++ targets. The `.mm` file extension is recognized automatically. Key considerations:

1. The `OpenCVBridge` target needs a `publicHeadersPath` pointing to `include/` so that the Swift target can import `OpenCVWrapper`.
2. SPM auto-generates a module map for the `OpenCVBridge` target, replacing the hand-written `module.modulemap`. The hand-written one is kept only for the XcodeGen build.
3. The `opencv2` binary target provides its own module map inside the XCFramework, allowing `#import <opencv2/core.hpp>` to resolve.

### Import changes in Swift code

Currently, Swift code in `Sources/Core/` accesses `OpenCVWrapper` and `lbfgsb.h` through the framework's unified module map. With SPM, these become separate modules:

```swift
// Before (framework build — everything in one module):
// OpenCVWrapper and lbfgsb are visible implicitly

// After (SPM — separate targets):
import OpenCVBridge  // for OpenCVWrapper
import CLBFGSB       // for lbfgsb functions
```

These imports need to be added to the Swift files that reference `OpenCVWrapper` or call LBFGSB C functions. Specifically:

Files needing `import OpenCVBridge`:
- `ContourDetector.swift` — findContours, moments, SVDecomp, boundingRect, maxColumnThickness
- `SpanAssembler.swift` — columnMeans, PCA, convexHull
- `Projection.swift` — projectPoints
- `Solver.swift` — solvePnP, Rodrigues
- `Remapper.swift` — resizeFloatMap, remap, adaptiveThreshold
- `DewarpPipeline.swift` — resize, createPageMask, computeDetectionMask

Files needing `import CLBFGSB`:
- `LBFGSBOptimizer.swift` — calls `setulb`

For the XcodeGen framework build, these imports would be redundant but harmless (the symbols are already visible through the framework module).

### Conditional compilation alternative

If the extra imports cause issues with the framework build, use conditional compilation:

```swift
#if SWIFT_PACKAGE
import OpenCVBridge
import CLBFGSB
#endif
```

This is a common pattern in libraries that support both SPM and Xcode project builds.

## User Experience

### Consumer integration

From a project like `fasola`:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/erykpiast/page-dewarp-swift.git", from: "1.0.0"),
],
targets: [
    .target(
        name: "Fasola",
        dependencies: [
            .product(name: "PageDewarp", package: "page-dewarp-swift"),
        ]
    ),
]
```

```swift
// Usage
import PageDewarp

let result = DewarpPipeline.process(image: someUIImage)
switch result {
case .success(let dewarped):
    // use dewarped image
case .failure(let error):
    // handle error
}
```

### Public API surface

Currently `DewarpPipeline` and other types are `internal`. For SPM consumption, the public API needs `public` access control:

- `DewarpPipeline` — `public class` (or `public enum` if made non-instantiable)
- `DewarpPipeline.process(image:)` — `public static func`
- `DewarpPipeline.DewarpError` — `public enum` with all cases `public`
- `DewarpConfig` — keep `internal` (consumers don't need to read algorithm tuning defaults)

This is a prerequisite change regardless of SPM — it's needed for any external consumption.

## Testing Strategy

### Unit tests

Since `PageDewarp` is iOS-only, `swift build` / `swift test` alone will fail on macOS. All SPM verification must go through `xcodebuild` with a simulator destination:

- **SPM build test**:
  ```bash
  xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build
  ```
- **SPM test run**:
  ```bash
  xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test
  ```
- **XcodeGen build test**: `xcodegen generate && xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build` still works after directory restructuring
- Existing golden-file tests validate that the algorithm produces identical results regardless of build system

### Integration tests

- **Binary target resolution**: Verify the OpenCV XCFramework downloads and links correctly from a clean `swift package resolve`

### Test resource handling

Golden files in `Tests/GoldenFiles/` need to be declared as SPM resources (`.copy("GoldenFiles")`). Test code that loads these files must use `Bundle.module` instead of `Bundle(for:)`:

```swift
// SPM resource access
#if SWIFT_PACKAGE
let url = Bundle.module.url(forResource: "boston_cooking_a_input", withExtension: "jpg", subdirectory: "GoldenFiles")!
#else
let bundle = Bundle(for: type(of: self))
let url = bundle.url(forResource: "boston_cooking_a_input", withExtension: "jpg")!
#endif
```

## Performance Considerations

- **Binary XCFramework download**: ~15–25 MB one-time download for the stripped OpenCV. SPM caches this, so subsequent builds are fast.
- **Build time**: SPM builds the CLBFGSB C target from source (~7 files, fast). The ObjC++ bridge is one `.mm` file. Swift compilation is the main cost.
- **No runtime performance difference** — same compiled code regardless of build system.

## Security Considerations

- The OpenCV XCFramework is hosted on our own GitHub releases, not a third-party URL. The checksum in `Package.swift` ensures integrity.
- The LBFGSB C code is already vendored and auditable in the repo.
- No new network dependencies at runtime.

## Documentation

- Update `README.md` with SPM installation instructions (Swift Package Manager section)
- Add a note about minimum Xcode version (15.0+ for the swift-tools-version)
- Document the OpenCV XCFramework build process (in a `scripts/` shell script for reproducibility)

## Implementation Phases

### Phase 1: Core SPM support

1. Restructure `Sources/` directories (Core → PageDewarp, LBFGSB → CLBFGSB with include/)
2. Move public headers into `include/` subdirectories
3. Add `public` access modifiers to `DewarpPipeline` API
4. Add conditional `#if SWIFT_PACKAGE` imports
5. Write `Package.swift` with local binary target path (for development)
6. Update `project.yml` to match new paths
7. Verify both builds work

### Phase 2: OpenCV XCFramework distribution

1. Create `scripts/build-opencv-xcframework.sh`
2. Build and verify the XCFramework
3. Upload as GitHub release
4. Update `Package.swift` with remote URL and checksum
5. Test from a clean checkout

### Phase 3: Test infrastructure

1. Update test resource loading for SPM (`Bundle.module`)
2. Verify tests pass via `xcodebuild test` with iOS simulator destination

## Open Questions

1. **OpenCV XCFramework hosting**: Should we host on our own repo's releases, or maintain a separate `opencv-xcframework` repo? Own releases are simpler; separate repo is more reusable.

2. **iOS simulator support**: The XCFramework must include both device (arm64) and simulator (arm64) slices. The CocoaPods `OpenCV-Dynamic-Framework` pod handles this automatically — we need to replicate it via `build_xcframework.py`'s `--iphonesimulator_archs` flag.

3. **build_xcframework.py vs manual CMake**: The `build_xcframework.py` script may not support the `BUILD_LIST` CMake variable directly — it uses `--without` flags for exclusion. If this proves insufficient, fall back to the manual CMake whitelist approach (see "Alternative" section). Need to verify with OpenCV 4.10.0.

## References

- [SPM documentation: Creating Swift Packages](https://developer.apple.com/documentation/xcode/creating-a-standalone-swift-package-with-xcode)
- [SPM binary targets](https://github.com/apple/swift-package-manager/blob/main/Documentation/PackageDescription.md#targets)
- [OpenCV iOS build guide](https://docs.opencv.org/4.x/d5/da3/tutorial_ios_install.html)
- Current project build: `project.yml` (XcodeGen) + `Podfile` (CocoaPods)
- OpenCV version: 4.10.0 (from `Podfile.lock`)
- Existing specs: `specs/feat-lbfgsb-optimizer.md`, `specs/feat-analytical-gradient.md`
