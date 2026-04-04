# Task Breakdown: Add Swift Package Manager Support

Generated: 2026-04-04
Source: specs/feat-swift-package-manager.md

## Overview

Add a `Package.swift` manifest so PageDewarp can be consumed as an SPM dependency. Requires restructuring source directories, building a stripped OpenCV XCFramework, adding public access modifiers, and updating test resource loading.

## Phase 1: Core SPM Support

### Task 1.1: Restructure source directories for SPM layout

**Description**: Rename and reorganize `Sources/` subdirectories to match SPM conventions, and move public headers into `include/` subdirectories.
**Size**: Medium
**Priority**: High
**Dependencies**: None
**Can run parallel with**: Nothing (all other tasks depend on this)

**Implementation Steps**:

1. Rename `Sources/Core/` → `Sources/PageDewarp/` using `git mv`:
   ```bash
   git mv Sources/Core Sources/PageDewarp
   ```

2. Rename `Sources/LBFGSB/` → `Sources/CLBFGSB/`:
   ```bash
   git mv Sources/LBFGSB Sources/CLBFGSB
   ```

3. Create `include/` subdirectories and move public headers:
   ```bash
   mkdir -p Sources/OpenCVBridge/include
   git mv Sources/OpenCVBridge/OpenCVWrapper.h Sources/OpenCVBridge/include/OpenCVWrapper.h
   mkdir -p Sources/CLBFGSB/include
   git mv Sources/CLBFGSB/lbfgsb.h Sources/CLBFGSB/include/lbfgsb.h
   ```

4. Delete the unused `Sources/OpenCVBridge/OpenCVBridge.h` file (nothing references it):
   ```bash
   git rm Sources/OpenCVBridge/OpenCVBridge.h
   ```

5. Update `#import` path in `Sources/OpenCVBridge/OpenCVWrapper.mm` if it references `OpenCVWrapper.h` by relative path.

6. Update the `module.modulemap` header paths:
   ```
   framework module PageDewarp {
       header "include/OpenCVWrapper.h"
       header "include/lbfgsb.h"
       export *
   }
   ```
   Note: The modulemap needs the paths updated to reflect the new `include/` subdirectory locations. Since the modulemap references headers relative to the framework, verify the exact paths needed after the move.

**Target directory layout**:
```
Sources/
├── PageDewarp/           # Swift (was Core/)
│   ├── AlgorithmCore.swift
│   ├── AnalyticalGradient.swift
│   ├── CameraMatrix.swift
│   ├── ContourDetector.swift
│   ├── DewarpConfig.swift
│   ├── DewarpPipeline.swift
│   ├── Keypoints.swift
│   ├── LBFGSBOptimizer.swift
│   ├── Normalisation.swift
│   ├── Objective.swift
│   ├── PowellOptimizer.swift
│   ├── Projection.swift
│   ├── PureProjection.swift
│   ├── Remapper.swift
│   ├── Solver.swift
│   └── SpanAssembler.swift
├── OpenCVBridge/
│   ├── include/
│   │   └── OpenCVWrapper.h
│   ├── OpenCVWrapper.mm
│   └── module.modulemap
└── CLBFGSB/
    ├── include/
    │   └── lbfgsb.h
    ├── lbfgsb.c
    ├── linesearch.c
    ├── linpack.c
    ├── miniCBLAS.c
    ├── print.c
    ├── subalgorithms.c
    ├── timer.c
    └── LICENSE
```

**Acceptance Criteria**:
- [ ] `Sources/Core/` no longer exists; all Swift files are in `Sources/PageDewarp/`
- [ ] `Sources/LBFGSB/` no longer exists; all C files are in `Sources/CLBFGSB/`
- [ ] `Sources/OpenCVBridge/include/OpenCVWrapper.h` exists
- [ ] `Sources/CLBFGSB/include/lbfgsb.h` exists
- [ ] `Sources/OpenCVBridge/OpenCVBridge.h` is deleted
- [ ] Git history is preserved via `git mv`

---

### Task 1.2: Update project.yml for new directory layout

**Description**: Update XcodeGen's `project.yml` to reference the renamed directories so the existing CocoaPods/Xcode build continues to work.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Task 1.3

**Implementation Steps**:

Replace the `targets.PageDewarp` section in `project.yml` with:

```yaml
targets:
  PageDewarp:
    type: framework
    platform: iOS
    deploymentTarget: "16.0"
    sources:
      - path: Sources/PageDewarp
      - path: Sources/OpenCVBridge
        excludes:
          - "module.modulemap"
      - path: Sources/OpenCVBridge/include/OpenCVWrapper.h
        type: header
        headerVisibility: public
      - path: Sources/CLBFGSB
        excludes:
          - "include"
          - "LICENSE"
      - path: Sources/CLBFGSB/include/lbfgsb.h
        type: header
        headerVisibility: public
    settings:
      base:
        MODULEMAP_FILE: Sources/OpenCVBridge/module.modulemap
        HEADER_SEARCH_PATHS: ["$(SRCROOT)/Sources/CLBFGSB/include"]
        PRODUCT_BUNDLE_IDENTIFIER: com.example.pagedewarp
        GENERATE_INFOPLIST_FILE: YES
        DEFINES_MODULE: YES
        SWIFT_EMIT_LOC_STRINGS: NO
        APPLICATION_EXTENSION_API_ONLY: YES
        SKIP_INSTALL: YES
```

Also update the test target sources path if it references anything under `Sources/`:
```yaml
  PageDewarpTests:
    type: bundle.unit-test
    platform: iOS
    deploymentTarget: "16.0"
    sources:
      - path: Tests
        excludes:
          - GoldenFiles/**
      - path: Tests/GoldenFiles
        buildPhase: resources
    dependencies:
      - target: PageDewarp
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.example.pagedewarp.tests
        GENERATE_INFOPLIST_FILE: YES
```

**Acceptance Criteria**:
- [ ] `xcodegen generate` succeeds without errors
- [ ] `xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build` succeeds
- [ ] `xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test` succeeds

---

### Task 1.3: Add public access modifiers to DewarpPipeline API

**Description**: Change `DewarpPipeline` and its public-facing types from `internal` (default) to `public` so they're accessible from outside the module.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Task 1.2, Task 1.4

**Implementation Steps**:

In `Sources/PageDewarp/DewarpPipeline.swift`:

1. Change `class DewarpPipeline` → `public class DewarpPipeline`
2. Change `enum DewarpError: Error` → `public enum DewarpError: Error` and ensure all cases are visible (enum cases are automatically public when the enum is public)
3. Change `static func process(image: UIImage) -> Result<UIImage, DewarpError>` → `public static func process(image: UIImage) -> Result<UIImage, DewarpError>`

Do NOT make `DewarpConfig` public — it's internal algorithm tuning.

**Acceptance Criteria**:
- [ ] `DewarpPipeline` is `public class`
- [ ] `DewarpPipeline.DewarpError` is `public enum`
- [ ] `DewarpPipeline.process(image:)` is `public static func`
- [ ] `DewarpConfig` remains `internal` (no access modifier)
- [ ] XcodeGen build still succeeds (public modifiers are harmless in the framework build)

---

### Task 1.4: Add conditional SPM imports to Swift source files

**Description**: Add `#if SWIFT_PACKAGE` import statements to Swift files that use `OpenCVWrapper` or LBFGSB C functions, so SPM can resolve cross-target dependencies.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Task 1.2, Task 1.3

**Implementation Steps**:

Add the following block near the top of each listed file (after existing imports, before any code):

Files needing `import OpenCVBridge`:
```swift
#if SWIFT_PACKAGE
import OpenCVBridge
#endif
```

Add to these 6 files:
- `Sources/PageDewarp/ContourDetector.swift`
- `Sources/PageDewarp/SpanAssembler.swift`
- `Sources/PageDewarp/Projection.swift`
- `Sources/PageDewarp/Solver.swift`
- `Sources/PageDewarp/Remapper.swift`
- `Sources/PageDewarp/DewarpPipeline.swift`

Files needing `import CLBFGSB`:
```swift
#if SWIFT_PACKAGE
import CLBFGSB
#endif
```

Add to this 1 file:
- `Sources/PageDewarp/LBFGSBOptimizer.swift`

**Acceptance Criteria**:
- [ ] All 7 files have the correct `#if SWIFT_PACKAGE` import blocks
- [ ] XcodeGen framework build still succeeds (conditional imports are skipped)
- [ ] No duplicate imports or import ordering issues

---

### Task 1.5: Write Package.swift

**Description**: Create the `Package.swift` manifest defining the 3 targets (PageDewarp, OpenCVBridge, CLBFGSB) plus the opencv2 binary target and test target.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 1.1, Task 1.3, Task 1.4
**Can run parallel with**: Nothing (this is the integration point)

**Implementation Steps**:

Create `Package.swift` at the project root with this content:

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
        // During development, use a local path:
        //   .binaryTarget(name: "opencv2", path: "opencv2.xcframework"),
        // For release, use remote URL:
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

Notes:
- The `OpenCVBridge` target excludes `module.modulemap` (SPM generates its own)
- The `CLBFGSB` target excludes `LICENSE` (not a source file)
- During Phase 2 development, temporarily switch to `.binaryTarget(name: "opencv2", path: "opencv2.xcframework")` for local iteration

**Acceptance Criteria**:
- [ ] `Package.swift` exists at project root
- [ ] `swift package dump-package` succeeds (validates manifest syntax)
- [ ] SPM build succeeds once the OpenCV XCFramework is available locally:
  ```bash
  xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build
  ```

---

### Task 1.6: Verify XcodeGen build after all Phase 1 changes

**Description**: End-to-end verification that the existing XcodeGen + CocoaPods build still works with the restructured directories.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.2, Task 1.3, Task 1.4
**Can run parallel with**: Nothing (validation gate)

**Implementation Steps**:

```bash
# Regenerate Xcode project from updated project.yml
xcodegen generate

# Build framework
xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build

# Run tests
xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test
```

If tests fail, fix the issue — do not skip. Common issues:
- `module.modulemap` paths may need updating after header moves
- `HEADER_SEARCH_PATHS` may need adjustment for the new `include/` subdirectory

**Acceptance Criteria**:
- [ ] `xcodegen generate` produces no errors
- [ ] Framework builds successfully
- [ ] All existing tests pass (golden-file tests produce identical results)

---

## Phase 2: OpenCV XCFramework Distribution

### Task 2.1: Create build script for stripped OpenCV XCFramework

**Description**: Write `scripts/build-opencv-xcframework.sh` that builds a minimal OpenCV XCFramework with only core, imgproc, flann, features2d, calib3d.
**Size**: Medium
**Priority**: High
**Dependencies**: None (can start in parallel with Phase 1)
**Can run parallel with**: All Phase 1 tasks

**Implementation Steps**:

Create `scripts/build-opencv-xcframework.sh`:

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
    --without python

# 3. Copy the XCFramework
rm -rf "$OUTPUT_DIR"
cp -R "$WORKDIR/build/opencv2.xcframework" "$OUTPUT_DIR"

# 4. Zip and compute checksum for SPM
cd "$(dirname "$OUTPUT_DIR")"
rm -f opencv2.xcframework.zip
zip -r opencv2.xcframework.zip opencv2.xcframework
CHECKSUM=$(swift package compute-checksum opencv2.xcframework.zip)

echo ""
echo "XCFramework ready at: $OUTPUT_DIR"
echo "Checksum: $CHECKSUM"
echo ""
echo "Upload opencv2.xcframework.zip as a GitHub release asset, then update Package.swift checksum."
```

Make it executable: `chmod +x scripts/build-opencv-xcframework.sh`

**Fallback**: If `build_xcframework.py` doesn't support the `--without` flags properly, use the CMake whitelist approach documented in the spec (direct CMake with `-DBUILD_LIST=core,imgproc,flann,features2d,calib3d`).

**Acceptance Criteria**:
- [ ] Script exists at `scripts/build-opencv-xcframework.sh` and is executable
- [ ] Script runs to completion and produces `opencv2.xcframework/`
- [ ] XCFramework contains both `ios-arm64` and `ios-arm64-simulator` slices
- [ ] XCFramework size is ~15–25 MB (not the full ~90 MB)
- [ ] `swift package compute-checksum` produces a valid sha256

---

### Task 2.2: Build, upload XCFramework, and update Package.swift

**Description**: Run the build script, upload the XCFramework as a GitHub release, and update Package.swift with the real URL and checksum.
**Size**: Small
**Priority**: High
**Dependencies**: Task 2.1, Task 1.5
**Can run parallel with**: Nothing

**Implementation Steps**:

1. Run the build script:
   ```bash
   cd /Users/eryk.napierala/Projects/page-dewarp-swift
   ./scripts/build-opencv-xcframework.sh
   ```

2. Upload as GitHub release:
   ```bash
   gh release create opencv-4.10.0-minimal opencv2.xcframework.zip \
       --title "OpenCV 4.10.0 Minimal XCFramework (core+imgproc+calib3d)" \
       --notes "Stripped OpenCV 4.10.0 with only core, imgproc, flann, features2d, calib3d. ~15-25MB."
   ```

3. Update `Package.swift` with the real checksum from the build script output:
   ```swift
   .binaryTarget(
       name: "opencv2",
       url: "https://github.com/erykpiast/page-dewarp-swift/releases/download/opencv-4.10.0-minimal/opencv2.xcframework.zip",
       checksum: "<real-checksum-from-build-output>"
   ),
   ```

4. Verify from a clean state:
   ```bash
   rm -rf .build
   xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build
   ```

**Acceptance Criteria**:
- [ ] GitHub release `opencv-4.10.0-minimal` exists with the zip attached
- [ ] `Package.swift` has real checksum (not placeholder)
- [ ] Clean SPM build resolves the binary target and compiles successfully

---

## Phase 3: Test Infrastructure

### Task 3.1: Update test resource loading for SPM compatibility

**Description**: Update `Tests/TestHelpers.swift` to use `Bundle.module` when built via SPM, with fallback to `Bundle(for:)` for the XcodeGen build.
**Size**: Small
**Priority**: Medium
**Dependencies**: Task 1.5
**Can run parallel with**: Task 2.1, Task 2.2

**Implementation Steps**:

In `Tests/TestHelpers.swift`, update the two resource-loading functions to use conditional compilation:

Current code uses:
```swift
let bundle = Bundle(for: Self.self)
guard let url = bundle.url(forResource: filename, withExtension: "json") else { ... }
```

Replace with:
```swift
#if SWIFT_PACKAGE
let bundle = Bundle.module
#else
let bundle = Bundle(for: Self.self)
#endif
guard let url = bundle.url(forResource: filename, withExtension: "json") else { ... }
```

Apply the same pattern to both `loadGoldenJSON` (or equivalent) and `loadGoldenFile` helper functions in `TestHelpers.swift`.

Note: SPM resource bundles use `.copy("GoldenFiles")` which preserves the directory structure, so `subdirectory: "GoldenFiles"` may be needed in the SPM path. Verify whether `Bundle.module.url(forResource:withExtension:)` finds the files with or without the subdirectory parameter and adjust accordingly.

**Acceptance Criteria**:
- [ ] `TestHelpers.swift` uses `#if SWIFT_PACKAGE` / `Bundle.module` pattern
- [ ] XcodeGen test build still passes (the `#else` branch is used)
- [ ] SPM test build passes once OpenCV XCFramework is available:
  ```bash
  xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test
  ```

---

### Task 3.2: End-to-end SPM build and test verification

**Description**: Final validation that the complete SPM package builds, tests pass, and the XcodeGen build remains functional.
**Size**: Small
**Priority**: High
**Dependencies**: Task 2.2, Task 3.1
**Can run parallel with**: Nothing (final gate)

**Implementation Steps**:

1. Clean SPM build:
   ```bash
   rm -rf .build
   xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build
   ```

2. SPM tests:
   ```bash
   xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test
   ```

3. XcodeGen build (verify backward compatibility):
   ```bash
   xcodegen generate
   xcodebuild -scheme PageDewarp -destination 'platform=iOS Simulator,name=iPhone 16' build
   xcodebuild -scheme PageDewarpTests -destination 'platform=iOS Simulator,name=iPhone 16' test
   ```

4. Verify `swift package dump-package` still succeeds (manifest validation).

**Acceptance Criteria**:
- [ ] SPM build succeeds from clean state
- [ ] All golden-file tests pass via SPM
- [ ] XcodeGen build succeeds
- [ ] All golden-file tests pass via XcodeGen
- [ ] `swift package dump-package` succeeds

---

## Dependency Graph

```
Task 1.1 (restructure dirs)
  ├──→ Task 1.2 (update project.yml)  ──→ Task 1.6 (verify XcodeGen)
  ├──→ Task 1.3 (public access)       ──→ Task 1.6
  ├──→ Task 1.4 (conditional imports)  ──→ Task 1.6
  └──→ Task 1.5 (Package.swift)       ──→ Task 2.2 (upload + update)
                                            └──→ Task 3.2 (final verify)

Task 2.1 (build script)  ──→ Task 2.2 (upload + update)

Task 1.5 ──→ Task 3.1 (test resources) ──→ Task 3.2 (final verify)
```

## Parallel Execution Opportunities

- **Task 2.1** (build script) can run in parallel with ALL of Phase 1
- **Tasks 1.2, 1.3, 1.4** can all run in parallel after Task 1.1 completes
- **Task 3.1** can run in parallel with Task 2.1 and 2.2

## Summary

| Phase | Tasks | Key Risk |
|-------|-------|----------|
| Phase 1 | 6 tasks | module.modulemap paths after header move |
| Phase 2 | 2 tasks | OpenCV build_xcframework.py may need CMake fallback |
| Phase 3 | 2 tasks | Bundle.module subdirectory handling |
| **Total** | **10 tasks** | |
