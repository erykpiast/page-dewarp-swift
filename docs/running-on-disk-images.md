# Running the Swift Pipeline on Arbitrary Disk Images

The PageDewarp library is iOS-only (requires UIKit + OpenCV xcframework). It cannot
be invoked as a macOS command-line tool. To process images from disk, use a one-off
XCTest that runs in the iOS Simulator.

## Prerequisites

```bash
xcodegen generate && pod install
```

## How It Works

1. Copy the input image to `/tmp/` (the simulator shares the host's `/tmp` on Apple Silicon).
2. Create a test file in `Tests/` that loads from `/tmp`, runs the pipeline, writes output to `/tmp`.
3. Run with `xcodebuild test` targeting the specific test.
4. Copy results from `/tmp` back to the desired location.

## Example: Process a Single Image

### 1. Write the test file

Create `Tests/RunDewarpOnFile.swift`:

```swift
import XCTest
@testable import PageDewarp

class RunDewarpOnFile: XCTestCase {
    func testDewarpExternalImage() throws {
        let inputPath = "/tmp/dewarp_input.jpeg"
        let outputPath = "/tmp/dewarp_output.png"

        guard let image = UIImage(contentsOfFile: inputPath) else {
            XCTFail("Could not load image at \(inputPath)")
            return
        }

        let result = DewarpPipeline.process(image: image)

        switch result {
        case .success(let output):
            let pngData = output.pngData()!
            try pngData.write(to: URL(fileURLWithPath: outputPath))
            print("Dewarped image written to \(outputPath)")
        case .failure(let error):
            XCTFail("Pipeline failed: \(error)")
        }
    }
}
```

### 2. Regenerate the Xcode project (to pick up the new test file)

```bash
xcodegen generate && pod install
```

### 3. Copy input and run

```bash
cp ~/path/to/photo.jpeg /tmp/dewarp_input.jpeg

xcodebuild test \
  -workspace PageDewarp.xcworkspace \
  -scheme PageDewarp \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:PageDewarpTests/RunDewarpOnFile/testDewarpExternalImage

cp /tmp/dewarp_output.png ~/path/to/output.png
```

## Batch Processing

To process multiple images, write one test method per image or loop in a single test:

```swift
func testDewarpBatch() throws {
    let files = ["IMG_1389", "IMG_1369", "IMG_1413"]
    for name in files {
        let inputPath = "/tmp/\(name).jpeg"
        let outputPath = "/tmp/\(name)_dewarped.png"
        guard let image = UIImage(contentsOfFile: inputPath) else {
            print("SKIP: \(name) not found"); continue
        }
        let result = DewarpPipeline.process(image: image)
        if case .success(let output) = result, let data = output.pngData() {
            try data.write(to: URL(fileURLWithPath: outputPath))
            print("OK: \(name)")
        } else {
            print("FAIL: \(name) -> \(result)")
        }
    }
}
```

## Comparing with Python Reference

```bash
# Run Python on the same image
page-dewarp ~/path/to/photo.jpeg

# Compare dimensions and visual output
# Python output is written next to the input as <name>_thresh.png
```

## Notes

- The simulator on Apple Silicon shares `/tmp` with the host — no `simctl` needed.
- After adding/removing test files, always run `xcodegen generate && pod install`.
- The simulator must have a matching device name (check `xcrun simctl list devices available`).
- UIImage handles EXIF orientation transparently — no manual rotation needed.
