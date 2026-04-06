#!/bin/bash
# compare-with-python.sh — Run both Swift and Python pipelines on test images and compare
#
# Usage: ./scripts/compare-with-python.sh [image1.jpeg image2.jpeg ...]
#        If no args, processes all IMG_*.jpeg files on ~/Desktop
#
# Output: prints a comparison table and writes results to /tmp/dewarp-comparison/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="/tmp/dewarp-comparison"
SIMULATOR="iPhone 17 Pro"

mkdir -p "$RESULTS_DIR"

# Collect input files
if [[ $# -gt 0 ]]; then
    INPUT_FILES=("$@")
else
    INPUT_FILES=(~/Desktop/IMG_*.jpeg)
fi

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "No input files found."
    exit 1
fi

echo "=== Dewarp Comparison: Swift vs Python ==="
echo "Images: ${#INPUT_FILES[@]}"
echo ""

# --- Step 1: Generate Python reference outputs ---
echo "--- Running Python (Powell) ---"
for img in "${INPUT_FILES[@]}"; do
    name=$(basename "$img" .jpeg)
    echo -n "  $name ... "
    cp "$img" "$RESULTS_DIR/${name}_input.jpeg"
    (cd "$RESULTS_DIR" && page-dewarp "${name}_input.jpeg" 2>/dev/null)
    if [[ -f "$RESULTS_DIR/${name}_input_thresh.png" ]]; then
        echo "ok"
    else
        echo "FAILED"
    fi
done
echo ""

# --- Step 2: Prepare Swift batch test ---
echo "--- Preparing Swift pipeline ---"

# Copy inputs to /tmp for simulator access
SWIFT_NAMES=()
for img in "${INPUT_FILES[@]}"; do
    name=$(basename "$img" .jpeg)
    cp "$img" "/tmp/${name}.jpeg"
    SWIFT_NAMES+=("$name")
done

# Generate the test file
NAMES_ARRAY=$(printf '"%s", ' "${SWIFT_NAMES[@]}")
NAMES_ARRAY="[${NAMES_ARRAY%, }]"

cat > "$PROJECT_DIR/Tests/ComparisonBatchTest.swift" << SWIFT_EOF
import XCTest
@testable import PageDewarp

class ComparisonBatchTest: XCTestCase {
    func testBatchDewarp() throws {
        let names: [String] = ${NAMES_ARRAY}
        for name in names {
            let inputPath = "/tmp/\(name).jpeg"
            let outputPath = "/tmp/\(name)_swift.png"
            guard let image = UIImage(contentsOfFile: inputPath) else {
                print("SKIP \(name): not found at \(inputPath)")
                continue
            }
            let start = CFAbsoluteTimeGetCurrent()
            let result = DewarpPipeline.process(image: image)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            switch result {
            case .success(let output):
                let w = Int(output.size.width * output.scale)
                let h = Int(output.size.height * output.scale)
                if let data = output.pngData() {
                    try data.write(to: URL(fileURLWithPath: outputPath))
                }
                print("OK \(name): \(w)x\(h) in \(String(format: "%.2f", elapsed))s")
            case .failure(let error):
                print("FAIL \(name): \(error)")
            }
        }
    }
}
SWIFT_EOF

# Rebuild and run
echo "--- Building and running Swift pipeline ---"
(cd "$PROJECT_DIR" && xcodegen generate 2>&1 | tail -1 && pod install 2>&1 | tail -1)

xcodebuild test \
    -workspace "$PROJECT_DIR/PageDewarp.xcworkspace" \
    -scheme PageDewarp \
    -destination "platform=iOS Simulator,name=$SIMULATOR" \
    -only-testing:PageDewarpTests/ComparisonBatchTest/testBatchDewarp \
    2>&1 | grep -E "^OK |^FAIL |^SKIP |Test Suite|TEST SUCCEEDED|TEST FAILED"

echo ""

# Copy Swift results
for name in "${SWIFT_NAMES[@]}"; do
    if [[ -f "/tmp/${name}_swift.png" ]]; then
        cp "/tmp/${name}_swift.png" "$RESULTS_DIR/"
    fi
done

# --- Step 3: Compare ---
echo "=== COMPARISON RESULTS ==="
echo ""
printf "%-20s %15s %15s %10s\n" "Image" "Python (WxH)" "Swift (WxH)" "Match"
printf "%-20s %15s %15s %10s\n" "-----" "------------" "-----------" "-----"

for name in "${SWIFT_NAMES[@]}"; do
    py_file="$RESULTS_DIR/${name}_input_thresh.png"
    sw_file="$RESULTS_DIR/${name}_swift.png"

    if [[ -f "$py_file" && -f "$sw_file" ]]; then
        py_dims=$(python3 -c "from PIL import Image; i=Image.open('$py_file'); print(f'{i.size[0]}x{i.size[1]}')")
        sw_dims=$(python3 -c "from PIL import Image; i=Image.open('$sw_file'); print(f'{i.size[0]}x{i.size[1]}')")

        # Compute dimension similarity (ratio of areas)
        py_area=$(python3 -c "from PIL import Image; i=Image.open('$py_file'); print(i.size[0]*i.size[1])")
        sw_area=$(python3 -c "from PIL import Image; i=Image.open('$sw_file'); print(i.size[0]*i.size[1])")
        ratio=$(python3 -c "print(f'{min($py_area,$sw_area)/max($py_area,$sw_area)*100:.0f}%')")

        printf "%-20s %15s %15s %10s\n" "$name" "$py_dims" "$sw_dims" "$ratio"
    else
        status="MISSING"
        [[ ! -f "$py_file" ]] && status="no python"
        [[ ! -f "$sw_file" ]] && status="no swift"
        printf "%-20s %15s %15s %10s\n" "$name" "-" "-" "$status"
    fi
done

echo ""
echo "Results in: $RESULTS_DIR/"

# Cleanup test file
rm -f "$PROJECT_DIR/Tests/ComparisonBatchTest.swift"
(cd "$PROJECT_DIR" && xcodegen generate 2>&1 | tail -1 && pod install 2>&1 | tail -1)
