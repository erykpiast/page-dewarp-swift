#!/bin/bash
# Run the Swift page-dewarp pipeline on an image.
#
# Usage: ./scripts/run-dewarp.sh [options] <input-image> [output-dir]
#
# Options:
#   --lbfgsb   Use L-BFGS-B optimizer (default: Powell)
#   --binary   Output adaptive-thresholded binary (default: color)
#
# Examples:
#   ./scripts/run-dewarp.sh ~/Desktop/IMG_1389.jpeg ~/Desktop
#   ./scripts/run-dewarp.sh --lbfgsb ~/Desktop/IMG_1389.jpeg
#   ./scripts/run-dewarp.sh --binary --lbfgsb ~/Desktop/IMG_1389.jpeg ~/Desktop

set -euo pipefail

METHOD="powell"
BINARY="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --lbfgsb) METHOD="lbfgsb"; shift ;;
        --binary) BINARY="1"; shift ;;
        -*)       echo "Unknown option: $1"; exit 1 ;;
        *)        break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--lbfgsb] [--binary] <input-image> [output-dir]"
    exit 1
fi

INPUT="$1"
BASENAME="$(basename "${INPUT%.*}")"
OUTPUT_DIR="${2:-$(dirname "$INPUT")}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Copy input and write options to /tmp (shared with simulator)
cp "$INPUT" /tmp/input.jpeg
printf "method=%s\nbinary=%s\n" "$METHOD" "$BINARY" > /tmp/dewarp_options.txt

# Run the test
cd "$PROJECT_DIR"
xcodebuild test \
    -workspace PageDewarp.xcworkspace \
    -scheme PageDewarp \
    -destination "platform=iOS Simulator,name=iPhone 17 Pro" \
    -only-testing:PageDewarpTests/RunSingleImageTest/testProcessImage \
    2>&1 | grep -E "(Powell|L-BFGS|failed)"

# Copy output
mkdir -p "$OUTPUT_DIR"
src="/tmp/output.png"
if [[ -f "$src" ]]; then
    suffix="${METHOD}"
    [[ "$BINARY" == "1" ]] && suffix="${suffix}_binary"
    dst="${OUTPUT_DIR}/${BASENAME}_${suffix}.png"
    cp "$src" "$dst"
    echo "Saved: $dst"
fi
