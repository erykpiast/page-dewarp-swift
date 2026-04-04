#!/bin/bash
set -euo pipefail

# Build a stripped OpenCV XCFramework with only the modules needed by PageDewarp:
#   core, imgproc, flann, features2d, calib3d
#
# Produces opencv2.xcframework.zip in the current directory and prints the
# sha256 checksum for use in Package.swift.

OPENCV_VERSION="${1:-4.10.0}"
WORKDIR="/tmp/opencv-spm-build"
OUTPUT_DIR="$(pwd)/opencv2.xcframework"

WITHOUT_FLAGS=(
    --without dnn
    --without highgui
    --without imgcodecs
    --without ml
    --without objdetect
    --without photo
    --without stitching
    --without video
    --without videoio
    --without gapi
    --without ts
    --without objc
    --without java
    --without python
)

echo "Building OpenCV $OPENCV_VERSION minimal XCFramework..."

# 1. Clone OpenCV source (if not already present)
if [ ! -d "$WORKDIR/opencv" ]; then
    rm -rf "$WORKDIR"
    git clone --depth 1 --branch "$OPENCV_VERSION" \
        https://github.com/opencv/opencv.git "$WORKDIR/opencv"
fi

BUILD_SCRIPT="$WORKDIR/opencv/platforms/ios/build_framework.py"

# 2. Build iphoneos (arm64)
echo "=== Building iphoneos arm64 ==="
python3 "$BUILD_SCRIPT" "$WORKDIR/build-iphoneos" \
    --iphoneos_archs arm64 \
    --build_only_specified_archs \
    --iphoneos_deployment_target 16.0 \
    --framework_name opencv2 \
    "${WITHOUT_FLAGS[@]}"

# 3. Build iphonesimulator (arm64)
echo "=== Building iphonesimulator arm64 ==="
python3 "$BUILD_SCRIPT" "$WORKDIR/build-iphonesimulator" \
    --iphonesimulator_archs arm64 \
    --build_only_specified_archs \
    --iphoneos_deployment_target 16.0 \
    --framework_name opencv2 \
    "${WITHOUT_FLAGS[@]}"

# 4. Create XCFramework
echo "=== Creating XCFramework ==="
rm -rf "$OUTPUT_DIR"
xcodebuild -create-xcframework \
    -framework "$WORKDIR/build-iphoneos/opencv2.framework" \
    -framework "$WORKDIR/build-iphonesimulator/opencv2.framework" \
    -output "$OUTPUT_DIR"

# 5. Zip and compute checksum for SPM
cd "$(dirname "$OUTPUT_DIR")"
rm -f opencv2.xcframework.zip
zip -r opencv2.xcframework.zip opencv2.xcframework
CHECKSUM=$(swift package compute-checksum opencv2.xcframework.zip)

echo ""
echo "=========================================="
echo "XCFramework ready at: $OUTPUT_DIR"
echo "Zip: $(pwd)/opencv2.xcframework.zip"
echo "Size: $(du -sh opencv2.xcframework.zip | cut -f1)"
echo "Checksum: $CHECKSUM"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload opencv2.xcframework.zip as a GitHub release asset"
echo "  2. Update Package.swift checksum to: $CHECKSUM"
