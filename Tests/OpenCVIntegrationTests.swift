import XCTest
@testable import PageDewarp

/// Verifies that OpenCV is linked and accessible from Swift via the ObjC++ bridge.
final class OpenCVIntegrationTests: XCTestCase {

    func testOpenCVVersionIsNonEmpty() {
        let version = OpenCVWrapper.versionString()
        XCTAssertFalse(version.isEmpty, "OpenCV version string should not be empty")
    }
}
