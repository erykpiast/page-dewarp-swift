import XCTest
@testable import PageDewarp

final class RemapperTests: XCTestCase {

    // MARK: - roundNearestMultiple tests

    func testRoundNearestMultipleAlreadyMultiple() {
        XCTAssertEqual(roundNearestMultiple(96, 16), 96)
    }

    func testRoundNearestMultipleRoundsUp() {
        XCTAssertEqual(roundNearestMultiple(100, 16), 112)
    }

    func testRoundNearestMultipleSmallValue() {
        XCTAssertEqual(roundNearestMultiple(1, 16), 16)
    }

    func testRoundNearestMultipleZero() {
        XCTAssertEqual(roundNearestMultiple(0, 16), 0)
    }

    func testRoundNearestMultipleExactly16() {
        XCTAssertEqual(roundNearestMultiple(16, 16), 16)
    }

    func testRoundNearestMultiple17() {
        XCTAssertEqual(roundNearestMultiple(17, 16), 32)
    }

    // MARK: - Output dimension calculation tests

    func testOutputDimsCalculation() {
        // For imgHeight=1000, pageDims=[0.7, 1.0], outputZoom=1.0, remapDecimate=16
        // heightF = 0.5 * 1.0 * 1.0 * 1000 = 500 → roundNearestMultiple(500, 16) = 512
        // widthF  = 512 * 0.7 / 1.0 = 358.4 → roundNearestMultiple(358, 16) = 368
        let imgHeight = 1000
        let pageDims = [0.7, 1.0]
        let heightF = Int(0.5 * pageDims[1] * DewarpConfig.outputZoom * Double(imgHeight))
        let height = roundNearestMultiple(heightF, DewarpConfig.remapDecimate)
        let widthF = Int(Double(height) * pageDims[0] / pageDims[1])
        let width = roundNearestMultiple(widthF, DewarpConfig.remapDecimate)

        XCTAssertEqual(height % DewarpConfig.remapDecimate, 0, "height must be multiple of remapDecimate")
        XCTAssertEqual(width % DewarpConfig.remapDecimate, 0, "width must be multiple of remapDecimate")
        XCTAssertGreaterThan(height, 0)
        XCTAssertGreaterThan(width, 0)
        // Aspect ratio preserved approximately
        let aspectDiff = abs(Double(width) / Double(height) - pageDims[0] / pageDims[1])
        XCTAssertLessThan(aspectDiff, 0.1)
    }

    func testDecimatedDimensions() {
        let height = 512
        let width = 368
        let hSmall = height / DewarpConfig.remapDecimate
        let wSmall = width / DewarpConfig.remapDecimate
        XCTAssertEqual(hSmall, 32)
        XCTAssertEqual(wSmall, 23)
    }
}
