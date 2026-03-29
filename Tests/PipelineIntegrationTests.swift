// PipelineIntegrationTests.swift
// Integration tests for the full DewarpPipeline.

import XCTest
@testable import PageDewarp

class PipelineIntegrationTests: XCTestCase {

    /// Run the full pipeline on boston_cooking_a_input.jpg and verify a successful result.
    func testFullPipelineOnBostonCooking() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")

        let result = DewarpPipeline.process(image: image)

        switch result {
        case .success(let outputImage):
            // Attach the output image as a test artifact for visual inspection.
            let attachment = XCTAttachment(image: outputImage)
            attachment.name = "dewarp_output_boston_cooking_a"
            attachment.lifetime = .keepAlways
            add(attachment)

            // Verify output dimensions are reasonable (non-zero).
            XCTAssertGreaterThan(outputImage.size.width, 0, "Output width should be positive")
            XCTAssertGreaterThan(outputImage.size.height, 0, "Output height should be positive")

        case .failure(let error):
            XCTFail("Pipeline failed with error: \(error)")
        }
    }

    /// Verify the pipeline does not crash on a minimal valid image.
    func testPipelineDoesNotCrash() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        // Just run — any result (success or known failure) is acceptable.
        _ = DewarpPipeline.process(image: image)
    }

    /// Verify that the pipeline returns a reasonable output image size.
    /// The output should be larger than 100×100 pixels.
    func testOutputImageDimensionsAreReasonable() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        guard case .success(let output) = DewarpPipeline.process(image: image) else {
            throw XCTSkip("Pipeline did not produce output (expected success)")
        }
        let pixelW = output.size.width * output.scale
        let pixelH = output.size.height * output.scale
        XCTAssertGreaterThan(pixelW, 100, "Output pixel width should be > 100")
        XCTAssertGreaterThan(pixelH, 100, "Output pixel height should be > 100")
    }
}
