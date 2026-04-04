// TestHelpers.swift
// Ported from specs/feat-ios-swift-port-tasks.md (Task P3.2)

import UIKit
import XCTest

// MARK: - Golden File Loader

class GoldenFileLoader {
    static func loadJSON<T: Decodable>(_ filename: String) throws -> T {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        let subdirectory: String? = "GoldenFiles"
        #else
        let bundle = Bundle(for: Self.self)
        let subdirectory: String? = nil
        #endif
        guard let url = bundle.url(forResource: filename, withExtension: "json", subdirectory: subdirectory) else {
            throw NSError(
                domain: "GoldenFiles",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing \(filename).json"]
            )
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }

    static func loadImage(_ filename: String) throws -> UIImage {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        let subdirectory: String? = "GoldenFiles"
        #else
        let bundle = Bundle(for: Self.self)
        let subdirectory: String? = nil
        #endif
        guard let url = bundle.url(forResource: filename, withExtension: nil, subdirectory: subdirectory) else {
            throw NSError(
                domain: "GoldenFiles",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Missing \(filename)"]
            )
        }
        guard let image = UIImage(contentsOfFile: url.path) else {
            throw NSError(
                domain: "GoldenFiles",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Cannot load \(filename)"]
            )
        }
        return image
    }
}

// MARK: - Golden Data Structs

struct GoldenContour: Decodable {
    let center: [Double]
    let tangent: [Double]
    let angle: Double
}

struct GoldenKeypoints: Decodable {
    let span_points: [[[Double]]]
    let corners: [[Double]]
    let ycoords: [Double]
    let xcoords: [[Double]]
}

struct GoldenParams: Decodable {
    let rvec: [Double]
    let tvec: [Double]
    let cubic: [Double]
    let ycoords: [Double]
    let xcoords: [[Double]]
    let page_dims: [Double]
    let span_counts: [Int]
    let final_loss: Double?
}

struct GoldenPageDims: Decodable {
    let width: Double
    let height: Double
}

// spans.json is a flat array of span groups, each group is an array of contour indices
typealias GoldenSpans = [[Int]]
