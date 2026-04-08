#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/// ObjC++ wrapper exposing OpenCV operations to Swift.
/// OpenCV has no official Swift bindings; all calls go through this ObjC++ layer.
@interface OpenCVWrapper : NSObject

/// Returns the OpenCV version string.
/// Used to verify the framework is linked and accessible from Swift.
+ (NSString *)versionString;

// MARK: - Contour detection

/// Find external contours in a grayscale binary mask image.
/// Returns an NSArray of contours, each contour being an NSArray of CGPoint-wrapped
/// NSValues (use NSValue.cgPointValue to unpack).
/// Ported from contours.py: cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)
+ (NSArray<NSArray<NSValue *> *> *)findContoursInGrayImage:(UIImage *)grayImage;

/// Compute image moments for a contour (array of CGPoint NSValues).
/// Returns NSDictionary with keys: "m00", "m10", "m01", "mu20", "mu11", "mu02"
/// Ported from contours.py: cv2.moments(contour)
+ (NSDictionary<NSString *, NSNumber *> *)momentsOfContour:(NSArray<NSValue *> *)contour;

/// Bounding rectangle of a contour.
/// Returns CGRect (x, y, width, height).
/// Ported from contours.py: cv2.boundingRect(contour)
+ (CGRect)boundingRectOfContour:(NSArray<NSValue *> *)contour;

// MARK: - Linear algebra

/// Singular value decomposition of a 2x2 matrix given as flat NSArray of 4 NSNumbers.
/// Returns NSDictionary with keys: "u" (4 values, row-major), "w" (2 values), "vt" (4 values).
/// Ported from contours.py: cv2.SVDecomp(covariance_matrix)
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)svDecompOfMatrix:(NSArray<NSNumber *> *)matrix
                                                                 rows:(NSInteger)rows
                                                                 cols:(NSInteger)cols;

/// PCA on a set of 2D points (NSArray of CGPoint NSValues).
/// Returns NSDictionary with keys: "mean" ([x, y]) and "eigenvectors" (4 values, 2x2 row-major).
/// Ported from spans.py: cv2.PCACompute(points, mean=None)
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)pcaComputeOnPoints:(NSArray<NSValue *> *)points;

// MARK: - Geometry

/// Convex hull of a set of 2D points (NSArray of CGPoint NSValues).
/// Returns NSArray of CGPoint NSValues forming the convex hull.
/// Ported from spans.py: cv2.convexHull(contour)
+ (NSArray<NSValue *> *)convexHullOfPoints:(NSArray<NSValue *> *)points;

// MARK: - Calibration / projection

/// Solve PnP to find rotation and translation vectors.
/// objectPoints: flat NSArray [x0,y0,z0, x1,y1,z1, ...] (count = N*3)
/// imagePoints:  flat NSArray [x0,y0, x1,y1, ...] (count = N*2)
/// cameraMatrix: flat NSArray of 9 values (3x3 row-major)
/// distCoeffs:   flat NSArray of 5 values
/// Returns NSDictionary: "success" (BOOL as NSNumber), "rvec" ([3]), "tvec" ([3])
/// Ported from normalisation.py: cv2.solvePnP(...)
+ (NSDictionary *)solvePnPWithObjectPoints:(NSArray<NSNumber *> *)objectPoints
                               imagePoints:(NSArray<NSNumber *> *)imagePoints
                              cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

/// Project 3D points into image space using rvec, tvec, cameraMatrix, distCoeffs.
/// points3D: flat NSArray [x0,y0,z0, ...] (count = N*3)
/// rvec, tvec: NSArray of 3 values each
/// cameraMatrix: 9 values, distCoeffs: 5 values
/// Returns NSArray of CGPoint NSValues (N projected image points).
/// Ported from projection.py: cv2.projectPoints(objpoints, rvec, tvec, K, zeros)
+ (NSArray<NSValue *> *)projectPointsWith3DPoints:(NSArray<NSNumber *> *)points3D
                                             rvec:(NSArray<NSNumber *> *)rvec
                                             tvec:(NSArray<NSNumber *> *)tvec
                                     cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                       distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

/// Convert a Rodrigues rotation vector (3 elements) to a 3x3 rotation matrix.
/// Returns NSArray of 9 NSNumbers (row-major).
/// Ported from normalisation.py: cv2.Rodrigues(rvec)
+ (NSArray<NSNumber *> *)rodriguesFromVector:(NSArray<NSNumber *> *)rvec;

// MARK: - Image operations

/// Remap a grayscale UIImage using float32 X and Y map arrays.
/// mapX, mapY: flat NSArrays of NSNumbers (row-major, length = outWidth * outHeight)
/// Ported from dewarp.py: cv2.remap(img_gray, map_x, map_y, INTER_CUBIC, BORDER_REPLICATE)
+ (nullable UIImage *)remapImage:(UIImage *)image
                            mapX:(NSArray<NSNumber *> *)mapX
                            mapY:(NSArray<NSNumber *> *)mapY
                           width:(NSInteger)outWidth
                          height:(NSInteger)outHeight;

/// Remap a grayscale image using raw float32 coordinate maps. Avoids NSNumber boxing.
/// mapXData/mapYData: raw bytes of float32 values (row-major, count = outWidth * outHeight)
+ (nullable UIImage *)remapImageData:(UIImage *)image
                            mapXData:(NSData *)mapXData
                            mapYData:(NSData *)mapYData
                               width:(NSInteger)outWidth
                              height:(NSInteger)outHeight;

/// Remap a color image using raw float32 coordinate maps. Preserves RGB channels.
+ (nullable UIImage *)remapColorImageData:(UIImage *)image
                                 mapXData:(NSData *)mapXData
                                 mapYData:(NSData *)mapYData
                                    width:(NSInteger)outWidth
                                   height:(NSInteger)outHeight;

/// Apply adaptive mean thresholding to a grayscale image.
/// Ported from dewarp.py: cv2.adaptiveThreshold(remapped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C)
+ (nullable UIImage *)adaptiveThresholdImage:(UIImage *)grayImage
                                    maxValue:(double)maxValue
                                   blockSize:(NSInteger)blockSize
                                           C:(double)C;

/// Resize a UIImage to the given dimensions.
/// interpolation: 0=INTER_NEAREST, 1=INTER_LINEAR, 2=INTER_CUBIC, 3=INTER_AREA
/// Ported from dewarp.py: cv2.resize(image, (width, height), interpolation=...)
+ (nullable UIImage *)resizeImage:(UIImage *)image
                            width:(NSInteger)width
                           height:(NSInteger)height
                    interpolation:(NSInteger)interpolation;

/// Resize a flat float coordinate map using INTER_CUBIC.
/// Ported from dewarp.py:105-114 — cv2.resize(image_x_coords, (width, height), INTER_CUBIC)
/// srcMap: flat NSArray of NSNumbers (float, row-major, count = srcWidth * srcHeight)
/// Returns flat NSArray of NSNumbers (float, row-major, count = dstWidth * dstHeight)
+ (nullable NSArray<NSNumber *> *)resizeFloatMap:(NSArray<NSNumber *> *)srcMap
                                        srcWidth:(NSInteger)srcWidth
                                       srcHeight:(NSInteger)srcHeight
                                        dstWidth:(NSInteger)dstWidth
                                       dstHeight:(NSInteger)dstHeight;

/// Resize a flat float32 coordinate map using INTER_CUBIC. Avoids NSNumber boxing.
/// srcData: raw bytes of float32 values (row-major, count = srcWidth * srcHeight)
/// Returns NSData of float32 values (row-major, count = dstWidth * dstHeight)
+ (nullable NSData *)resizeFloatMapData:(NSData *)srcData
                               srcWidth:(NSInteger)srcWidth
                              srcHeight:(NSInteger)srcHeight
                               dstWidth:(NSInteger)dstWidth
                              dstHeight:(NSInteger)dstHeight;

/// Compute the maximum column thickness of a contour using a tight binary mask.
/// Matches Python's tight_mask.sum(axis=0).max() check in contours.py:231.
/// contour: NSArray of CGPoint NSValues (raw image coordinates).
/// rect: bounding rect of the contour (CGRect with x, y, width, height).
/// Returns the maximum column occupancy (number of filled pixels in the tallest column).
+ (NSInteger)maxColumnThicknessOfContour:(NSArray<NSValue *> *)contour
                             boundingRect:(CGRect)rect;

/// Compute per-column vertical mean y-positions within a tight contour mask.
/// For each column x in [0, rect.width), returns the mean y of filled pixels
/// within the contour's tight bounding-rect mask (local coordinates).
/// Returns NSArray<NSNumber> of count = (int)rect.size.width, values are local y means.
/// Ported from spans.py:186-188 — (yvals * mask).sum(axis=0) / mask.sum(axis=0)
+ (NSArray<NSNumber *> *)columnMeansOfContour:(NSArray<NSValue *> *)contour
                                  boundingRect:(CGRect)rect;

/// Draw filled contours on a copy of the input image (for debug visualization).
/// contours: NSArray of contours, each contour an NSArray of CGPoint NSValues.
/// color: [R, G, B] as NSArray of 3 NSNumbers (0–255).
/// thickness: -1 for filled.
+ (nullable UIImage *)drawContoursOnImage:(UIImage *)image
                                 contours:(NSArray<NSArray<NSValue *> *> *)contours
                                    color:(NSArray<NSNumber *> *)color
                                thickness:(NSInteger)thickness;

/// Create a page mask: white filled rectangle on a black background, with margins.
/// Ported from image.py:calculate_page_extents — cv2.rectangle(pagemask, ...)
/// Returns a single-channel (grayscale) UIImage of size width × height.
+ (UIImage *)createPageMaskWithWidth:(NSInteger)width
                              height:(NSInteger)height
                             marginX:(NSInteger)marginX
                             marginY:(NSInteger)marginY;

/// Compute the binary detection mask used for text or line contour extraction.
///
/// Matches the Python Mask.calculate() sequence in mask.py:67-100.
/// Steps:
///   1. Convert colorImage to grayscale.
///   2. Apply adaptive threshold (ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
///      blockSize=adaptiveWinsz, C=25 for text, C=7 for lines).
///   3. Text mode: dilate with 9x1, then erode with 1x3.
///      Line mode: erode with 3x1 (3 iterations), then dilate with 8x2.
///   4. Bitwise AND with pagemask to clip to page region.
///
/// - Parameters:
///   - colorImage:    Full-colour (RGB/RGBA) page image (the "small" downsampled image).
///   - pagemask:      Single-channel (grayscale) page region mask — white rectangle
///                    inside, black margins outside.
///   - isText:        YES for text detection mode; NO for line/table detection mode.
///   - adaptiveWinsz: Block size for adaptive threshold (must be odd, >= 3).
/// - Returns: Single-channel binary mask suitable for contour extraction, or nil on error.
+ (nullable UIImage *)computeDetectionMask:(UIImage *)colorImage
                                  pagemask:(UIImage *)pagemask
                                    isText:(BOOL)isText
                             adaptiveWinsz:(NSInteger)adaptiveWinsz;

@end

NS_ASSUME_NONNULL_END
