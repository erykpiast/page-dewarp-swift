// OpenCV must be imported first to avoid conflicts with UIKit/Foundation.
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/calib3d.hpp>

#import "include/OpenCVWrapper.h"

// MARK: - Private helpers

/// Convert a UIImage to a BGR cv::Mat.
/// Handles grayscale and RGBA source images.
/// Normalizes EXIF orientation so pixels match cv2.imread behavior (Python side).
static cv::Mat cvMatFromUIImage(UIImage *image) {
    UIImage *normalizedImage = image;
    if (image.imageOrientation != UIImageOrientationUp) {
        UIGraphicsBeginImageContextWithOptions(image.size, YES, image.scale);
        [image drawInRect:CGRectMake(0, 0, image.size.width, image.size.height)];
        normalizedImage = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
    }
    CGImageRef cgImage = normalizedImage.CGImage;
    size_t cols = CGImageGetWidth(cgImage);
    size_t rows = CGImageGetHeight(cgImage);

    // Allocate RGBA buffer.
    cv::Mat rgba(rows, cols, CV_8UC4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data, cols, rows, 8, rgba.step[0], colorSpace,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(ctx, CGRectMake(0, 0, cols, rows), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);

    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

/// Convert a grayscale UIImage to a single-channel cv::Mat.
/// Normalizes EXIF orientation so pixels match cv2.imread behavior (Python side).
static cv::Mat cvGrayMatFromUIImage(UIImage *image) {
    UIImage *normalizedImage = image;
    if (image.imageOrientation != UIImageOrientationUp) {
        UIGraphicsBeginImageContextWithOptions(image.size, YES, image.scale);
        [image drawInRect:CGRectMake(0, 0, image.size.width, image.size.height)];
        normalizedImage = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
    }
    CGImageRef cgImage = normalizedImage.CGImage;
    size_t cols = CGImageGetWidth(cgImage);
    size_t rows = CGImageGetHeight(cgImage);

    cv::Mat gray(rows, cols, CV_8UC1);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGContextRef ctx = CGBitmapContextCreate(
        gray.data, cols, rows, 8, gray.step[0], colorSpace,
        kCGImageAlphaNone | kCGBitmapByteOrderDefault);
    CGContextDrawImage(ctx, CGRectMake(0, 0, cols, rows), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);

    return gray;
}

/// Convert a grayscale cv::Mat to a UIImage.
static UIImage *UIImageFromGrayMat(const cv::Mat &mat) {
    NSData *data = [NSData dataWithBytes:mat.data
                                  length:mat.total() * mat.elemSize()];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cgImage = CGImageCreate(
        mat.cols, mat.rows, 8, 8, mat.step[0],
        colorSpace, kCGImageAlphaNone | kCGBitmapByteOrderDefault,
        provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *result = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return result;
}

/// Convert a BGR cv::Mat to a UIImage.
static UIImage *UIImageFromBGRMat(const cv::Mat &bgrMat) {
    cv::Mat rgba;
    cv::cvtColor(bgrMat, rgba, cv::COLOR_BGR2RGBA);
    NSData *data = [NSData dataWithBytes:rgba.data
                                  length:rgba.total() * rgba.elemSize()];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cgImage = CGImageCreate(
        rgba.cols, rgba.rows, 8, 32, rgba.step[0],
        colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault,
        provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *result = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return result;
}

// MARK: - OpenCVWrapper implementation

@implementation OpenCVWrapper

+ (NSString *)versionString {
    return [NSString stringWithUTF8String:cv::getVersionString().c_str()];
}

// MARK: Contour detection

/// Ported from contours.py:219
/// cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)
+ (NSArray<NSArray<NSValue *> *> *)findContoursInGrayImage:(UIImage *)grayImage {
    cv::Mat gray = cvGrayMatFromUIImage(grayImage);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    NSMutableArray<NSArray<NSValue *> *> *result = [NSMutableArray array];
    for (const auto &contour : contours) {
        NSMutableArray<NSValue *> *pts = [NSMutableArray arrayWithCapacity:contour.size()];
        for (const auto &p : contour) {
            [pts addObject:[NSValue valueWithCGPoint:CGPointMake(p.x, p.y)]];
        }
        [result addObject:[pts copy]];
    }
    return [result copy];
}

/// Ported from contours.py:66
/// cv2.moments(contour)
+ (NSDictionary<NSString *, NSNumber *> *)momentsOfContour:(NSArray<NSValue *> *)contour {
    std::vector<cv::Point2f> pts;
    pts.reserve(contour.count);
    for (NSValue *v in contour) {
        CGPoint p = v.CGPointValue;
        pts.push_back(cv::Point2f(p.x, p.y));
    }
    cv::Moments m = cv::moments(pts, false);
    return @{
        @"m00":  @(m.m00),
        @"m10":  @(m.m10),
        @"m01":  @(m.m01),
        @"mu20": @(m.mu20),
        @"mu11": @(m.mu11),
        @"mu02": @(m.mu02),
    };
}

/// Ported from contours.py:228
/// cv2.boundingRect(contour)
+ (CGRect)boundingRectOfContour:(NSArray<NSValue *> *)contour {
    std::vector<cv::Point2f> pts;
    pts.reserve(contour.count);
    for (NSValue *v in contour) {
        CGPoint p = v.CGPointValue;
        pts.push_back(cv::Point2f(p.x, p.y));
    }
    cv::Rect r = cv::boundingRect(pts);
    return CGRectMake(r.x, r.y, r.width, r.height);
}

// MARK: Linear algebra

/// Ported from contours.py:75
/// cv2.SVDecomp(covariance_matrix) — used for blob orientation from 2nd-order moments.
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)svDecompOfMatrix:(NSArray<NSNumber *> *)matrix
                                                                  rows:(NSInteger)rows
                                                                  cols:(NSInteger)cols {
    cv::Mat src((int)rows, (int)cols, CV_64F);
    for (NSInteger i = 0; i < rows * cols; i++) {
        src.at<double>((int)i) = [matrix[(NSUInteger)i] doubleValue];
    }
    cv::Mat w, u, vt;
    cv::SVDecomp(src, w, u, vt);

    // Flatten to NSArray<NSNumber>.
    auto toArray = [](const cv::Mat &m) -> NSArray<NSNumber *> * {
        NSMutableArray<NSNumber *> *arr = [NSMutableArray arrayWithCapacity:m.total()];
        for (int i = 0; i < (int)m.total(); i++) {
            [arr addObject:@(m.at<double>(i))];
        }
        return [arr copy];
    };

    return @{
        @"u":  toArray(u),
        @"w":  toArray(w),
        @"vt": toArray(vt),
    };
}

/// Ported from spans.py:13
/// cv2.PCACompute(points, mean=None)
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)pcaComputeOnPoints:(NSArray<NSValue *> *)points {
    int N = (int)points.count;
    cv::Mat data(N, 2, CV_32F);
    for (int i = 0; i < N; i++) {
        CGPoint p = points[(NSUInteger)i].CGPointValue;
        data.at<float>(i, 0) = (float)p.x;
        data.at<float>(i, 1) = (float)p.y;
    }
    cv::Mat mean, eigenvectors;
    cv::PCACompute(data, mean, eigenvectors);

    auto toArray = [](const cv::Mat &m) -> NSArray<NSNumber *> * {
        NSMutableArray<NSNumber *> *arr = [NSMutableArray arrayWithCapacity:m.total()];
        for (int i = 0; i < (int)m.total(); i++) {
            [arr addObject:@(m.at<float>(i))];
        }
        return [arr copy];
    };

    return @{
        @"mean":         toArray(mean),
        @"eigenvectors": toArray(eigenvectors),
    };
}

// MARK: Geometry

/// Ported from spans.py:13
/// cv2.convexHull(contour)
+ (NSArray<NSValue *> *)convexHullOfPoints:(NSArray<NSValue *> *)points {
    std::vector<cv::Point2f> pts;
    pts.reserve(points.count);
    for (NSValue *v in points) {
        CGPoint p = v.CGPointValue;
        pts.push_back(cv::Point2f(p.x, p.y));
    }
    std::vector<cv::Point2f> hull;
    cv::convexHull(pts, hull);

    NSMutableArray<NSValue *> *result = [NSMutableArray arrayWithCapacity:hull.size()];
    for (const auto &p : hull) {
        [result addObject:[NSValue valueWithCGPoint:CGPointMake(p.x, p.y)]];
    }
    return [result copy];
}

// MARK: Calibration / projection

/// Ported from normalisation.py (solvePnP usage)
+ (NSDictionary *)solvePnPWithObjectPoints:(NSArray<NSNumber *> *)objectPoints
                               imagePoints:(NSArray<NSNumber *> *)imagePoints
                              cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                distCoeffs:(NSArray<NSNumber *> *)distCoeffs {
    NSUInteger N = objectPoints.count / 3;

    std::vector<cv::Point3d> obj;
    obj.reserve(N);
    for (NSUInteger i = 0; i < N; i++) {
        obj.push_back(cv::Point3d(
            [objectPoints[i*3]   doubleValue],
            [objectPoints[i*3+1] doubleValue],
            [objectPoints[i*3+2] doubleValue]));
    }

    std::vector<cv::Point2d> img;
    img.reserve(N);
    for (NSUInteger i = 0; i < N; i++) {
        img.push_back(cv::Point2d(
            [imagePoints[i*2]   doubleValue],
            [imagePoints[i*2+1] doubleValue]));
    }

    // Build 3x3 camera matrix.
    cv::Mat K(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        K.at<double>(i) = [cameraMatrix[(NSUInteger)i] doubleValue];
    }

    // Build distortion coefficients (5x1).
    cv::Mat dist(5, 1, CV_64F);
    for (int i = 0; i < 5; i++) {
        dist.at<double>(i) = [distCoeffs[(NSUInteger)i] doubleValue];
    }

    cv::Mat rvec, tvec;
    bool success = false;
    try {
        success = cv::solvePnP(obj, img, K, dist, rvec, tvec);
    } catch (const cv::Exception &) {
        return @{ @"success": @(false), @"rvec": @[], @"tvec": @[] };
    } catch (...) {
        return @{ @"success": @(false), @"rvec": @[], @"tvec": @[] };
    }

    if (!success || rvec.empty() || tvec.empty()) {
        return @{ @"success": @(false), @"rvec": @[], @"tvec": @[] };
    }

    NSMutableArray<NSNumber *> *rvecArr = [NSMutableArray arrayWithCapacity:3];
    NSMutableArray<NSNumber *> *tvecArr = [NSMutableArray arrayWithCapacity:3];
    for (int i = 0; i < 3; i++) {
        [rvecArr addObject:@(rvec.at<double>(i))];
        [tvecArr addObject:@(tvec.at<double>(i))];
    }

    return @{
        @"success": @(success),
        @"rvec":    [rvecArr copy],
        @"tvec":    [tvecArr copy],
    };
}

/// Ported from projection.py:50-56
/// cv2.projectPoints(objpoints, rvec, tvec, K, distCoeffs)
+ (NSArray<NSValue *> *)projectPointsWith3DPoints:(NSArray<NSNumber *> *)points3D
                                             rvec:(NSArray<NSNumber *> *)rvec
                                             tvec:(NSArray<NSNumber *> *)tvec
                                     cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                       distCoeffs:(NSArray<NSNumber *> *)distCoeffs {
    NSUInteger N = points3D.count / 3;
    std::vector<cv::Point3d> obj;
    obj.reserve(N);
    for (NSUInteger i = 0; i < N; i++) {
        obj.push_back(cv::Point3d(
            [points3D[i*3]   doubleValue],
            [points3D[i*3+1] doubleValue],
            [points3D[i*3+2] doubleValue]));
    }

    cv::Mat rvecMat(3, 1, CV_64F);
    cv::Mat tvecMat(3, 1, CV_64F);
    for (int i = 0; i < 3; i++) {
        rvecMat.at<double>(i) = [rvec[(NSUInteger)i] doubleValue];
        tvecMat.at<double>(i) = [tvec[(NSUInteger)i] doubleValue];
    }

    cv::Mat K(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        K.at<double>(i) = [cameraMatrix[(NSUInteger)i] doubleValue];
    }

    cv::Mat dist(5, 1, CV_64F);
    for (int i = 0; i < 5; i++) {
        dist.at<double>(i) = [distCoeffs[(NSUInteger)i] doubleValue];
    }

    std::vector<cv::Point2d> projected;
    cv::Mat jacobian;
    cv::projectPoints(obj, rvecMat, tvecMat, K, dist, projected, jacobian);

    NSMutableArray<NSValue *> *result = [NSMutableArray arrayWithCapacity:N];
    for (const auto &p : projected) {
        [result addObject:[NSValue valueWithCGPoint:CGPointMake(p.x, p.y)]];
    }
    return [result copy];
}

/// Ported from normalisation.py
/// cv2.Rodrigues(rvec) → 3x3 rotation matrix
+ (NSArray<NSNumber *> *)rodriguesFromVector:(NSArray<NSNumber *> *)rvec {
    cv::Mat rvecMat(3, 1, CV_64F);
    for (int i = 0; i < 3; i++) {
        rvecMat.at<double>(i) = [rvec[(NSUInteger)i] doubleValue];
    }
    cv::Mat rotMat;
    cv::Rodrigues(rvecMat, rotMat);

    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:9];
    for (int i = 0; i < 9; i++) {
        [result addObject:@(rotMat.at<double>(i))];
    }
    return [result copy];
}

// MARK: Image operations

/// Ported from dewarp.py:118-125
/// cv2.remap(img_gray, map_x, map_y, INTER_CUBIC, None, BORDER_REPLICATE)
+ (nullable UIImage *)remapImage:(UIImage *)image
                            mapX:(NSArray<NSNumber *> *)mapX
                            mapY:(NSArray<NSNumber *> *)mapY
                           width:(NSInteger)outWidth
                          height:(NSInteger)outHeight {
    @autoreleasepool {
        cv::Mat gray = cvGrayMatFromUIImage(image);

        int h = (int)outHeight;
        int w = (int)outWidth;
        NSUInteger total = (NSUInteger)(w * h);

        cv::Mat mX(h, w, CV_32F);
        cv::Mat mY(h, w, CV_32F);
        for (NSUInteger i = 0; i < total; i++) {
            mX.at<float>((int)i) = [mapX[i] floatValue];
            mY.at<float>((int)i) = [mapY[i] floatValue];
        }

        cv::Mat remapped;
        cv::remap(gray, remapped, mX, mY, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        return UIImageFromGrayMat(remapped);
    }
}

/// Ported from dewarp.py:130-137
/// cv2.adaptiveThreshold(remapped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C)
+ (nullable UIImage *)adaptiveThresholdImage:(UIImage *)grayImage
                                    maxValue:(double)maxValue
                                   blockSize:(NSInteger)blockSize
                                           C:(double)C {
    @autoreleasepool {
        cv::Mat gray = cvGrayMatFromUIImage(grayImage);
        cv::Mat threshed;
        // blockSize must be odd and >= 3.
        int bs = (int)blockSize;
        if (bs % 2 == 0) bs += 1;
        if (bs < 3) bs = 3;
        cv::adaptiveThreshold(gray, threshed, maxValue,
                              cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                              bs, C);
        return UIImageFromGrayMat(threshed);
    }
}

/// Ported from dewarp.py:105-113
/// cv2.resize(image, (width, height), interpolation=...)
+ (nullable UIImage *)resizeImage:(UIImage *)image
                            width:(NSInteger)width
                           height:(NSInteger)height
                    interpolation:(NSInteger)interpolation {
    @autoreleasepool {
        // Detect if source is grayscale or color.
        CGImageRef cgImage = image.CGImage;
        size_t bitsPerPixel = CGImageGetBitsPerPixel(cgImage);
        BOOL isGray = (bitsPerPixel == 8);

        int interp;
        switch (interpolation) {
            case 0: interp = cv::INTER_NEAREST; break;
            case 1: interp = cv::INTER_LINEAR;  break;
            case 2: interp = cv::INTER_CUBIC;   break;
            case 3: interp = cv::INTER_AREA;    break;
            default: interp = cv::INTER_LINEAR; break;
        }

        cv::Size dstSize((int)width, (int)height);
        cv::Mat resized;

        if (isGray) {
            cv::Mat gray = cvGrayMatFromUIImage(image);
            cv::resize(gray, resized, dstSize, 0, 0, interp);
            return UIImageFromGrayMat(resized);
        } else {
            cv::Mat bgr = cvMatFromUIImage(image);
            cv::resize(bgr, resized, dstSize, 0, 0, interp);
            return UIImageFromBGRMat(resized);
        }
    }
}

/// Ported from dewarp.py:105-114
/// cv2.resize(image_x_coords, (width, height), interpolation=INTER_CUBIC).astype(np.float32)
/// Resizes a flat float coordinate map from (srcHeight x srcWidth) to (dstHeight x dstWidth).
+ (nullable NSArray<NSNumber *> *)resizeFloatMap:(NSArray<NSNumber *> *)srcMap
                                        srcWidth:(NSInteger)srcWidth
                                       srcHeight:(NSInteger)srcHeight
                                        dstWidth:(NSInteger)dstWidth
                                       dstHeight:(NSInteger)dstHeight {
    @autoreleasepool {
        int sw = (int)srcWidth, sh = (int)srcHeight;
        int dw = (int)dstWidth, dh = (int)dstHeight;
        NSUInteger total = (NSUInteger)(sw * sh);

        cv::Mat src(sh, sw, CV_32F);
        for (NSUInteger i = 0; i < total; i++) {
            src.at<float>((int)i) = [srcMap[i] floatValue];
        }

        cv::Mat dst;
        cv::resize(src, dst, cv::Size(dw, dh), 0, 0, cv::INTER_CUBIC);

        NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:(NSUInteger)(dw * dh)];
        for (int i = 0; i < dw * dh; i++) {
            [result addObject:@(dst.at<float>(i))];
        }
        return [result copy];
    }
}

/// Ported from contours.py:171-200, 231
/// tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
/// tight_mask.sum(axis=0).max()  →  max column pixel count
+ (NSInteger)maxColumnThicknessOfContour:(NSArray<NSValue *> *)contour
                             boundingRect:(CGRect)rect {
    int w = (int)rect.size.width;
    int h = (int)rect.size.height;
    if (w <= 0 || h <= 0) return 0;

    // Build tight contour shifted to local bounding-rect origin.
    std::vector<cv::Point> pts;
    pts.reserve(contour.count);
    for (NSValue *v in contour) {
        CGPoint p = v.CGPointValue;
        pts.push_back(cv::Point((int)(p.x - rect.origin.x),
                                (int)(p.y - rect.origin.y)));
    }
    std::vector<std::vector<cv::Point>> contours = {pts};

    // Draw filled contour into a zeros mask of the bounding-rect size.
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::drawContours(mask, contours, 0, cv::Scalar(1), cv::FILLED);

    // Sum along rows (axis=0 in numpy), yielding a 1-row matrix of column sums.
    cv::Mat colSums;
    cv::reduce(mask, colSums, 0, cv::REDUCE_SUM, CV_32S);

    // Return the maximum column sum.
    double maxVal = 0.0;
    cv::minMaxLoc(colSums, nullptr, &maxVal);
    return (NSInteger)maxVal;
}

/// Ported from spans.py:186-188
/// (yvals * mask).sum(axis=0) / mask.sum(axis=0)  →  mean y per column
+ (NSArray<NSNumber *> *)columnMeansOfContour:(NSArray<NSValue *> *)contour
                                   boundingRect:(CGRect)rect {
    int w = (int)rect.size.width;
    int h = (int)rect.size.height;
    if (w <= 0 || h <= 0) return @[];

    // Build tight contour shifted to local bounding-rect origin.
    std::vector<cv::Point> pts;
    pts.reserve(contour.count);
    for (NSValue *v in contour) {
        CGPoint p = v.CGPointValue;
        pts.push_back(cv::Point((int)(p.x - rect.origin.x),
                                (int)(p.y - rect.origin.y)));
    }
    std::vector<std::vector<cv::Point>> contours = {pts};

    // Draw filled contour into a zeros mask.
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::drawContours(mask, contours, 0, cv::Scalar(1), cv::FILLED);

    // For each column x, compute mean y = sum(y * mask[y,x]) / sum(mask[y,x]).
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:w];
    for (int x = 0; x < w; x++) {
        double total = 0.0, count = 0.0;
        for (int y = 0; y < h; y++) {
            if (mask.at<uchar>(y, x)) {
                total += y;
                count += 1.0;
            }
        }
        [result addObject:@(count > 0 ? total / count : 0.0)];
    }
    return [result copy];
}

/// Draw contours on a color image for debug visualization.
/// Ported from contours.py:261-262, 199
+ (nullable UIImage *)drawContoursOnImage:(UIImage *)image
                                 contours:(NSArray<NSArray<NSValue *> *> *)contours
                                    color:(NSArray<NSNumber *> *)color
                                thickness:(NSInteger)thickness {
    @autoreleasepool {
        cv::Mat bgr = cvMatFromUIImage(image);

        cv::Scalar cvColor(
            [color[2] doubleValue],  // B
            [color[1] doubleValue],  // G
            [color[0] doubleValue]   // R
        );

        std::vector<std::vector<cv::Point>> cvContours;
        cvContours.reserve(contours.count);
        for (NSArray<NSValue *> *contour in contours) {
            std::vector<cv::Point> pts;
            pts.reserve(contour.count);
            for (NSValue *v in contour) {
                CGPoint p = v.CGPointValue;
                pts.push_back(cv::Point((int)p.x, (int)p.y));
            }
            cvContours.push_back(pts);
        }

        cv::drawContours(bgr, cvContours, -1, cvColor, (int)thickness);
        return UIImageFromBGRMat(bgr);
    }
}

/// Create a page mask with white filled rectangle, ignoring margins.
/// Ported from image.py:calculate_page_extents
+ (UIImage *)createPageMaskWithWidth:(NSInteger)width
                              height:(NSInteger)height
                             marginX:(NSInteger)marginX
                             marginY:(NSInteger)marginY {
    cv::Mat mask = cv::Mat::zeros((int)height, (int)width, CV_8UC1);
    cv::Point topLeft((int)marginX, (int)marginY);
    cv::Point bottomRight((int)(width - marginX), (int)(height - marginY));
    cv::rectangle(mask, topLeft, bottomRight, cv::Scalar(255), -1);
    return UIImageFromGrayMat(mask);
}

/// Compute binary detection mask for text or line contour extraction.
/// Ported from mask.py:67-100 (Mask.calculate)
+ (nullable UIImage *)computeDetectionMask:(UIImage *)colorImage
                                  pagemask:(UIImage *)pagemask
                                    isText:(BOOL)isText
                             adaptiveWinsz:(NSInteger)adaptiveWinsz {
    @autoreleasepool {
        // Step 1: convert colour → grayscale matching Python mask.py:77.
        // Python uses COLOR_RGB2GRAY on BGR data (cv2.imread returns BGR), which
        // applies weights 0.299*B + 0.587*G + 0.114*R instead of the correct
        // 0.114*B + 0.587*G + 0.299*R. We replicate this intentionally to match output.
        cv::Mat bgr = cvMatFromUIImage(colorImage);
        cv::Mat sgray;
        cv::cvtColor(bgr, sgray, cv::COLOR_RGB2GRAY);  // intentional: match Python's COLOR_RGB2GRAY on BGR

        // Step 2: adaptive threshold, binary inverse (mask.py:78-85)
        int bs = (int)adaptiveWinsz;
        if (bs % 2 == 0) bs += 1;
        if (bs < 3) bs = 3;
        double C = isText ? 25.0 : 7.0;
        cv::Mat mask;
        cv::adaptiveThreshold(sgray, mask, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                              bs, C);

        // Step 3: morphological operations (mask.py:89-98)
        if (isText) {
            // Text: dilate(box(9,1)), then erode(box(1,3))
            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
            cv::dilate(mask, mask, kernel1);
            cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
            cv::erode(mask, mask, kernel2);
        } else {
            // Lines: erode(box(3,1), iterations=3), then dilate(box(8,2))
            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
            cv::erode(mask, mask, kernel1, cv::Point(-1,-1), 3);
            cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 2));
            cv::dilate(mask, mask, kernel2);
        }

        // Step 4: apply pagemask — np.minimum(mask, pagemask) → bitwise AND (mask.py:100)
        cv::Mat pm = cvGrayMatFromUIImage(pagemask);
        cv::Mat result;
        cv::bitwise_and(mask, pm, result);

        return UIImageFromGrayMat(result);
    }
}

@end
