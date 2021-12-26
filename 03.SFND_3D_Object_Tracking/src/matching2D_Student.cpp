#include "matching2D.hpp"

#include <chrono>
#include <numeric>

using sc = std::chrono::system_clock;
using duration_ms = std::chrono::duration<double, std::milli>;

int getNormType(const std::string &descriptorType) {
    if (descriptorType == "BRISK") {
        return cv::NORM_HAMMING;
    } else if (descriptorType == "BRIEF") {
        return cv::NORM_HAMMING;
    } else if (descriptorType == "ORB") {
        return cv::NORM_HAMMING;
    } else if (descriptorType == "FREAK") {
        return cv::NORM_HAMMING;
    } else if (descriptorType == "AKAZE") {
        return cv::NORM_HAMMING;
    } else if (descriptorType == "SIFT") {
        return cv::NORM_L2;
    } else {
        std::cout << "Invalid descriptorType: " << descriptorType << "\n";
        std::abort();
    }
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(
    cv::Mat &descSource,
    cv::Mat &descRef,
    std::vector<cv::DMatch> &matches,
    std::string descriptorType,
    std::string matcherType,
    std::string selectorType) {

    cv::Ptr<cv::DescriptorMatcher> matcher;

    // configure matcher
    if (matcherType == "MAT_BF") {
        int normType = getNormType(descriptorType);
        matcher = cv::BFMatcher::create(normType, false);
    } else if (matcherType == "MAT_FLANN") {
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    } else {
        std::cout << "Invalid matcherType: " << matcherType << "\n";
        std::abort();
    }

    // perform matching task
    if (selectorType == "SEL_NN") { // nearest neighbor (best match)
        // Finds the best match for each descriptor in desc1
        matcher->match(descSource, descRef, matches);
    } else if (selectorType == "SEL_KNN") { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // filter matches using descriptor distance ratio test
        constexpr double distanceRatio = 0.8;
        for (const auto &knn : knn_matches) {
            if (knn[0].distance < distanceRatio * knn[1].distance) {
                matches.push_back(knn[0]);
            }
        }
    } else {
        std::cout << "Invalid selectorType: " << selectorType << "\n";
        std::abort();
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &img,
    cv::Mat &descriptors,
    std::string descriptorType) {

    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType == "BRISK") {
        extractor = cv::BRISK::create();
    } else if (descriptorType == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType == "ORB") {
        extractor = cv::ORB::create();
    } else if (descriptorType == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType == "AKAZE") {
        extractor = cv::AKAZE::create();
    } else if (descriptorType == "SIFT") {
        extractor = cv::SIFT::create();
    } else {
        std::cout << "Invalid descriptorType: " << descriptorType << "\n";
        std::abort();
    }

    // perform feature description
    const sc::time_point t_start = sc::now();
    extractor->compute(img, keypoints, descriptors);
    const duration_ms duration_desc = sc::now() - t_start;
    std::cout << descriptorType << " descriptor extraction in " << duration_desc.count()
              << " ms\n";

    return duration_desc.count();
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {

    // compute detector parameters based on image size
    int blockSize = 4; //  size of an average block for computing a derivative covariation matrix
                       //  over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        cv::Mat(),
        blockSize,
        false,
        k);

    // add corners to result vector
    for (const auto &corner : corners) {
        keypoints.emplace_back(corner.x, corner.y, blockSize);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Add corners to result vector with NMS
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during
                             // non-maxima suppression
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse) { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // Perform NMS in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response >
                            (*it)
                                .response) { // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                // only add new key point if no overlap has been found in previous in NMS
                if (!bOverlap) {
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
}

void detKeypointsModern(
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &img,
    std::string detectorType) {

    /// string-based selection based on detectorType / -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType == "FAST") {
        detector = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
    } else if (detectorType == "BRISK") {
        detector = cv::BRISK::create();
    } else if (detectorType == "ORB") {
        detector = cv::ORB::create();
    } else if (detectorType == "AKAZE") {
        detector = cv::AKAZE::create();
    } else if (detectorType == "SIFT") {
        detector = cv::SIFT::create();
    } else {
        std::cout << "Invalid detectorType: " << detectorType << "\n";
        std::abort();
    }

    detector->detect(img, keypoints);
}

double detKeypoints(
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &img,
    std::string detectorType,
    bool bVis) {

    const sc::time_point t_start = sc::now();

    /// Detect Keypoints
    if (detectorType == "SHITOMASI") {
        detKeypointsShiTomasi(keypoints, img);
    } else if (detectorType == "HARRIS") {
        detKeypointsHarris(keypoints, img);
    } else {
        detKeypointsModern(keypoints, img, detectorType);
    }

    // Measure time for detection
    const duration_ms duration_det = sc::now() - t_start;
    std::cout << detectorType << " corners detection with n=" << keypoints.size()
              << " keypoints in " << duration_det.count() << " ms\n";

    // Visualiztion
    if (bVis) {
        const std::string windowName = detectorType + " keypoint detector";
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img,
            keypoints,
            visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName); // , 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return duration_det.count();
}
