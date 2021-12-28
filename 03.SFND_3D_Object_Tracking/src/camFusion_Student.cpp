#include "camFusion.hpp"
#include "dataStructures.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(
    std::vector<BoundingBox> &boundingBoxes,
    std::vector<LidarPoint> &lidarPoints,
    float shrinkFactor,
    cv::Mat &P_rect_xx,
    cv::Mat &R_rect_xx,
    cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        std::vector<std::vector<BoundingBox>::iterator>
            enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin();
             it2 != boundingBoxes.end();
             ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around
            // the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/*
 * The show3DObjects() function below can handle different output image sizes, but the text output
 * has been manually tuned to fit the 2000x2000 size. However, you can make this function work for
 * other sizes too. For instance, to use a 1000x1000 size, adjusting the text positions by dividing
 * them by 2.
 */
void show3DObjects(
    std::vector<BoundingBox> &boundingBoxes,
    cv::Size worldSize,
    cv::Size imageSize,
    bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor =
            cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2.0;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(
            topviewImg,
            cv::Point(left, top),
            cv::Point(right, bottom),
            cv::Scalar(0, 0, 0),
            2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(
            topviewImg,
            str1,
            cv::Point2f(left - 250, bottom + 50),
            cv::FONT_ITALIC,
            2,
            currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(
            topviewImg,
            str2,
            cv::Point2f(left - 250, bottom + 125),
            cv::FONT_ITALIC,
            2,
            currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (int i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(
            topviewImg,
            cv::Point(0, y),
            cv::Point(imageSize.width, y),
            cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(
    BoundingBox &boundingBox,
    std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> &kptMatches) {

    // Find all matches where currKeypts are included in BB.
    std::vector<std::pair<cv::DMatch, float>> inMatches; // match and distance
    for (const auto &match : kptMatches) {
        const auto &prevMatchPt = kptsPrev.at(match.queryIdx).pt;
        const auto &currMatchPt = kptsCurr.at(match.trainIdx).pt;

        if (boundingBox.roi.contains(currMatchPt)) {
            inMatches.emplace_back(match, cv::norm(prevMatchPt - currMatchPt));
        }
    }

    // Sort matches by the distance bw kpts
    std::sort(inMatches.begin(), inMatches.end(), [](const auto &a, const auto &b) {
        return std::get<1>(a) < std::get<1>(b);
    });

    // Find median distance.
    float medianDist = std::get<1>(inMatches.at(inMatches.size() / 2));

    // To eliminate noise, exclude the matches that have fartest 20% distances to median.
    std::sort(inMatches.begin(), inMatches.end(), [medianDist](const auto &a, const auto &b) {
        return std::abs(std::get<1>(a) - medianDist) < std::abs(std::get<1>(b) - medianDist);
    });
    inMatches.resize(inMatches.size() - inMatches.size() * 0.2);

    for (const auto &inMatch : inMatches) {
        boundingBox.kptMatches.push_back(std::get<0>(inMatch));
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
double computeTTCCamera(
    std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> kptMatches,
    double frameRate) {

    // compute distance ratios between all matched keypoints
    std::vector<double> distRatios;
    for (const auto &matchOuter : kptMatches) { // outer match loop

        cv::KeyPoint kpOuterPrev = kptsPrev.at(matchOuter.queryIdx);
        cv::KeyPoint kpOuterCurr = kptsCurr.at(matchOuter.trainIdx);

        for (const auto &matchInner : kptMatches) { // inner match loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerPrev = kptsPrev.at(matchInner.queryIdx);
            cv::KeyPoint kpInnerCurr = kptsCurr.at(matchInner.trainIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() &&
                distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        return -1;
    }

    // Compute camera-based TTC from distance ratios
    // Use median diatance ratio to reduce noise.
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios.at(distRatios.size() / 2);

    return -(1 / frameRate) / (1 - medianDistRatio);
}

double computeTTCLidar(
    std::vector<LidarPoint> &lidarPointsPrev,
    std::vector<LidarPoint> &lidarPointsCurr,
    double frameRate) {

    // Sort lidar points by distance
    std::sort(
        lidarPointsPrev.begin(),
        lidarPointsPrev.end(),
        [](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; });

    std::sort(
        lidarPointsCurr.begin(),
        lidarPointsCurr.end(),
        [](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; });

    // To eliminate noise, exclude the nearest 10% points.
    double minXPrev = lidarPointsPrev.at(static_cast<int>(lidarPointsPrev.size() * 0.1)).x;
    double minXCurr = lidarPointsCurr.at(static_cast<int>(lidarPointsCurr.size() * 0.1)).x;

    // Estimate TTC
    return minXCurr / (minXPrev - minXCurr) / frameRate;
}

void matchBoundingBoxes(
    std::vector<cv::DMatch> &matches,
    std::map<int, int> &bbBestMatches,
    DataFrame &prevFrame,
    DataFrame &currFrame) {

    for (const auto &currBB : currFrame.boundingBoxes) {
        // Count # of times each prevBB matches the currBB.
        std::vector<int> count(prevFrame.boundingBoxes.size());
        for (const auto &match : matches) {
            const auto &prevMatchPt = prevFrame.keypoints.at(match.queryIdx).pt;
            const auto &currMatchPt = currFrame.keypoints.at(match.trainIdx).pt;

            if (!currBB.roi.contains(currMatchPt)) {
                continue;
            }
            for (const auto &prevBB : prevFrame.boundingBoxes) {
                if (prevBB.roi.contains(prevMatchPt)) {
                    count.at(prevBB.boxID) += 1;
                }
            }
        }

        // Find the most matched prevBB.
        int maxIdx = std::distance(count.begin(), std::max_element(count.begin(), count.end()));
        if (count.at(maxIdx) > 0) {
            bbBestMatches[currBB.boxID] = maxIdx;
        }
    }
}
