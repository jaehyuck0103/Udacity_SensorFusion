/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "dataStructures.h"
#include "matching2D.hpp"

// Returns: seqKeypoints, numMatches, durations
std::tuple<std::vector<std::vector<cv::KeyPoint>>, std::vector<size_t>, std::vector<double>>
runSeq(
    std::string detectorType,
    std::string descriptorType,
    std::string matcherType,
    std::string selectorType,
    bool bVis) {

    std::cout << "\n\n\n\n\n\n";
    std::cout << "-----------------------\n";
    std::cout << "Detector: " << detectorType << "\n";
    std::cout << "Descriptor: " << descriptorType << "\n";
    std::cout << "Matcher: " << matcherType << "\n";
    std::cout << "Selector: " << selectorType << "\n";
    std::cout << "Visualization: " << (bVis ? "True" : "False") << "\n";
    std::cout << "-----------------------\n";

    if ((descriptorType == "AKAZE" && detectorType != "AKAZE") ||
        (descriptorType == "ORB" && detectorType == "SIFT")) {
        std::cout << "Invalid combination of detector and descriptor\n";
        return {};
    }

    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have
                           // identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    //// no. of images which are held in memory (ring buffer) at the same time
    size_t dataBufferSize = 2;
    //// list of data frames which are held in memory at the same time
    std::vector<DataFrame> dataBuffer;

    // Returns for benchmark
    std::vector<std::vector<cv::KeyPoint>> seqKeypoints;
    std::vector<size_t> numMatches;
    std::vector<double> durations;

    /* MAIN LOOP OVER ALL IMAGES */
    for (int imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {

        std::cout << "\n";

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // pop the first image from dataBuffer
        if (dataBuffer.size() == dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        } else if (dataBuffer.size() > dataBufferSize) {
            std::cout << "Impossible case: dataBuffer.size() > dataBufferSize\n";
            std::abort();
        }

        // push_back a new image into dataBuffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        std::cout << "#1 : LOAD IMAGE INTO BUFFER done\n";

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable
        /// string-based selection based on detectorType
        ///  -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double duration_det_ms = detKeypoints(keypoints, imgGray, detectorType, bVis);
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle) {
            keypoints.erase(
                std::remove_if(
                    keypoints.begin(),
                    keypoints.end(),
                    [&vehicleRect](const auto keypt) { return !vehicleRect.contains(keypt.pt); }),
                keypoints.end());
        }
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            // there is no response info,
            // so keep the first 50 as they are sorted in descending quality order
            if (detectorType == "SHITOMASI") {
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!\n";
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        seqKeypoints.push_back(keypoints);
        std::cout << "#2 : DETECT KEYPOINTS done\n";

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable
        /// string-based selection based on descriptorType / -> BRISK, BRIEF, ORB, FREAK, AKAZE,
        /// SIFT

        cv::Mat descriptors;
        double duration_desc_ms = descKeypoints(
            (dataBuffer.end() - 1)->keypoints,
            (dataBuffer.end() - 1)->cameraImg,
            descriptors,
            descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        durations.push_back(duration_det_ms + duration_desc_ms);
        std::cout << "#3 : EXTRACT DESCRIPTORS done\n";

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */
            std::vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio
            /// filtering with t=0.8 in file matching2D.cpp
            matchDescriptors(
                (dataBuffer.end() - 2)->descriptors,
                (dataBuffer.end() - 1)->descriptors,
                matches,
                descriptorType,
                matcherType,
                selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            numMatches.push_back(matches.size());
            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done\n";

            // visualize matches between current and previous image
            if (bVis) {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches(
                    (dataBuffer.end() - 2)->cameraImg,
                    (dataBuffer.end() - 2)->keypoints,
                    (dataBuffer.end() - 1)->cameraImg,
                    (dataBuffer.end() - 1)->keypoints,
                    matches,
                    matchImg,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                std::string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                std::cout << "Press key to continue to next image\n";
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

    return {seqKeypoints, numMatches, durations};
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

    if (argc == 5) { // Specific Mode with Visualization

        /// SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        std::string detectorType = argv[1];

        /// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        std::string descriptorType = argv[2];

        /// MAT_BF, MAT_FLANN
        std::string matcherType = argv[3];

        /// SEL_NN, SEL_KNN
        std::string selectorType = argv[4];

        runSeq(detectorType, descriptorType, matcherType, selectorType, true);

    } else { // Benchmark Mode

        const std::vector<std::string> detectorTypes =
            {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
        const std::vector<std::string> descriptorTypes =
            {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

        // MP7. number of keypoints, distribution of keypoint sizes
        {
            std::ofstream keypoints_csv("../keypoints.csv");
            keypoints_csv
                << "Each Element = "
                << "(Number of keypoints / Mean of keypoint sizes / Std of keypoint sizes)";
            keypoints_csv << "\n\n\n" << std::setw(10) << "";
            for (int i = 0; i < 10; ++i) {
                keypoints_csv << ",          image " << i;
            }
            for (const auto &detectorType : detectorTypes) {
                auto [seqKeypoints, _, __] =
                    runSeq(detectorType, "BRISK", "MAT_BF", "SEL_KNN", false);

                // Write keypoints.csv
                keypoints_csv << "\n" << std::setw(10) << detectorType;
                for (const auto &keypoints : seqKeypoints) {
                    float size_sum = 0;
                    for (const auto &pt : keypoints) {
                        size_sum += pt.size;
                    }
                    float size_mean = size_sum / keypoints.size();

                    float size_variance = 0;
                    for (const auto &pt : keypoints) {
                        size_variance += (pt.size - size_mean) * (pt.size - size_mean);
                    }
                    size_variance /= keypoints.size();
                    float size_stddev = sqrt(size_variance);

                    keypoints_csv << ",";
                    keypoints_csv << std::setw(5) << keypoints.size();
                    keypoints_csv << "/";
                    keypoints_csv << std::setw(5) << std::setprecision(1) << std::fixed
                                  << size_mean;
                    keypoints_csv << "/";
                    keypoints_csv << std::setw(5) << std::setprecision(1) << std::fixed
                                  << size_stddev;
                }
            }
        }

        // MP8. Count the number of matches
        {
            std::ofstream matches_csv("../matches.csv");
            matches_csv << "Average # of matches\n\n\n";
            matches_csv << std::setw(20) << "Detector\\Descriptor";
            for (const auto &descriptorType : descriptorTypes) {
                matches_csv << "," << std::setw(10) << descriptorType;
            }
            for (const auto &detectorType : detectorTypes) {
                matches_csv << "\n" << std::setw(20) << detectorType;
                for (const auto &descriptorType : descriptorTypes) {

                    auto [_, numMatches, __] =
                        runSeq(detectorType, descriptorType, "MAT_BF", "SEL_KNN", false);

                    float meanNumMatches =
                        std::accumulate(numMatches.begin(), numMatches.end(), 0.0) /
                        numMatches.size();

                    matches_csv << "," << std::setw(10) << std::setprecision(2) << std::fixed
                                << meanNumMatches;
                }
            }
        }

        // MP9. Measure time (keypoint detection + descriptor extraction)
        {
            std::ofstream time_csv("../time.csv");
            time_csv << "Average time for keypoint detection and descriptor extraction (ms)\n\n\n";
            time_csv << std::setw(10) << "";
            for (const auto &descriptorType : descriptorTypes) {
                time_csv << "," << std::setw(10) << descriptorType;
            }
            for (const auto &detectorType : detectorTypes) {
                time_csv << "\n" << std::setw(10) << detectorType;
                for (const auto &descriptorType : descriptorTypes) {
                    auto [_, __, durations] =
                        runSeq(detectorType, descriptorType, "MAT_BF", "SEL_KNN", false);

                    double meanDurations =
                        std::accumulate(durations.begin(), durations.end(), 0.0) /
                        durations.size();

                    time_csv << "," << std::setw(10) << std::setprecision(2) << std::fixed
                             << meanDurations;
                }
            }
        }
    }

    return 0;
}
