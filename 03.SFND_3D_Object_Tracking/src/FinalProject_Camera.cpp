#include "camFusion.hpp"
#include "dataStructures.h"
#include "lidarData.hpp"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// Returns: LidarTTCs, CameraTTCs
std::tuple<std::vector<float>, std::vector<float>> runSeq(
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

    std::vector<float> ttcLidars;
    std::vector<float> ttcCameras;

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have
                           // identical naming convention)
    int imgEndIndex = 18;  // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4; // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    std::string yoloBasePath = dataPath + "dat/yolo/";
    std::string yoloClassesFile = yoloBasePath + "coco.names";
    std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    std::string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(
        3,
        4,
        cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(
        4,
        4,
        cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    //// no. of images which are held in memory (ring buffer) at the same time
    size_t dataBufferSize = 2;
    //// list of data frames which are held in memory at the same time
    std::vector<DataFrame> dataBuffer;

    /* MAIN LOOP OVER ALL IMAGES */
    for (int imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // pop the first image from dataBuffer
        if (dataBuffer.size() == dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        } else if (dataBuffer.size() > dataBufferSize) {
            std::cout << "Impossible case: dataBuffer.size() > dataBufferSize\n";
            std::abort();
        }

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done\n";

        /* DETECT & CLASSIFY OBJECTS */
        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects(
            (dataBuffer.end() - 1)->cameraImg,
            (dataBuffer.end() - 1)->boundingBoxes,
            confThreshold,
            nmsThreshold,
            yoloClassesFile,
            yoloModelConfiguration,
            yoloModelWeights,
            bVis);

        std::cout << "#2 : DETECT & CLASSIFY OBJECTS done\n";

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        std::string lidarFullFilename =
            imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        //// focus on ego lane
        float minZ = -1.5, maxZ = -0.9;
        float minX = 2.0, maxX = 20.0;
        float maxY = 2.0;
        float minR = 0.1;
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        std::cout << "#3 : CROP LIDAR POINTS done\n";

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D
                                   // object merging at the edges of an ROI
        clusterLidarWithROI(
            (dataBuffer.end() - 1)->boundingBoxes,
            (dataBuffer.end() - 1)->lidarPoints,
            shrinkFactor,
            P_rect_00,
            R_rect_00,
            RT);

        // Visualize 3D objects
        if (bVis) {
            show3DObjects(
                (dataBuffer.end() - 1)->boundingBoxes,
                cv::Size(4.0, 20.0),
                cv::Size(1000, 1000),
                true);
        }

        std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done\n";

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        detKeypoints(keypoints, imgGray, detectorType, bVis);

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            // there is no response info,
            // so keep the first 50 as they are sorted in descending quality order
            if (detectorType.compare("SHITOMASI") == 0) {
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!\n";
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        std::cout << "#5 : DETECT KEYPOINTS done\n";

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints(
            (dataBuffer.end() - 1)->keypoints,
            (dataBuffer.end() - 1)->cameraImg,
            descriptors,
            descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        std::cout << "#6 : EXTRACT DESCRIPTORS done\n";

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */
            std::vector<cv::DMatch> matches;
            matchDescriptors(
                (dataBuffer.end() - 2)->descriptors,
                (dataBuffer.end() - 1)->descriptors,
                matches,
                descriptorType,
                matcherType,
                selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done\n";

            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and
            /// previous frame (implement ->matchBoundingBoxes)
            // Associate bounding boxes between current and previous frame using keypoint matches
            std::map<int, int> bbBestMatches; // key: currIdx, val: prevIdx
            matchBoundingBoxes(
                matches,
                bbBestMatches,
                *(dataBuffer.end() - 2),
                *(dataBuffer.end() - 1));
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

            std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done\n";

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (const auto bbBestMatch : bbBestMatches) {
                // find bounding boxes associates with current match
                BoundingBox *currBB = nullptr;
                for (auto &bb : (dataBuffer.end() - 1)->boundingBoxes) {
                    // check wether current match partner corresponds to this BB
                    if (bbBestMatch.first == bb.boxID) {
                        currBB = &bb;
                    }
                }

                BoundingBox *prevBB = nullptr;
                for (auto &bb : (dataBuffer.end() - 2)->boundingBoxes) {
                    // check wether current match partner corresponds to this BB
                    if (bbBestMatch.second == bb.boxID) {
                        prevBB = &bb;
                    }
                }

                // compute TTC for current match
                if (currBB->lidarPoints.size() > 0 &&
                    prevBB->lidarPoints.size() > 0) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    // TASK FP.2 -> compute time-to-collision based on Lidar data
                    double ttcLidar =
                        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate);
                    ttcLidars.push_back(ttcLidar);
                    // TASK FP.3 -> assign enclosed keypoint matches to bounding box
                    clusterKptMatchesWithROI(
                        *currBB,
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->keypoints,
                        (dataBuffer.end() - 1)->kptMatches);
                    // TASK FP.4 -> compute time-to-collision based on camera
                    double ttcCamera = computeTTCCamera(
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->keypoints,
                        currBB->kptMatches,
                        sensorFrameRate);
                    ttcCameras.push_back(ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    if (bVis) {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(
                            visImg,
                            currBB->lidarPoints,
                            P_rect_00,
                            R_rect_00,
                            RT,
                            &visImg);
                        cv::rectangle(
                            visImg,
                            cv::Point(currBB->roi.x, currBB->roi.y),
                            cv::Point(
                                currBB->roi.x + currBB->roi.width,
                                currBB->roi.y + currBB->roi.height),
                            cv::Scalar(0, 255, 0),
                            2);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(
                            visImg,
                            str,
                            cv::Point2f(80, 50),
                            cv::FONT_HERSHEY_PLAIN,
                            2,
                            cv::Scalar(0, 0, 255));

                        std::string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        std::cout << "Press key to continue to next frame\n";
                        cv::waitKey(0);
                    }

                } // eof TTC computation
            }     // eof loop over all BB matches
        }

    } // eof loop over all images

    return {ttcLidars, ttcCameras};
}

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
        // FP5: Estimate Lidar TTCs
        {
            auto [lidarTTCs, _] = runSeq("FAST", "BRIEF", "MAT_BF", "SEL_KNN", false);

            std::ofstream csv("../LidarTTCs.csv");
            for (size_t i = 0; i < lidarTTCs.size(); ++i) {
                csv << "," << std::setw(7) << "image" << std::setw(3) << i + 1;
            }
            csv << "\n";

            for (const auto &ttc : lidarTTCs) {
                csv << "," << std::setw(10) << std::setprecision(2) << std::fixed << ttc;
            }
        }

        // FP6: Estimate Camera TTCs
        {
            std::ofstream csv("../CameraTTCs.csv");
            const std::vector<std::string> detectorTypes =
                {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
            const std::vector<std::string> descriptorTypes =
                {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

            csv << std::setw(10) << "Detector"
                << "," << std::setw(10) << "Descriptor";
            for (size_t i = 0; i < 18; ++i) {
                csv << "," << std::setw(7) << "image" << std::setw(3) << i + 1;
            }

            for (const auto &detectorType : detectorTypes) {
                for (const auto &descriptorType : descriptorTypes) {
                    csv << "\n";
                    csv << std::setw(10) << detectorType << "," << std::setw(10) << descriptorType;

                    auto [_, cameraTTCs] =
                        runSeq(detectorType, descriptorType, "MAT_BF", "SEL_KNN", false);

                    for (const auto &ttc : cameraTTCs) {
                        csv << "," << std::setw(10) << std::setprecision(2) << std::fixed << ttc;
                    }
                    if (cameraTTCs.empty()) {
                        for (size_t i = 0; i < 18; ++i) {
                            csv << "," << std::setw(10) << " ";
                        }
                    }
                }
            }
        }
    }
}
