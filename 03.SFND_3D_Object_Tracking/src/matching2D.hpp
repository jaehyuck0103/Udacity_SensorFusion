#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

// Keypoints
double detKeypoints(
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &img,
    std::string detectorType,
    bool bVis = false);

// Descriptor
double descKeypoints(
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &img,
    cv::Mat &descriptors,
    std::string descriptorType);

// Matching
void matchDescriptors(
    cv::Mat &descSource,
    cv::Mat &descRef,
    std::vector<cv::DMatch> &matches,
    std::string descriptorType,
    std::string matcherType,
    std::string selectorType);
