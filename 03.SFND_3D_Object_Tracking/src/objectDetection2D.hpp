#pragma once

#include "dataStructures.h"

#include <opencv2/core.hpp>

#include <stdio.h>

void detectObjects(
    cv::Mat &img,
    std::vector<BoundingBox> &bBoxes,
    float confThreshold,
    float nmsThreshold,
    std::string basePath,
    std::string classesFile,
    std::string modelConfiguration,
    std::string modelWeights,
    bool bVis);
