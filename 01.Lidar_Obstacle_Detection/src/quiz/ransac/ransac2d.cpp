/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../processPointClouds.h"
#include "../../render/render.h"
#include <unordered_set>
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData2D() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // Add inliers
    float scatter = 0.6;
    for (int i = -5; i < 5; i++) {
        double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        pcl::PointXYZ point;
        point.x = i + scatter * rx;
        point.y = i + scatter * ry;
        point.z = 0;

        cloud->points.push_back(point);
    }
    // Add outliers
    int numOutliers = 10;
    while (numOutliers--) {
        double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        pcl::PointXYZ point;
        point.x = 5 * rx;
        point.y = 5 * ry;
        point.z = 0;

        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D() {
    ProcessPointClouds<pcl::PointXYZ> pointProcessor;
    return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}

pcl::visualization::PCLVisualizer::Ptr initScene() {
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("2D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
    viewer->addCoordinateSystem(1.0);
    return viewer;
}

std::unordered_set<int>
Ransac2D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol) {
    std::unordered_set<int> inliersResult;
    srand(time(NULL));

    // TODO: Fill in this function

    // For max iterations
    while (maxIterations--) {
        // Randomly sample subset and fit line (Ax + By + C = 0)
        std::unordered_set<int> inliers;
        while (inliers.size() < 2) {
            inliers.insert(rand() % cloud->size());
        }
        std::vector point_idxs(inliers.begin(), inliers.end());
        float x1 = cloud->at(point_idxs.at(0)).x;
        float y1 = cloud->at(point_idxs.at(0)).y;
        float x2 = cloud->at(point_idxs.at(1)).x;
        float y2 = cloud->at(point_idxs.at(1)).y;
        float A = y1 - y2;
        float B = x2 - x1;
        float C = x1 * y2 - x2 * y1;

        // Measure distance between every point and fitted line
        // If distance is smaller than threshold count it as inlier
        for (int idx = 0; idx < cloud->size(); ++idx) {
            const auto &pt = cloud->at(idx);
            float distance = abs(A * pt.x + B * pt.y + C) / sqrt(A * A + B * B);
            if (distance <= distanceTol) {
                inliers.insert(idx);
            }
        }

        // Return indicies of inliers from fitted line with most inliers
        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }

    return inliersResult;
}

std::unordered_set<int>
Ransac3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol) {
    std::unordered_set<int> inliersResult;
    srand(time(NULL));

    // TODO: Fill in this function

    // For max iterations
    while (maxIterations--) {
        // Randomly sample subset and fit plane (Ax + By + Cz + D = 0)
        std::unordered_set<int> inliers;
        while (inliers.size() < 3) {
            inliers.insert(rand() % cloud->size());
        }
        std::vector point_idxs(inliers.begin(), inliers.end());
        float x1 = cloud->at(point_idxs.at(0)).x;
        float y1 = cloud->at(point_idxs.at(0)).y;
        float z1 = cloud->at(point_idxs.at(0)).z;
        float x2 = cloud->at(point_idxs.at(1)).x;
        float y2 = cloud->at(point_idxs.at(1)).y;
        float z2 = cloud->at(point_idxs.at(1)).z;
        float x3 = cloud->at(point_idxs.at(2)).x;
        float y3 = cloud->at(point_idxs.at(2)).y;
        float z3 = cloud->at(point_idxs.at(2)).z;
        float A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
        float B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
        float C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        float D = -(A * x1 + B * y1 + C * z1);

        // Measure distance between every point and fitted line
        // If distance is smaller than threshold count it as inlier
        for (int idx = 0; idx < cloud->size(); ++idx) {
            const auto &pt = cloud->at(idx);
            float distance = abs(A * pt.x + B * pt.y + C * pt.z + D) / sqrt(A * A + B * B + C * C);
            if (distance <= distanceTol) {
                inliers.insert(idx);
            }
        }

        // Return indicies of inliers from fitted line with most inliers
        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }

    return inliersResult;
}

int main() {

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

    // Create data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData2D();
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();

    // TODO: Change the max iteration and distance tolerance arguments for Ransac function
    std::unordered_set<int> inliers = Ransac2D(cloud, 50, 0.5);
    // std::unordered_set<int> inliers = Ransac3D(cloud, 100, 0.5);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

    for (int index = 0; index < cloud->points.size(); index++) {
        pcl::PointXYZ point = cloud->points[index];
        if (inliers.count(index))
            cloudInliers->points.push_back(point);
        else
            cloudOutliers->points.push_back(point);
    }

    // Render 2D point cloud with inliers and outliers
    if (inliers.size()) {
        renderPointCloud(viewer, cloudInliers, "inliers", Color(0, 1, 0));
        renderPointCloud(viewer, cloudOutliers, "outliers", Color(1, 0, 0));
    } else {
        renderPointCloud(viewer, cloud, "data");
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }
}
