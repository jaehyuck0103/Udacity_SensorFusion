// PCL lib Functions for processing point clouds

#pragma once

#include "render/box.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <string>
#include <vector>

template <typename PointT> class ProcessPointClouds {
  public:
    // constructor
    ProcessPointClouds();
    // deconstructor
    ~ProcessPointClouds();

    void numPoints(typename pcl::PointCloud<PointT>::Ptr cloud);

    typename pcl::PointCloud<PointT>::Ptr FilterCloud(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        float filterRes,
        Eigen::Vector4f minPoint,
        Eigen::Vector4f maxPoint);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
    SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
    SegmentPlane(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        int maxIterations,
        float distanceThreshold);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
    SegmentPlaneWithoutPCL(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        int maxIterations,
        float distanceThreshold);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        float clusterTolerance,
        int minSize,
        int maxSize);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> ClusteringWithoutPCL(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        float clusterTolerance,
        int minSize,
        int maxSize);

    Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);

    void savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file);

    typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file);

    std::vector<boost::filesystem::path> streamPcd(std::string dataPath);
};

// Structure to represent node of kd tree
struct Node {
    std::vector<float> point;
    int id;
    Node *left;
    Node *right;

    Node(std::vector<float> arr, int setId) : point(arr), id(setId), left(NULL), right(NULL) {}

    ~Node() {
        delete left;
        delete right;
    }
};

struct KdTree {
    Node *root;

    KdTree() : root(NULL) {}

    ~KdTree() { delete root; }

    void insert(std::vector<float> point, int id) {
        Node **curNode = &root;
        for (int depth = 0; *curNode; ++depth) {
            if (point.at(depth % 3) < (*curNode)->point.at(depth % 3)) {
                curNode = &((*curNode)->left);
            } else {
                curNode = &((*curNode)->right);
            }
        }
        *curNode = new Node(point, id);
    }

    void searchHelper(
        std::vector<float> target,
        Node *node,
        float distanceTol,
        int depth,
        std::vector<int> &ids) const {

        if (!node) {
            return;
        }

        float x_diff = target.at(0) - node->point.at(0);
        float y_diff = target.at(1) - node->point.at(1);
        float z_diff = target.at(2) - node->point.at(2);

        if (abs(x_diff) < distanceTol && abs(y_diff) < distanceTol && abs(z_diff) < distanceTol) {
            float distance = sqrt(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2));
            if (distance < distanceTol) {
                ids.push_back(node->id);
            }
        }

        if ((target.at(depth % 3) - distanceTol) < node->point.at(depth % 3)) {
            searchHelper(target, node->left, distanceTol, depth + 1, ids);
        }
        if ((target.at(depth % 3) + distanceTol) > node->point.at(depth % 3)) {
            searchHelper(target, node->right, distanceTol, depth + 1, ids);
        }
    }

    // Return a list of point ids in the tree that are within distance of target
    std::vector<int> search(std::vector<float> target, float distanceTol) const {
        std::vector<int> ids;
        searchHelper(target, root, distanceTol, 0, ids);

        return ids;
    }
};
