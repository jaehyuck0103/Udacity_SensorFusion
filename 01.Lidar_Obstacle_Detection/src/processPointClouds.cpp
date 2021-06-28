// PCL lib Functions for processing point clouds

#include "processPointClouds.h"

#include <unordered_set>

template <typename PointT>
void clusterHelper(
    const typename pcl::PointCloud<PointT>::Ptr cloud,
    const KdTree &tree,
    float distanceTol,
    int idx,
    std::vector<bool> &processed,
    std::vector<int> &cluster) {

    processed.at(idx) = true;
    cluster.push_back(idx);
    std::vector<int> nearby =
        tree.search({cloud->at(idx).x, cloud->at(idx).y, cloud->at(idx).z}, distanceTol);

    for (int each : nearby) {
        if (!processed.at(each)) {
            clusterHelper<PointT>(cloud, tree, distanceTol, each, processed, cluster);
        }
    }
}

template <typename PointT>
std::tuple<float, float, float, float>
getPlaneCoeff(const PointT &pt1, const PointT &pt2, const PointT &pt3) {
    float A = (pt2.y - pt1.y) * (pt3.z - pt1.z) - (pt2.z - pt1.z) * (pt3.y - pt1.y);
    float B = (pt2.z - pt1.z) * (pt3.x - pt1.x) - (pt2.x - pt1.x) * (pt3.z - pt1.z);
    float C = (pt2.x - pt1.x) * (pt3.y - pt1.y) - (pt2.y - pt1.y) * (pt3.x - pt1.x);
    float D = -(A * pt1.x + B * pt1.y + C * pt1.z);

    return {A, B, C, D};
}

// constructor:
template <typename PointT> ProcessPointClouds<PointT>::ProcessPointClouds() {}

// de-constructor:
template <typename PointT> ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud) {
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    float filterRes,
    Eigen::Vector4f minPoint,
    Eigen::Vector4f maxPoint) {

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: voxel grid point reduction
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> vg; // voxel 당 point 하나씩만 남긴다.
    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    // TODO:: Region Based Filtering
    typename pcl::PointCloud<PointT>::Ptr cloudRegion(new pcl::PointCloud<PointT>);
    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloudFiltered);
    region.filter(*cloudRegion);

    // TODO:: Remove ego car roof points
    std::vector<int> indices;
    pcl::CropBox<PointT> roof(true);
    roof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
    roof.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    for (int point : indices)
        inliers->indices.push_back(point);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudRegion);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloudRegion);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SeparateClouds(
    pcl::PointIndices::Ptr inliers,
    typename pcl::PointCloud<PointT>::Ptr cloud) {

    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    // ExtractIndices를 사용하지 않고, 단순히 loop 돌면서 push_back 하는 방법도 있다.
    typename pcl::PointCloud<PointT>::Ptr obstCloud{new pcl::PointCloud<PointT>};
    typename pcl::PointCloud<PointT>::Ptr planeCloud{new pcl::PointCloud<PointT>};
    pcl::ExtractIndices<PointT> extract;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    extract.setNegative(true);
    extract.filter(*obstCloud);

    extract.setNegative(false);
    extract.filter(*planeCloud);

    return {obstCloud, planeCloud};
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SegmentPlane(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    int maxIterations,
    float distanceThreshold) {

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in this function to find inliers for the cloud.
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers{new pcl::PointIndices()};
    pcl::ModelCoefficients::Ptr coefficients{new pcl::ModelCoefficients()};

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    // Segment the largest planer component from the input cloud
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        std::cout << "Could not estimate a planar model for the given dataset.\n";
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return SeparateClouds(inliers, cloud);
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SegmentPlaneWithoutPCL(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    int maxIterations,
    float distanceThreshold) {

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Ransac 3D
    std::unordered_set<int> inliersResult;
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Randomly sample subset and fit plane (Ax + By + Cz + D = 0)
        std::unordered_set<int> inliers;
        while (inliers.size() < 3) {
            inliers.insert(rand() % cloud->size());
        }
        std::vector point_idxs(inliers.begin(), inliers.end());
        auto [A, B, C, D] = getPlaneCoeff(
            cloud->at(point_idxs.at(0)),
            cloud->at(point_idxs.at(1)),
            cloud->at(point_idxs.at(2)));

        // Measure distance between every point and fitted line
        // If distance is smaller than threshold count it as inlier
        for (size_t idx = 0; idx < cloud->size(); ++idx) {
            const auto &pt = cloud->at(idx);
            float distance = abs(A * pt.x + B * pt.y + C * pt.z + D) / sqrt(A * A + B * B + C * C);
            if (distance <= distanceThreshold) {
                inliers.insert(idx);
            }
        }

        // Return indicies of inliers from fitted line with most inliers
        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    pcl::PointIndices::Ptr inliers_pcl = pcl::make_shared<pcl::PointIndices>();
    inliers_pcl->indices.insert(
        inliers_pcl->indices.end(),
        inliersResult.begin(),
        inliersResult.end());
    return SeparateClouds(inliers_pcl, cloud);
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    float clusterTolerance,
    int minSize,
    int maxSize) {

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (const auto &indices : cluster_indices) {
        typename pcl::PointCloud<PointT>::Ptr thisCluster{new pcl::PointCloud<PointT>};
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(pcl::make_shared<const pcl::PointIndices>(indices));
        extract.filter(*thisCluster);

        clusters.push_back(thisCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found "
              << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::ClusteringWithoutPCL(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    float clusterTolerance,
    int minSize,
    int maxSize) {

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    // Construct KdTree
    KdTree tree;
    for (size_t i = 0; i < cloud->size(); i++)
        tree.insert({cloud->points[i].x, cloud->points[i].y, cloud->points[i].z}, i);

    // Euclidean Clusterirng
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    std::vector<bool> processed(cloud->size(), false);
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (processed.at(i)) {
            continue;
        }

        std::vector<int> new_cluster_indices;
        clusterHelper<PointT>(cloud, tree, clusterTolerance, i, processed, new_cluster_indices);
        if (static_cast<int>(new_cluster_indices.size()) >= minSize &&
            static_cast<int>(new_cluster_indices.size()) <= maxSize) {
            typename pcl::PointCloud<PointT>::Ptr thisCluster{new pcl::PointCloud<PointT>};
            for (int idx : new_cluster_indices) {
                thisCluster->emplace_back(cloud->at(idx));
            }
            clusters.push_back(thisCluster);
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found "
              << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster) {

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    std::string file) {
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file) {

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath) {

    std::vector<boost::filesystem::path> paths(
        boost::filesystem::directory_iterator{dataPath},
        boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}
