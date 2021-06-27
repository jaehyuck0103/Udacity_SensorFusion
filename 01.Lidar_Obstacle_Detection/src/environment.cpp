/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "processPointClouds.h"
#include "render/render.h"
#include "sensors/lidar.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

std::vector<Car> initHighway(bool renderScene, pcl::visualization::PCLVisualizer::Ptr &viewer) {

    Car egoCar(Vect3(0, 0, 0), Vect3(4, 2, 2), Color(0, 1, 0), "egoCar");
    Car car1(Vect3(15, 0, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car1");
    Car car2(Vect3(8, -4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car2");
    Car car3(Vect3(-12, 4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car3");

    std::vector<Car> cars;
    cars.push_back(egoCar);
    cars.push_back(car1);
    cars.push_back(car2);
    cars.push_back(car3);

    if (renderScene) {
        renderHighway(viewer);
        egoCar.render(viewer);
        car1.render(viewer);
        car2.render(viewer);
        car3.render(viewer);
    }

    return cars;
}

void simpleHighway(pcl::visualization::PCLVisualizer::Ptr &viewer) {
    // ----------------------------------------------------
    // -----Open 3D viewer and display simple highway -----
    // ----------------------------------------------------

    // RENDER OPTIONS
    bool renderScene = false; // true;
    std::vector<Car> cars = initHighway(renderScene, viewer);

    // Create lidar sensor
    Lidar *lidar = new Lidar(cars, 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud = lidar->scan();
    // renderRays(viewer, lidar->position, inputCloud);
    // renderPointCloud(viewer, inputCloud, "inputCloud");

    // Segment plane
    ProcessPointClouds<pcl::PointXYZ> pointProcessor;
    auto [obstCloud, planeCloud] = pointProcessor.SegmentPlane(inputCloud, 100, 0.2);
    // renderPointCloud(viewer, obstCloud, "obstCloud", Color(1, 0, 0));
    // renderPointCloud(viewer, planeCloud, "planeCloud", Color(0, 1, 0));

    // Clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters =
        pointProcessor.Clustering(obstCloud, 1.0, 3, 30);

    std::vector<Color> colors = {Color(1, 0, 0), Color(0, 1, 0), Color(0, 0, 1)};

    int clusterId = 0;
    for (pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : cloudClusters) {
        std::cout << "cluster size " << cluster->size() << std::endl;
        renderPointCloud(
            viewer,
            cluster,
            "obstCloud" + std::to_string(clusterId),
            colors[clusterId % colors.size()]);

        Box box = pointProcessor.BoundingBox(cluster);
        renderBox(viewer, box, clusterId);

        ++clusterId;
    }
}

void cityBlock(
    pcl::visualization::PCLVisualizer::Ptr &viewer,
    ProcessPointClouds<pcl::PointXYZI> &pointProcessorI,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &inputCloud) {
    // ----------------------------------------------------
    // -----Open 3D viewer and display City Block     -----
    // ----------------------------------------------------

    // Filtering
    pcl::PointCloud<pcl::PointXYZI>::Ptr filteredCloud;
    filteredCloud =
        pointProcessorI.FilterCloud(inputCloud, 0.2, {-50, -15, -10, 1}, {50, 15, 10, 1});
    // renderPointCloud(viewer, filteredCloud, "filteredCloud");

    // Segment plane
    // auto [obstCloud, planeCloud] = pointProcessorI.SegmentPlane(filteredCloud, 100, 0.2);
    auto [obstCloud, planeCloud] = pointProcessorI.SegmentPlaneWithoutPCL(filteredCloud, 200, 0.2);
    renderPointCloud(viewer, obstCloud, "obstCloud", Color(0.2, 0.2, 0.2));
    renderPointCloud(viewer, planeCloud, "planeCloud", Color(0, 1, 0));

    // Clustering
    // std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters =
    //    pointProcessorI.Clustering(obstCloud, 0.4, 20, 1000);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters =
        pointProcessorI.ClusteringWithoutPCL(obstCloud, 0.4, 10, 1000);

    std::vector<Color> colors = {Color(1, 0, 0), Color(1, 1, 0), Color(0, 0, 1)};

    int clusterId = 0;
    for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters) {
        std::cout << "cluster size " << cluster->size() << std::endl;
        renderPointCloud(
            viewer,
            cluster,
            "obstCloud" + std::to_string(clusterId),
            colors[clusterId % colors.size()]);

        Box box = pointProcessorI.BoundingBox(cluster);
        renderBox(viewer, box, clusterId);

        ++clusterId;
    }
}

// setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr &viewer) {

    viewer->setBackgroundColor(0, 0, 0);

    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;

    switch (setAngle) {
    case CameraAngle::XY:
        viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0);
        break;
    case CameraAngle::TopDown:
        viewer->setCameraPosition(0, 0, distance, 1, 0, 1);
        break;
    case CameraAngle::Side:
        viewer->setCameraPosition(0, -distance, 0, 0, 0, 1);
        break;
    case CameraAngle::FPS:
        viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if (setAngle != CameraAngle::FPS)
        viewer->addCoordinateSystem(1.0);
}

int main() {
    std::cout << "starting enviroment" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));
    initCamera(CameraAngle::XY, viewer);

    /*
    // -------------------------------------------------
    // Simple Highway
    // -------------------------------------------------
    simpleHighway(viewer);
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }
    */

    // -------------------------------------------------
    // City Block
    // -------------------------------------------------
    ProcessPointClouds<pcl::PointXYZI> pointProcessorI;
    std::vector<boost::filesystem::path> stream =
        pointProcessorI.streamPcd("../src/sensors/data/pcd/data_1");
    auto streamIterator = stream.begin();
    pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;

    while (!viewer->wasStopped()) {
        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        // Load pcd and run obstacle detection process
        inputCloudI = pointProcessorI.loadPcd((*streamIterator).string());
        cityBlock(viewer, pointProcessorI, inputCloudI);

        streamIterator++;
        if (streamIterator == stream.end())
            streamIterator = stream.begin();

        viewer->spinOnce();
    }
}
