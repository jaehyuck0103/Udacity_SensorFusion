cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(camera_fusion)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

find_package(OpenCV 4.1 REQUIRED)

add_executable(
  3D_object_tracking
  src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp
  src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_include_directories(3D_object_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_directories(3D_object_tracking PRIVATE ${OpenCV_LIBRARY_DIRS})
target_link_libraries(3D_object_tracking PRIVATE ${OpenCV_LIBRARIES})
