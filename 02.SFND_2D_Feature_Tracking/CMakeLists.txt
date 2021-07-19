cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(camera_fusion)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

find_package(OpenCV 4.1 REQUIRED)

# Executable for create matrix exercise
add_executable(2D_feature_tracking src/matching2D_Student.cpp
                                   src/MidTermProject_Camera_Student.cpp)
target_include_directories(2D_feature_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_directories(2D_feature_tracking PRIVATE ${OpenCV_LIBRARY_DIRS})
target_link_libraries(2D_feature_tracking ${OpenCV_LIBRARIES})
