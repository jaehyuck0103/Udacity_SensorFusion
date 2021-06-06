cmake_minimum_required(VERSION 3.9)

project(playback)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(quizCluster cluster.cpp ../../render/render.cpp)
target_link_libraries(quizCluster ${PCL_LIBRARIES})
