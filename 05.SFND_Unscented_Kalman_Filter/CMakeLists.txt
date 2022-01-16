cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

project(playback)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE third_party/eigen-3.4.0)

add_executable(ukf_highway src/main.cpp src/ukf.cpp src/tools.cpp
                           src/render/render.cpp)
target_link_libraries(ukf_highway ${PCL_LIBRARIES} eigen)
