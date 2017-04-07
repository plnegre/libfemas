cmake_minimum_required(VERSION 2.8.3)
project(libfemas)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    tf
    image_geometry
    cmake_modules)

# Dependencies - OpenCV:
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

catkin_package(INCLUDE_DIRS include LIBRARIES femas)

# Add the Image Hashing library
add_library(femas
            src/extractor.cpp
            src/matcher.cpp
            src/estimator.cpp)
target_link_libraries(femas ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

# Include directories
include_directories(${catkin_INCLUDE_DIRS} include)