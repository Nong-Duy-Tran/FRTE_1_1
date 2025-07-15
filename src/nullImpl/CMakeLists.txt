project(nullImpl)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/onnxruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/opencv2
)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/opencv2
)

# # Build the shared libraries
# add_library (frvt_11_null_001 SHARED nullimplfrvt11.cpp)

# Add the main executable
add_executable(main_exec main.cpp RetinaFace.cpp)

target_link_libraries(main_exec libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so libopencv_dnn.so libopencv_highgui.so libonnxruntime.so)
