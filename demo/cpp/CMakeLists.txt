cmake_minimum_required(VERSION 2.8)
project(CAFFE_SSD_DEMO)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} not suppoet C++11.")
endif()

find_package(OpenCV 3.4.1)
find_package(CUDA)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(DEMO_FILE
    main.cpp
    cf_ssd_util.cpp
    cf_ssd_detector.cpp
    inf_face_detection_ssd_api.cpp
    )

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../../caffe_ssd/include
    ${OpneCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
    ${PROJECT_SOURCE_DIR}/../../caffe_ssd/build/lib
    )

set(LINKER_EXT_LIBS
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    libcaffe.so
    "/usr/lib/x86_64-linux-gnu/libglog.so.0"
)

add_executable(demo ${DEMO_FILE})
target_link_libraries(demo ${LINKER_EXT_LIBS})