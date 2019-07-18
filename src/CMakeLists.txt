cmake_minimum_required(VERSION 2.8)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(SRC_FILE
    cf_ssd_detector.cpp
    inf_face_detection_ssd_api.cpp
	)

include_directories(
    ${PROJECT_SOURCE_DIR}/caffe/include
    ${OpneCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
    ${PROJECT_SOURCE_DIR}/caffe
)

set(LINKER_EXT_LIBS
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    libcaffe.so
    libglog.so
)

cuda_add_library(cf_ssd SHARED ${SRC_FILE})
target_link_libraries(cf_ssd ${LINKER_EXT_LIBS})
