cmake_minimum_required(VERSION 2.8)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(UNIT_FILE
    main.cpp
    cf_ssd_util.cpp
    )

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/caffe/include
)

link_directories(
    ${Boost_LIBRARY_DIRS}
    ${PROJECT_SOURCE_DIR}/lib
    ${PROJECT_SOURCE_DIR}/caffe
    )

set(LINKER_EXT_LIBS
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    libcaffe.so
    libcf_ssd.so
)

add_executable(target ${UNIT_FILE})
target_link_libraries(target ${LINKER_EXT_LIBS})
