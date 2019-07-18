cmake_minimum_required(VERSION 2.8)
project(CAFFE_SSD)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} not suppoet C++11.")
endif()

find_package(OpenCV 3.4.1)
find_package(CUDA)
find_package(Boost REQUIRED COMPONENTS system filesystem)

add_subdirectory(src)
add_subdirectory(unit_test)