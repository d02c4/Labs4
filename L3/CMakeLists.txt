cmake_minimum_required(VERSION 3.28)
project(untitled5)

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled5 main.cpp)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
target_link_libraries(untitled5 ${GSL_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")