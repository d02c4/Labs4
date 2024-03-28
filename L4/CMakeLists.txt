cmake_minimum_required(VERSION 3.28)
project(lab4)

set(CMAKE_CXX_STANDARD 17)



find_package(MPI)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    add_executable(lab4 main.cpp)
    target_link_libraries(lab4 ${MPI_LIBRARIES})
endif()