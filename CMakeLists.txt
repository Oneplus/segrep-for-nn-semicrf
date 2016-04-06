project(lstm-cnn)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g")
endif()

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

enable_testing()

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnn)
set(WITH_EIGEN_BACKEND 1)

# look for Boost
if(DEFINED ENV{BOOST_ROOT})
      set(Boost_NO_SYSTEM_PATHS ON)
endif()

set(Boost_REALPATH ON)
if(WIN32)
    find_package(Boost COMPONENTS program_options serialization log_setup log thread system filesystem date_time chrono REQUIRED)
else()
    add_definitions (-DBOOST_LOG_DYN_LINK)
    find_package(Boost COMPONENTS program_options serialization log_setup log thread system REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIR})
if(WIN32)
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
else()
    set(LIBS ${LIBS} ${Boost_LIBRARIES} pthread)
endif()


# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)
set(source_directory ${PROJECT_SOURCE_DIR}/src)
add_subdirectory(cnn/cnn)
add_subdirectory(src)
