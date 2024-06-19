cmake_minimum_required(VERSION 3.10)
project(gpu_programming
    VERSION 0.1
    DESCRIPTION "GPU Programming"
    LANGUAGES CXX CUDA
)

if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CUDA_ARCHITECTURES "86;89")

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

include_directories(src)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu" "src/*.cuh")
file(GLOB TEST_SOURCES "tests/*.cu")

enable_testing()

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC} ${CUDA_SOURCES})
    target_link_libraries(${TEST_NAME} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CUDA_LIBRARIES} pthread)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
