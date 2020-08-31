cmake_minimum_required(VERSION 3.16)

project(OpenCL_test C)

add_compile_options(${OPENCL_CFLAGS})

set(CMAKE_C_STANDARD 99)
set(MODE ENCRYPT)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)
find_package(TIFF REQUIRED)

#find_package(PkgConfig REQUIRED) //asta e pentru POCL??

include_directories(${OpenCL_INCLUDE_DIRS})

link_directories(${OpenCL_LIBRARY})
link_directories(${TIFF_LIBRARIES})


add_executable(OpenCL_test main.c)

target_include_directories (OpenCL_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries (OpenCL_test ${OpenCL_LIBRARY})
target_link_libraries(OpenCL_test m)
target_link_libraries(OpenCL_test ${TIFF_LIBRARIES})
target_link_libraries("OpenCL_test" ${POCLU_LINK_OPTIONS})
target_link_libraries(OpenCL_test Threads::Threads)
