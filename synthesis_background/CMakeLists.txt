cmake_minimum_required(VERSION 3.5)

project(main LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED HINTS "$ENV{HOME}/lib/opencv3_4_11/share/OpenCV")

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main Threads::Threads)