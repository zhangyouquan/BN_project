cmake_minimum_required( VERSION 3.1 )
project(3D-vision)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

if(NOT OpenGL_FOUND)
    message(FATAL_ERROR "OpenGL not found.")
endif()

include_directories(${LIBREALSENSE_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
        "/usr/include/GLFW"
         )

set(THIRD_PARTY_LIBS
        /usr/local/lib/librealsense2.so
        ${OpenCV_LIBS} 
		${OPENGL_LIBRARIES}
		-lglfw
		libdarknet.so
		libpthread.so.0
        )

include_directories(${PROJECT_SOURCE_DIR}/headfile)
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)

install(FILES COPYRIGHT README DESTINATION share/doc/BN_projects)
install(DIRECTORY doc/ DESTINATION share/doc/BN_projects)
