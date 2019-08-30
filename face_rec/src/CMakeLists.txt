cmake_minimum_required(VERSION 2.8)

project(face_rec CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

#FIND_PACKAGE(Boost COMPONENTS python)
#FIND_PACKAGE(PythonInterp)
#FIND_PACKAGE(PythonLibs)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
#include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include ${CMAKE_CURRENT_LIST_DIR})
#LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/ncnn)


#list(APPEND LINKER_LIBS opencv_videoio opencv_video opencv_imgcodecs opencv_core opencv_highgui opencv_imgproc ncnn)
list(APPEND LINKER_LIBS opencv_videoio opencv_video opencv_imgcodecs opencv_core opencv_imgproc 
	ncnn z jpeg v4l1 v4l2 v4lconvert 
	avcodec avdevice avfilter avformat avutil avresample swresample swscale bz2)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/*.h
              ${CMAKE_CURRENT_LIST_DIR}/featuredb.cpp
	      ${CMAKE_CURRENT_LIST_DIR}/mobilefacenet.cpp
	      ${CMAKE_CURRENT_LIST_DIR}/mtcnn.cpp)

# main app
file(GLOB BINSRC ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
add_executable(facedect ${SRC} ${BINSRC})
set_target_properties(facedect PROPERTIES PREFIX "")
target_link_libraries(facedect ${LINKER_LIBS} pthread)
#genfeature app
file(GLOB GENFSRC ${CMAKE_CURRENT_LIST_DIR}/genfeature.cpp)
add_executable(genfeature ${SRC} ${GENFSRC})
target_link_libraries(genfeature ${LINKER_LIBS})
install(TARGETS facedect genfeature  RUNTIME DESTINATION bin)
