cmake_minimum_required(VERSION 3.10)
project(csrt)
enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add sanitizers
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Build type check
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()
if (CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif ()

# Third-party library check
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Eigen")
    message(FATAL_ERROR "The Eigen library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Stb")
    message(FATAL_ERROR "The Stb library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Spdlog")
    message(FATAL_ERROR "The Spdlog library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Fftw3")
    message(FATAL_ERROR "The Fftw3 library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Dirent")
    message(FATAL_ERROR "The Dirent library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/KinectV1")
    message(FATAL_ERROR "The KinectV1 library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/KinectV2")
    message(FATAL_ERROR "The KinectV2 library is missing!")
endif ()
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/RealSense")
    message(FATAL_ERROR "The RealSense library is missing!")
endif ()

# Add sanitizers and threads
find_package(Sanitizers REQUIRED)
find_package(Threads REQUIRED)

# Configure head only third-party library
include_directories(
        ThirdParty/Eigen
        ThirdParty/Stb
        ThirdParty/Spdlog/include)
if (WIN32)
    include_directories(ThirdParty/Dirent)
endif ()

# Configure compiled third-party lobrary
if (WIN32 AND MSVC)
    # FFTW
    include_directories(ThirdParty/Fftw3/Windows/include)
    link_directories(ThirdParty/Fftw3/Windows/lib)
    set(FFTW3_LIBS libfftw3f-3)
    # KinectV1
    include_directories(ThirdParty/KinectV1/inc)
    link_directories(ThirdParty/KinectV1/lib/amd64/)
    set(KINECTV1_LIBS Kinect10)
    # KinectV2
    include_directories(ThirdParty/KinectV2/inc)
    link_directories(ThirdParty/KinectV2/lib/x64)
    set(KINECTV2_LIBS Kinect20)
    # RealSense
    include_directories(ThirdParty/RealSense/Windows/include)
    if (CMAKE_BUILD_TYPE MATCHES RELEASE)
        link_directories(ThirdParty/RealSense/Windows/lib/release)
    else ()
        link_directories(ThirdParty/RealSense/Windows/lib/debug)
    endif ()
    set(REALSENSE_LIBS realsense2)
elseif (APPLE)
    # FFTW
    include_directories(ThirdParty/Fftw3/macOS/include)
    link_directories(ThirdParty/Fftw3/macOS/lib)
    set(FFTW3_LIBS fftw3f fftw3f_threads)
    # RealSense
    include_directories(ThirdParty/RealSense/macOS/include)
    if (CMAKE_BUILD_TYPE MATCHES RELEASE)
        link_directories(ThirdParty/RealSense/macOS/lib/release)
    else ()
        link_directories(ThirdParty/RealSense/macOS/lib/debug)
    endif ()
    set(REALSENSE_LIBS realsense2.2)
else ()
    message(FATAL_ERROR, "Unsupported platform.")
endif ()

# Set the source files
set(CSRT_CAMERA_SOURCE
        Source/Camera/KinectService1.cpp
        Source/Camera/KinectService2.cpp
        Source/Camera/RealSense.cpp Source/Camera/Camera.cpp)

SET(CSRT_CAMERA_HEADERS
        Source/Camera/Camera.h
        Source/Camera/KinectService1.h
        Source/Camera/KinectService2.h
        Source/Camera/RealSense.h)

set(CSRT_SOURCE
        Source/CSRT.cpp
        Source/Utility/FFT.cpp
        Source/Utility/Mat.cpp
        Source/Utility/Memory.cpp
        Source/Utility/Parallel.cpp
        Source/Utility/Smoother.cpp
        Source/Core/DSST.cpp
        Source/Core/FDSST.cpp
        Source/Core/FeaturesExtractor.cpp
        Source/Core/Filter.cpp
        Source/Core/InfoProvider.cpp
        Source/Core/Processor.cpp
        Source/Core/Segment.cpp
        Source/Core/Tracker.cpp
        Source/Core/FeatureColorName.cpp
        Source/Export/CSRTracker.cpp)

set(CSRT_HEADERS
        Source/CSRT.h
        Source/Utility/FFT.h
        Source/Utility/Mat.h
        Source/Utility/Memory.h
        Source/Utility/Parallel.h
        Source/Utility/Smoother.h
        Source/Utility/Geometry.h
        Source/Utility/StringPrint.h
        Source/Core/DSST.h
        Source/Core/FDSST.h
        Source/Core/FeaturesExtractor.h
        Source/Core/Filter.h
        Source/Core/InfoProvider.h
        Source/Core/Processor.h
        Source/Core/Segment.h
        Source/Core/Tracker.h
        Source/Export/CSRTracker.h)

include_directories(
        Source
        Source/Camera
        Source/Utility
        Source/Core
        Source/Export
        Resampler
        Resampler/Base
        Resampler/Kernels
        Resampler/Operators
        Resampler/Utilities)

# Camera support library
add_library(csrt_camera SHARED
        ${CSRT_CAMERA_SOURCE}
        ${CSRT_CAMERA_HEADERS})
add_sanitizers(csrt_camera)
target_compile_definitions(csrt_camera PUBLIC CSRT_EXPORT)
if (WIN32 AND MSVC)
    target_link_libraries(csrt_camera ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS})
else ()
    target_link_libraries(csrt_camera ${REALSENSE_LIBS})
endif ()

# Main library
add_library(csrt SHARED
        ${CSRT_SOURCE}
        ${CSRT_HEADERS})
add_sanitizers(csrt)
target_compile_definitions(csrt PUBLIC CSRT_EXPORT)
target_link_libraries(csrt ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Test executable
find_package(OpenCV REQUIRED) # OpenCV is required for test cases.
include_directories(${OpenCV_INCLUDE_DIRS})

if(APPLE)
    # Compile for macOS
    add_executable(test_camera Run/testCamera.cpp)
    add_sanitizers(test_camera)
    target_link_libraries(test_camera csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_eval_direct Run/testEvalDirect.cpp)
    add_sanitizers(test_eval_direct)
    target_link_libraries(test_eval_direct csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_features Run/testFeatures.cpp)
    add_sanitizers(test_features)
    target_link_libraries(test_features csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_run_direct Run/testRunDirect.cpp)
    add_sanitizers(test_run_direct)
    target_link_libraries(test_run_direct csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_run_direct_depth Run/testRunDirectDepth.cpp)
    add_sanitizers(test_run_direct_depth)
    target_link_libraries(test_run_direct_depth csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_run_video Run/testRunVideo.cpp)
    add_sanitizers(test_run_video)
    target_link_libraries(test_run_video csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_run_video_depth Run/testRunVideoDepth.cpp)
    add_sanitizers(test_run_video_depth)
    target_link_libraries(test_run_video_depth csrt csrt_camera ${OpenCV_LIBS})

    add_executable(test_image_resizer Run/testImageResizer.cpp)
    add_sanitizers(test_image_resizer)
    target_link_libraries(test_image_resizer csrt ${OpenCV_LIBS})
elseif(WIN32 AND MSVC)
    # Windows platform and MSVC compiler
    add_executable(test_camera Run/testCamera.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_camera)
    target_link_libraries(test_camera ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_eval_direct Run/testEvalDirect.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_eval_direct)
    target_link_libraries(test_eval_direct ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_features Run/testFeatures.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_features)
    target_link_libraries(test_features ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_run_direct Run/testRunDirect.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_run_direct)
    target_link_libraries(test_run_direct ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_run_direct_depth Run/testRunDirectDepth.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_run_direct_depth)
    target_link_libraries(test_run_direct_depth ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_run_video Run/testRunVideo.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_run_video)
    target_link_libraries(test_run_video ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_run_video_depth Run/testRunVideoDepth.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_run_video_depth)
    target_link_libraries(test_run_video_depth ${OpenCV_LIBS} ${KINECTV1_LIBS} ${KINECTV2_LIBS} ${REALSENSE_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

    add_executable(test_image_resizer Run/testImageResizer.cpp ${CSRT_CAMERA_SOURCE} ${CSRT_CAMERA_HEADERS} ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(test_image_resizer)
    target_link_libraries(test_image_resizer ${OpenCV_LIBS} ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})
else()
    message(FATAL_ERROR, "Unsupported platform.")
endif()

# vot integration
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Trax")
    message(FATAL_ERROR "The Trax library is missing!")
endif ()

if (WIN32 AND MSVC)
    # Trax
    include_directories(ThirdParty/Trax/Windows/include)
    if (CMAKE_BUILD_TYPE MATCHES RELEASE)
        link_directories(ThirdParty/Trax/Windows/lib/release)
    else ()
        link_directories(ThirdParty/Trax/Windows/lib/debug)
    endif ()
	set(TRAX_LIBS traxd)
elseif (APPLE)
    # Trax
    include_directories(ThirdParty/Trax/macOS/include)
    if (CMAKE_BUILD_TYPE MATCHES RELEASE)
        link_directories(ThirdParty/Trax/macOS/lib/release)
    else ()
        link_directories(ThirdParty/Trax/macOS/lib/debug)
    endif ()
    set(TRAX_LIBS trax)
else ()
    message(FATAL_ERROR, "Unsupported platform.")
endif ()

if (WIN32 AND MSVC)
    add_executable(vot_interface Run/votInterface.cpp Run/vot.h ${CSRT_SOURCE} ${CSRT_HEADERS})
    add_sanitizers(vot_interface)
    target_link_libraries(vot_interface ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${TRAX_LIBS})
elseif (APPLE)
    add_executable(vot_interface Run/votInterface.cpp Run/vot.h)
    add_sanitizers(vot_interface)
    target_link_libraries(vot_interface csrt ${TRAX_LIBS})
else ()
    message(FATAL_ERROR, "Unsupported platform.")
endif ()

# OpenCV original csrt tracker
set(OPENCV_CSRT_SOURCE
        cvcsrt/featureColorName.cpp
        cvcsrt/trackerCSRT.cpp
        cvcsrt/trackerCSRTScaleEstimation.cpp
        cvcsrt/trackerCSRTSegmentation.cpp
        cvcsrt/trackerCSRTUtils.cpp)

set(OPENCV_CSRT_HEADERS
        cvcsrt/precomp.hpp
        cvcsrt/tracker.hpp
        cvcsrt/trackerCSRTScaleEstimation.hpp
        cvcsrt/trackerCSRTSegmentation.hpp
        cvcsrt/trackerCSRTUtils.hpp)

add_library(cvcsrt SHARED
        ${OPENCV_CSRT_SOURCE}
        ${OPENCV_CSRT_HEADERS})
add_sanitizers(cvcsrt)
target_link_libraries(cvcsrt ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(vot_interface_cvcsrt Run/votInterface_cvcsrt.cpp Run/vot.h)
add_sanitizers(vot_interface_cvcsrt)
target_link_libraries(vot_interface_cvcsrt cvcsrt ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT}  ${TRAX_LIBS})