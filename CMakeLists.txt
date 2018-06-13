cmake_minimum_required(VERSION 3.10)
project(csrt)
enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Judge architecture
if (NOT ARCHITECTURE)
    message(WARNING "Architure not defined, set to system default.")
    set(ARCHITECTURE "Default")
endif ()
if (ARCHITECTURE MATCHES "Android-*")
    message(STATUS "Detect Android platform ${ARCHITECTURE}.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")
    add_definitions(-D__ANDROID__)
elseif(WIN32)
    add_definitions(-DCSRT_EXPORT)
endif ()

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

# Add sanitizers and threads
find_package(Sanitizers REQUIRED)
find_package(Threads REQUIRED)

# Configure head only third-party library
include_directories(
        ThirdParty/Eigen
        ThirdParty/Stb
        ThirdParty/Spdlog/include)
if (WIN32 AND ARCHITECTURE STREQUAL "Default")
    include_directories(ThirdParty/Dirent)
endif ()

# Configure compiled third-party lobrary
if (ARCHITECTURE STREQUAL "Android-arm")
    include_directories(ThirdParty/Fftw3/Android/include)
    link_directories(ThirdParty/Fftw3/Android/lib/arm)
    set(FFTW3_LIBS fftw3f fftw3f_threads)
elseif (ARCHITECTURE STREQUAL "Android-arm64")
    include_directories(ThirdParty/Fftw3/Android/include)
    link_directories(ThirdParty/Fftw3/Android/lib/arm64)
    set(FFTW3_LIBS fftw3f fftw3f_threads)
elseif (ARCHITECTURE STREQUAL "Android-x86")
    include_directories(ThirdParty/Fftw3/Android/include)
    link_directories(ThirdParty/Fftw3/Android/lib/x86)
    set(FFTW3_LIBS fftw3f fftw3f_threads)
elseif (ARCHITECTURE STREQUAL "Android-x86_64")
    include_directories(ThirdParty/Fftw3/Android/include)
    link_directories(ThirdParty/Fftw3/Android/lib/x86_64)
    set(FFTW3_LIBS fftw3f fftw3f_threads)
elseif (ARCHITECTURE STREQUAL "Default")
    if (WIN32 AND MSVC)
        include_directories(ThirdParty/Fftw3/Windows/include)
        link_directories(ThirdParty/Fftw3/Windows/lib)
        set(FFTW3_LIBS libfftw3f-3)
    elseif (APPLE)
        include_directories(ThirdParty/Fftw3/macOS/include)
        link_directories(ThirdParty/Fftw3/macOS/lib)
        set(FFTW3_LIBS fftw3f fftw3f_threads)
    else ()
        message(FATAL_ERROR, "Unsupported platform.")
    endif ()
else ()
    message(FATAL_ERROR "Unknown architecture.")
endif ()

# Set the source files
set(CSRT_SOURCE
        Source/CSRT.cpp
        Source/Utility/FFT.cpp
        Source/Utility/Mat.cpp
        Source/Utility/Memory.cpp
        Source/Utility/Parallel.cpp
        Source/Core/DSST.cpp
        Source/Core/FeaturesExtractor.cpp
        Source/Core/Filter.cpp
        Source/Core/Segment.cpp
        Source/Core/Tracker.cpp
        Source/Core/FeatureColorName.cpp
        Source/Export/CSRTracker.cpp
        Source/Utility/Smoother.cpp)

set(CSRT_HEADERS
        Source/CSRT.h
        Source/Utility/FFT.h
        Source/Utility/Mat.h
        Source/Utility/Memory.h
        Source/Utility/Parallel.h
        Source/Utility/Geometry.h
        Source/Utility/StringPrint.h
        Source/Core/DSST.h
        Source/Core/FeaturesExtractor.h
        Source/Core/Filter.h
        Source/Core/Segment.h
        Source/Core/Tracker.h
        Source/Export/CSRTracker.h
        Source/Utility/Smoother.h)

include_directories(
        Source
        Source/Utility
        Source/Core
        Source/Export)

# Main library
add_library(csrt SHARED
        ${CSRT_SOURCE}
        ${CSRT_HEADERS})
add_sanitizers(csrt)
if (WIN32 AND MSVC)
    # Avoid a name clash when building on Visual Studio
    set_target_properties(csrt PROPERTIES OUTPUT_NAME libcsrt)
endif ()
target_link_libraries(csrt ${FFTW3_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Test executable
if (ARCHITECTURE STREQUAL "Default")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(csrt_exe Source/test.cpp)
    add_sanitizers(csrt_exe)
    set_target_properties(csrt_exe PROPERTIES OUTPUT_NAME csrt)
    target_link_libraries(csrt_exe csrt ${OpenCV_LIBS})

    add_executable(csrt_demo_exe Source/run.cpp)
    add_sanitizers(csrt_demo_exe)
    set_target_properties(csrt_demo_exe PROPERTIES OUTPUT_NAME csrt_demo)
    target_link_libraries(csrt_demo_exe csrt ${OpenCV_LIBS})
endif ()