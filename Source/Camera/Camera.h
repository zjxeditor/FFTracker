//
// Provide common utilities for camera operations.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_CAMERA_H
#define CSRT_CAMERA_H

#ifdef _WIN32
#ifdef CSRT_EXPORT
#define CSRT_API __declspec(dllexport)
#else
#define CSRT_API __declspec(dllimport)
#endif
#else
#define CSRT_API
#endif

#include <vector>
#include <cstdint>
#include <memory>

namespace CSRT {

// Represent point on 2D image.
struct CameraPoint2 {
	CameraPoint2() : X(0.0f), Y(0.0f) {}
	CameraPoint2(float x, float y) : X(x), Y(y) {}
	float X, Y;	// Only assign positive integers.
};

// Represent point in 3D space.
struct CameraPoint3 {
	CameraPoint3() : X(0.0f), Y(0.0f), Z(0.0f) {}
	CameraPoint3(float x, float y, float z) : X(x), Y(y), Z(z) {}
	float X, Y, Z;
};

template <typename T, typename U, typename V>
inline T CameraClamp(T val, U low, V high) {
	if (val < low)
		return low;
	else if (val > high)
		return high;
	else
		return val;
}

// Common interface for any camera service.
class CameraService {
public:
	// Destructor.
	virtual ~CameraService() = default;
	// Do initialization work.
	virtual bool Initialize(bool color, bool depth) = 0;
	// Tick in every frame to fetch new image data.
	virtual void Tick() = 0;
	// Release resource and dispose camera data.
	virtual void Release() = 0;
	// Color data in bgra format. Ensure the memory has been pre-allocated for colorWidth*colorHeight*4 uint8_t. 
	virtual void SetColorData(uint8_t* color) = 0;
	// Set the normalized depth frame data in float format. Ensure the memory has been pre-allocated for depthWidth*depthHeight float.
	virtual void SetDepthData(float* depth) = 0;
	// Set the point cloud data, uint m. Ensure the memory has been pre-allocated for depthWidth*depthHeight CameraPoint3.
	virtual void SetPointCloud(CameraPoint3* pc) = 0;
	// Get the frame dimensions.
	virtual int GetColorWidth() = 0;
	virtual int GetColorHeight() = 0;
	virtual int GetDepthWidth() = 0;
	virtual int GetDepthHeight() = 0;
	// Get maximum reliable depth value, uint m.
	virtual float GetMaxDepth() = 0;
	// Get minimum reliable depth value, uint m.
	virtual float GetMinDepth() = 0;
	// Map from depth frame to color frame.
	virtual void MapDepth2Color(const CameraPoint2 *depthPoints, CameraPoint2 *colorPoints, int count) = 0;
	// Map from color frame to depth frame.
	virtual void MapColor2Depth(const CameraPoint2 *colorPoints, CameraPoint2 *depthPoints, int count) = 0;
};

#ifdef _WIN32
enum class CameraType {
    RealSense = 0,
    KinectV1 = 1,
    KinectV2 = 2,
};
#else
enum class CameraType {
    RealSense = 0
};
#endif

// Create a camera service of specific type. 'dpPostProcessing' and 'dimReduce' only used for RealSense device.
// If 'dpPostProcessing' is set to 'true', the depth image will pass a post processing stage to remove noise.
// The final depth image size will be reduces by 'dimReduce' times.
CSRT_API std::unique_ptr<CameraService> CreateCameraService(CameraType type, bool dpPostProcessing = true, int dimReduce = 2);

#ifdef _WIN32
// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease) {
	if (pInterfaceToRelease != nullptr) {
		pInterfaceToRelease->Release();
		pInterfaceToRelease = nullptr;
	}
}
#endif	// WIN32

}	// namespace CSRT

#endif	// CSRT_CAMERA_H
