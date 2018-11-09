//
// Provide native core methods for KinectV2. Only for Windows platform.
//

#ifdef _WIN32

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_KINECTV2_H
#define CSRT_KINECTV2_H

#include "Camera.h"
#include "Kinect.h"

namespace CSRT {

class KinectService2 : public CameraService {
public:
	KinectService2();
	~KinectService2();

	// Initialize Kinect sensor
	bool Initialize(bool color, bool depth) override;
	// Called at every tick
	void Tick() override;
	// Release resource and dispose camera data.
	void Release() override;
	// Color data in bgra format. Ensure the memory has been pre-allocated for colorWidth*colorHeight*4 uint8_t. 
	void SetColorData(uint8_t* color) override { colorBuffer = color; }
	// Set the normalized depth frame data in float format. Ensure the memory has been pre-allocated for depthWidth*depthHeight float.
	void SetDepthData(float* depth) override { depthBuffer = depth; }
	// Set the point cloud data, uint m. Ensure the memory has been pre-allocated for depthWidth*depthHeight CameraPoint3.
	void SetPointCloud(CameraPoint3* pc) override { pointCloud = pc; }
	// Get data dimensions
	int GetColorWidth() override { return colorWidth; }
	int GetColorHeight() override { return colorHeight; }
	int GetDepthWidth() override { return depthWidth; }
	int GetDepthHeight() override { return depthHeight; }
	// Get maximum depth value, uint m.
	float GetMaxDepth() override { return maxDepth / 1000.0f; }
	// Get minimum depth value, uint m.
	float GetMinDepth() override { return minDepth / 1000.0f; }
	// Coordinate mappers
	void MapDepth2Color(const CameraPoint2 *depthPoints, CameraPoint2 *colorPoints, int count) override;
	void MapColor2Depth(const CameraPoint2 *colorPoints, CameraPoint2 *depthPoints, int count) override;
	
private:
	// Kinect data processing
	void ProcessKinectColor(const uint8_t* buffer);
	void ProcessKinectDepth(const uint16_t* buffer);
	
	// Generate point cloud from current depth data, uint m.
	void GeneratePointCloud();

	// Kinect related members
	int depthWidth;
	int depthHeight;
	int colorWidth;
	int colorHeight;
	int maxDepth;
	int minDepth;
	ColorImageFormat imageFormat;

	IKinectSensor* kinectSensor;
	ICoordinateMapper* coordinateMapper;
	IColorFrameReader* colorFrameReader;
	IDepthFrameReader* depthFrameReader;

	std::vector<uint8_t> colorCacheBuffer;
	uint8_t *colorBuffer;
	std::vector<uint16_t> depthCacheBuffer;
	float *depthBuffer;
	CameraPoint3 *pointCloud;

	std::vector<CameraPoint2> color2DepthCache;
	std::vector<uint16_t> depth2ColorCache;

	bool enableColor;
	bool enableDepth;
};

}	// namespace CSRT

#endif	// CSRT_KINECTV2_H

#endif	// WIN32
