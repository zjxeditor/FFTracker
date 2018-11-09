//
// Provide native core methods for RealSense device.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_REALSENSE_H
#define CSRT_REALSENSE_H

#include "Camera.h"
#include "librealsense2/rs.hpp"

namespace CSRT {

class RealSense : public CameraService {
public:
	RealSense(bool dpPostProcessing, int dimReduce);

	~RealSense();

	// Initialize RealSense sensor
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
	float GetMaxDepth() override { return maxDepth; }
	// Get minimum depth value, uint m.
	float GetMinDepth() override { return minDepth; }
	// Coordinate mappers
	void MapDepth2Color(const CameraPoint2* depthPoints, CameraPoint2* colorPoints, int count) override;
	void MapColor2Depth(const CameraPoint2* colorPoints, CameraPoint2* depthPoints, int count) override;

private:
	// Kinect data processing
	void ProcessColor(const uint8_t* buffer);
	void ProcessDepth(uint16_t* buffer);

	// Private members.
	int depthWidth;
	int depthHeight;
	int colorWidth;
	int colorHeight;
	int fps;
	rs2_format color_format;
	rs2_format depth_format;
	float depthScale;	// The distance in units of meters.
	float maxDepth;		// Maximum depth in meters.	
	float minDepth;		// Minimum depth in meters.

	rs2::pipeline pipe;
	rs2::pointcloud pc;
	rs2_extrinsics depth2Color;
	rs2_extrinsics color2Depth;
	rs2_intrinsics depthIntrinsics;
	rs2_intrinsics colorIntrinsics;

	// Whether to do post processing for depth frame.
	bool postProcessing;

	// Decimation filter reduces the amount of data while preserving best samples.
	rs2::decimation_filter dec_filter;
	int dec_filter_step;	// 2 is a good option.
	// Disparity conversion filters.
	rs2::disparity_transform depth2disparity_filter;
	rs2::disparity_transform disparity2depth_filter;
	// Define spatial filter edge-preserving.
	rs2::spatial_filter spat_filter;
	// Define temporal filter
	rs2::temporal_filter temp_filter;

	// Buffers
	uint8_t *colorBuffer;
	float *depthBuffer;
	std::vector<uint16_t> depthCacheBuffer;
	CameraPoint3 *pointCloud;

	bool enableColor;
	bool enableDepth;
};


}	// namespace CSRT

#endif	// CSRT_REALSENSE_H