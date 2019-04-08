//
// Provide native core methods for KinectV1. Only for Windows platform.
//

#ifdef _WIN32

#include "KinectService1.h"
#include <stdexcept>
#include <cmath>

namespace CSRT {

KinectService1::KinectService1() :
	depthWidth(0),
	depthHeight(0),
	colorWidth(0),
	colorHeight(0),
	maxDepth(0),
	minDepth(0),
	nuiSensor(nullptr),
	nuiMapper(nullptr),
	colorStreamHandle(INVALID_HANDLE_VALUE),
	colorNextFrameEvent(INVALID_HANDLE_VALUE),
	depthStreamHandle(INVALID_HANDLE_VALUE),
	depthNextFrameEvent(INVALID_HANDLE_VALUE),
	colorBuffer(nullptr),
	depthBuffer(nullptr),
	pointCloud(nullptr),
	enableColor(false),
	enableDepth(false) {}

KinectService1::~KinectService1() {
	if (nuiSensor) nuiSensor->NuiShutdown();
	if (colorNextFrameEvent != INVALID_HANDLE_VALUE) CloseHandle(colorNextFrameEvent);
	if (depthNextFrameEvent != INVALID_HANDLE_VALUE) CloseHandle(depthNextFrameEvent);
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;
	SafeRelease(nuiMapper);
	SafeRelease(nuiSensor);
}

void KinectService1::Release() {
	if (nuiSensor) nuiSensor->NuiShutdown();
	if (colorNextFrameEvent != INVALID_HANDLE_VALUE) CloseHandle(colorNextFrameEvent);
	if (depthNextFrameEvent != INVALID_HANDLE_VALUE) CloseHandle(depthNextFrameEvent);
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;
	SafeRelease(nuiMapper);
	SafeRelease(nuiSensor);
}

bool KinectService1::Initialize(bool color, bool depth) {
	enableColor = color;
	enableDepth = depth;
	if (!enableColor && !enableDepth) {
		throw std::invalid_argument("KinectService1::Initialize: read nothing from Kinect device.");
	}

	nuiSensor = nullptr;
	INuiSensor *tempSensor = nullptr;
	HRESULT hr;

	int sensorCount = 0;
	hr = NuiGetSensorCount(&sensorCount);
	if (FAILED(hr)) {
		throw std::runtime_error("KinectService1::Initialize: cannot get any Kinect device.");
		return false;
	}

	// Look at each Kinect sensor
	for (int i = 0; i < sensorCount; ++i) {
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &tempSensor);
		if (FAILED(hr)) continue;
		// Get the status of the sensor, and if connected, then we can initialize it
		hr = tempSensor->NuiStatus();
		if (S_OK == hr) {
			nuiSensor = tempSensor;
			break;
		}
		// This sensor wasn't OK, so release it since we're not using it
		tempSensor->Release();
	}

	if (nuiSensor == nullptr) {
		throw std::runtime_error("KinectService1::Initialize: cannot get any valid Kinect device.");
		return false;
	}

	// Initialize the Kinect and specify that what we'll use
	if (enableColor && enableDepth) {
		hr = nuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH);
	} else if (enableColor) {
		hr = nuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR);
	} else if (enableDepth) {
		hr = nuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH);
	}

	if (enableColor) {
		// Create an event that will be signaled when color data is available
		colorNextFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
		// Open a color image stream to receive color frames
		if (SUCCEEDED(hr))
			hr = nuiSensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_COLOR,
				NUI_IMAGE_RESOLUTION_640x480,
				0,
				2,
				colorNextFrameEvent,
				&colorStreamHandle);

		colorWidth = 640;
		colorHeight = 480;
		color2DepthCache.resize(colorWidth*colorHeight);
	}
	if (enableDepth) {
		// Create an event that will be signaled when depth data is available
		depthNextFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
		// Open a depth image stream to receive depth frames
		if (SUCCEEDED(hr))
			hr = nuiSensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_DEPTH,
				NUI_IMAGE_RESOLUTION_320x240,
				NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE,
				2,
				depthNextFrameEvent,
				&depthStreamHandle);

		depthWidth = 320;
		depthHeight = 240;
		minDepth = NUI_IMAGE_DEPTH_MINIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
		maxDepth = NUI_IMAGE_DEPTH_MAXIMUM_NEAR_MODE >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
		depthCacheBuffer.resize(depthWidth*depthHeight * 2);
		pointCloudCache.resize(depthWidth*depthHeight);
		depth2ColorCache0.resize(depthWidth*depthHeight);
		depth2ColorCache1.resize(depthWidth*depthHeight);
	}

	if (SUCCEEDED(hr))
		hr = nuiSensor->NuiGetCoordinateMapper(&nuiMapper);
	if (FAILED(hr)) {
		throw std::runtime_error("KinectService1::Initialize: Kinect initialization process failed.");
		return false;
	}
	return true;
}

void KinectService1::Tick() {
	// Fetch Kinect color frame data
	if (enableColor && colorBuffer && WaitForSingleObject(colorNextFrameEvent, 0) == WAIT_OBJECT_0) {
		NUI_IMAGE_FRAME colorFrame;
		HRESULT hr = nuiSensor->NuiImageStreamGetNextFrame(colorStreamHandle, 0, &colorFrame);

		if (SUCCEEDED(hr)) {
			INuiFrameTexture *texture = colorFrame.pFrameTexture;
			NUI_LOCKED_RECT LockedRect;
			// Lock the frame data so the Kinect knows not to modify it while we're reading it
			texture->LockRect(0, &LockedRect, nullptr, 0);
			// Make sure we've received valid data
			if (LockedRect.Pitch != 0 && LockedRect.pBits != nullptr && LockedRect.size == colorWidth * colorHeight * 4)
                ProcessKinectColor(LockedRect.pBits);
			// We're done with the texture so unlock it
			texture->UnlockRect(0);
			// Release the frame
			nuiSensor->NuiImageStreamReleaseFrame(colorStreamHandle, &colorFrame);
		}
	}
	// Fetch Kinect depth frame data
	if (enableDepth && depthBuffer && WaitForSingleObject(depthNextFrameEvent, 0) == WAIT_OBJECT_0) {
		NUI_IMAGE_FRAME depthFrame;
		HRESULT hr = nuiSensor->NuiImageStreamGetNextFrame(depthStreamHandle, 0, &depthFrame);

		if (SUCCEEDED(hr)) {
			BOOL nearMode;
			INuiFrameTexture* texture = nullptr;
			// Get the depth image pixel texture
			hr = nuiSensor->NuiImageFrameGetDepthImagePixelFrameTexture(
				depthStreamHandle, &depthFrame, &nearMode, &texture);
			if (SUCCEEDED(hr)) {
				NUI_LOCKED_RECT LockedRect;
				// Lock the frame data so the Kinect knows not to modify it while we're reading it
				texture->LockRect(0, &LockedRect, nullptr, 0);
				// Make sure we've received valid data
				if (LockedRect.Pitch != 0 && LockedRect.pBits != nullptr && LockedRect.size == depthWidth * depthHeight * 4)
                    ProcessKinectDepth(reinterpret_cast<uint16_t*>(LockedRect.pBits));
				// We're done with the texture so unlock it
				texture->UnlockRect(0);
				texture->Release();
			}
			// Release the frame
			nuiSensor->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);
		}
	}
}

void KinectService1::ProcessKinectColor(const uint8_t* buffer) {
	memcpy(&colorBuffer[0], buffer, colorWidth * colorHeight * 4);
}

void KinectService1::ProcessKinectDepth(const uint16_t* buffer) {
	// Data in buffer are stored as NUI_DEPTH_IMAGE_PIXEL format.
	int len = depthWidth * depthHeight;
	
	//memcpy(&depthCacheBuffer[0], buffer, len * 4);
	
	uint16_t *dst0 = &depthCacheBuffer[0];
	float* dst1 = &depthBuffer[0];
	uint16_t temp;
	float scale = 1.0f / (maxDepth - minDepth);
	float offset = -minDepth * scale;
	for (int i = 0; i < len; ++i) {
		*(dst0++) = *(buffer++);
		temp = *(buffer++);
		*dst0 = CameraClamp(temp, minDepth, maxDepth);
		*dst1 = (*dst0) * scale + offset;
		++dst0;
		++dst1;
	}

	if (pointCloud) GeneratePointCloud();
}

void KinectService1::MapColor2Depth(const CameraPoint2* colorPoints, CameraPoint2* depthPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("KinectService1::MapColor2Depth: cannot do mapping for the lack of data.");
	}

	HRESULT hr = nuiMapper->MapColorFrameToDepthFrame(
		NUI_IMAGE_TYPE_COLOR,
		NUI_IMAGE_RESOLUTION_640x480,
		NUI_IMAGE_RESOLUTION_320x240,
		depthWidth*depthHeight,
		reinterpret_cast<NUI_DEPTH_IMAGE_PIXEL*>(&depthCacheBuffer[0]),
		colorWidth*colorHeight,
		&color2DepthCache[0]);
	if(FAILED(hr)) {
		throw std::runtime_error("KinectService1::MapColor2Depth: failed to map color points to depth points.");
		return;
	}

	for (int i = 0; i < count; ++i) {
		auto &cp = colorPoints[i];
		auto &dp = depthPoints[i];
		auto index = (int)(cp.Y*colorWidth + cp.X);
		dp.X = (float)color2DepthCache[index].x;
		dp.Y = (float)color2DepthCache[index].y;
		if (std::isinf(dp.X) || std::isinf(dp.Y)) {
			dp.X = 0.0f;
			dp.Y = 0.0f;
		}
	}
}

void KinectService1::MapDepth2Color(const CameraPoint2* depthPoints, CameraPoint2* colorPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("KinectService1::MapDepth2Color: cannot do mapping for the lack of data.");
	}

	for(int i = 0; i < count; ++i) {
		auto &dp = depthPoints[i];
		depth2ColorCache0[i].x = (LONG)dp.X;
		depth2ColorCache0[i].y = (LONG)dp.Y;
		depth2ColorCache0[i].depth = (LONG)depthBuffer[(int)(dp.Y*depthWidth + dp.X)];
	}
	HRESULT hr = nuiMapper->MapDepthPointToColorPoint(
		NUI_IMAGE_RESOLUTION_320x240, 
		&depth2ColorCache0[0], 
		NUI_IMAGE_TYPE_COLOR, 
		NUI_IMAGE_RESOLUTION_640x480, 
		&depth2ColorCache1[0]);
	if (FAILED(hr)) {
		throw std::runtime_error("KinectService1::MapDepth2Color: failed to map depth points to color points.");
		return;
	}
	for(int i = 0; i < count; ++i) {
		auto &cp = colorPoints[i];
		cp.X = (float)depth2ColorCache1[i].x;
		cp.Y = (float)depth2ColorCache1[i].y;
		if (std::isinf(cp.X) || std::isinf(cp.Y)) {
			cp.X = 0.0f;
			cp.Y = 0.0f;
		}
	}
}

void KinectService1::GeneratePointCloud() {
	if (!enableDepth) {
		throw std::runtime_error("KinectService1::GeneratePointCloud: cannot do mapping for the lack of depth data.");
		return;
	}

	HRESULT hr = nuiMapper->MapDepthFrameToSkeletonFrame(
		NUI_IMAGE_RESOLUTION_320x240,
		depthWidth*depthHeight,
		reinterpret_cast<NUI_DEPTH_IMAGE_PIXEL*>(&depthCacheBuffer[0]),
		depthWidth*depthHeight,
		&pointCloudCache[0]);

	if (FAILED(hr))
		throw std::runtime_error("KinectService1::GeneratePointCloud: failed to generate point cloud.");

	float *pd = reinterpret_cast<float*>(&pointCloud[0]);
	float *ps = reinterpret_cast<float*>(&pointCloudCache[0]);

	int len = depthWidth * depthHeight;
	for (int i = 0; i < len; ++i) {
		pd[0] = ps[0] / ps[3];
		pd[1] = ps[1] / ps[3];
		pd[2] = ps[2] / ps[3];
		pd += 3;
		ps += 4;
	}
}

}	// namespace CSRT

#endif	// WIN32