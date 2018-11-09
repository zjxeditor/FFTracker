//
// Provide native core methods for KinectV2. Only for Windows platform.
//

#ifdef _WIN32

#include "KinectService2.h"

namespace CSRT {

KinectService2::KinectService2() :
	depthWidth(0),
	depthHeight(0),
	colorWidth(0),
	colorHeight(0),
	maxDepth(0),
	minDepth(0),
	imageFormat(ColorImageFormat_None),
	kinectSensor(nullptr),
	coordinateMapper(nullptr),
	colorFrameReader(nullptr),
	depthFrameReader(nullptr),
	colorBuffer(nullptr),
	depthBuffer(nullptr),
	pointCloud(nullptr),
	enableColor(false),
	enableDepth(false) {}

KinectService2::~KinectService2() {
	// Clean up kinect members
	SafeRelease(colorFrameReader);
	SafeRelease(depthFrameReader);
	SafeRelease(coordinateMapper);
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;

	// Close the Kinect Sensor
	if (kinectSensor) {
		kinectSensor->Close();
		SafeRelease(kinectSensor);
	}
}

void KinectService2::Release() {
	// Clean up kinect members
	SafeRelease(colorFrameReader);
	SafeRelease(depthFrameReader);
	SafeRelease(coordinateMapper);
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;

	// Close the Kinect Sensor
	if (kinectSensor) {
		kinectSensor->Close();
		SafeRelease(kinectSensor);
	}
}

bool KinectService2::Initialize(bool color, bool depth) {
	enableColor = color;
	enableDepth = depth;
	if (!enableColor && !enableDepth) {
		throw std::invalid_argument("KinectService2::Initialize: read nothing from Kinect device.");
	}

	HRESULT hr;
	hr = GetDefaultKinectSensor(&kinectSensor);
	if (FAILED(hr) || kinectSensor == nullptr) {
		throw std::runtime_error("KinectService2::Initialize: cannot get any Kinect device.");
		return false;
	}

	// Open Kinect sensor and get the coordinate mapper
	hr = kinectSensor->Open();
	if (SUCCEEDED(hr))
		hr = kinectSensor->get_CoordinateMapper(&coordinateMapper);

	// Open the color frame reader
	if (SUCCEEDED(hr) && enableColor) {
		IColorFrameSource* colorFrameSource = nullptr;
		if (SUCCEEDED(hr))
			hr = kinectSensor->get_ColorFrameSource(&colorFrameSource);
		if (SUCCEEDED(hr))
			hr = colorFrameSource->OpenReader(&colorFrameReader);
		SafeRelease(colorFrameSource);

		// Get color frame attributes
		IColorFrame* colorFrame = nullptr;
		if (SUCCEEDED(hr))
			while(FAILED(hr = colorFrameReader->AcquireLatestFrame(&colorFrame))) {}
		if (SUCCEEDED(hr)) {
			IFrameDescription* frameDescription = nullptr;
			if (SUCCEEDED(hr))
				hr = colorFrame->get_FrameDescription(&frameDescription);
			if (SUCCEEDED(hr))
				hr = frameDescription->get_Width(&colorWidth);
			if (SUCCEEDED(hr))
				hr = frameDescription->get_Height(&colorHeight);
			if (SUCCEEDED(hr))
				hr = colorFrame->get_RawColorImageFormat(&imageFormat);
			SafeRelease(frameDescription);
		}
		SafeRelease(colorFrame);
		colorCacheBuffer.resize(colorWidth*colorHeight * 4);
	}

	// Open the depth frame reader
	if (SUCCEEDED(hr) && enableDepth) {
		IDepthFrameSource* depthFrameSource = nullptr;
		if (SUCCEEDED(hr))
			hr = kinectSensor->get_DepthFrameSource(&depthFrameSource);
		if (SUCCEEDED(hr))
			hr = depthFrameSource->OpenReader(&depthFrameReader);
		SafeRelease(depthFrameSource);

		// Get depth frame attributes
		IDepthFrame* depthFrame = nullptr;
		uint16_t tempMinDepth = 0;
		uint16_t tempMaxDepth = 0;

		if (SUCCEEDED(hr))
			while(FAILED(hr = depthFrameReader->AcquireLatestFrame(&depthFrame))) {}
		if (SUCCEEDED(hr)) {
			IFrameDescription* frameDescription = nullptr;
			if (SUCCEEDED(hr))
				hr = depthFrame->get_FrameDescription(&frameDescription);
			if (SUCCEEDED(hr))
				hr = frameDescription->get_Width(&depthWidth);
			if (SUCCEEDED(hr))
				hr = frameDescription->get_Height(&depthHeight);
			if (SUCCEEDED(hr))
				hr = depthFrame->get_DepthMinReliableDistance(&tempMinDepth);
			if (SUCCEEDED(hr))
				hr = depthFrame->get_DepthMaxReliableDistance(&tempMaxDepth);
			SafeRelease(frameDescription);
		}
		SafeRelease(depthFrame);
		minDepth = tempMinDepth;
		maxDepth = tempMaxDepth;
		depthCacheBuffer.resize(depthWidth*depthHeight);
	}

	if (enableColor && enableDepth) {
		color2DepthCache.resize(colorWidth*colorHeight);
		depth2ColorCache.resize(depthWidth*depthHeight);
	}

	if (FAILED(hr)) {
		throw std::runtime_error("KinectService2::Initialize: Kinect initialization process failed.");
		return false;
	}
	return true;
}

void KinectService2::Tick() {
	// Fetch Kinect color frame data
	if (colorFrameReader && colorBuffer) {
		IColorFrame* colorFrame = nullptr;
		HRESULT hr = colorFrameReader->AcquireLatestFrame(&colorFrame);
		if (SUCCEEDED(hr)) {
			uint8_t* buffer = nullptr;
			uint32_t bufferSize = 0;
			if (imageFormat == ColorImageFormat_Bgra)
				hr = colorFrame->AccessRawUnderlyingBuffer(&bufferSize, &buffer);
			else if (colorCacheBuffer.size() != 0) {
				buffer = &colorCacheBuffer[0];
				bufferSize = colorWidth * colorHeight * sizeof(uint8_t) * 4;
				hr = colorFrame->CopyConvertedFrameDataToArray(bufferSize, buffer, ColorImageFormat_Bgra);
			} else
				hr = E_FAIL;
			if (SUCCEEDED(hr)) ProcessKinectColor(buffer);
		}
		SafeRelease(colorFrame);
	}

	// Fetch Kinect depth frame data
	if (depthFrameReader && depthBuffer) {
		IDepthFrame* depthFrame = nullptr;
		HRESULT hr = depthFrameReader->AcquireLatestFrame(&depthFrame);
		if (SUCCEEDED(hr)) {
			UINT bufferSize = 0;
			uint16_t* buffer = nullptr;
			if (SUCCEEDED(hr))
				hr = depthFrame->AccessUnderlyingBuffer(&bufferSize, &buffer);
			if (SUCCEEDED(hr))
				ProcessKinectDepth(buffer);
		}
		SafeRelease(depthFrame);
	}
}

void KinectService2::ProcessKinectColor(const uint8_t* buffer) {
	// Make sure we've received valid data
	if (!buffer) return;
	memcpy(&colorBuffer[0], buffer, colorWidth * colorHeight * 4);
}

void KinectService2::ProcessKinectDepth(const uint16_t* buffer) {
	// Make sure we've received valid data
	if (!buffer) return;

	//memcpy(&depthCacheBuffer[0], buffer, depthWidth*depthHeight * sizeof(uint16_t));

	uint16_t *dst0 = &depthCacheBuffer[0];
	float *dst1 = &depthBuffer[0];
	int len = depthWidth * depthHeight;
	uint16_t temp;
	float scale = 1.0f / (maxDepth - minDepth);
	float offset = -minDepth * scale;
	for (int i = 0; i < len; ++i) {
		temp = *(buffer++);
		*dst0 = CameraClamp(temp, minDepth, maxDepth);
		*dst1 = (*dst0) * scale + offset;
		++dst0;
		++dst1;
	}

	if (pointCloud) GeneratePointCloud();
}

void KinectService2::MapColor2Depth(const CameraPoint2* colorPoints, CameraPoint2* depthPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("KinectService2::MapColor2Depth: cannot do mapping for the lack of data.");
		return;
	}

	HRESULT hr = coordinateMapper->MapColorFrameToDepthSpace(
		depthCacheBuffer.size(),
		&depthCacheBuffer[0],
		color2DepthCache.size(),
		reinterpret_cast<DepthSpacePoint*>(&color2DepthCache[0]));
	if (FAILED(hr)) {
		throw std::runtime_error("KinectService2::MapColor2Depth: failed to map color points to depth points.");
		return;
	}
	for (int i = 0; i < count; ++i) {
		auto &cp = colorPoints[i];
		auto &dp = depthPoints[i];
		dp = color2DepthCache[(int)(cp.Y*colorWidth + cp.X)];
		if (std::isinf(dp.X) || std::isinf(dp.Y)) {
			dp.X = 0.0f;
			dp.Y = 0.0f;
		}
	}
}

void KinectService2::MapDepth2Color(const CameraPoint2* depthPoints, CameraPoint2* colorPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("KinectService2::MapDepth2Color: cannot do mapping for the lack of data.");
	}

	for (int i = 0; i < count; ++i) {
		auto &dp = depthPoints[i];
		depth2ColorCache[i] = depthCacheBuffer[(int)(dp.Y*depthWidth + dp.X)];
	}
	HRESULT hr = coordinateMapper->MapDepthPointsToColorSpace(
		count,
		reinterpret_cast<const DepthSpacePoint*>(depthPoints),
		count,
		&depth2ColorCache[0],
		count,
		reinterpret_cast<ColorSpacePoint*>(colorPoints));
	if (FAILED(hr)) {
		throw std::runtime_error("KinectService2::MapDepth2Color: failed to map depth points to color points.");
		return;
	}
	for (int i = 0; i < count; ++i) {
		auto& cp = colorPoints[i];
		if (std::isinf(cp.X) || std::isinf(cp.Y)) {
			cp.X = 0.0f;
			cp.Y = 0.0f;
		}
	}
}

void KinectService2::GeneratePointCloud() {
	if(!enableDepth) {
		throw std::runtime_error("KinectService2::GeneratePointCloud: cannot do mapping for the lack of depth data.");
		return;
	}

	HRESULT hr = coordinateMapper->MapDepthFrameToCameraSpace(depthCacheBuffer.size(),
		&depthCacheBuffer[0],
		depthCacheBuffer.size(),
		reinterpret_cast<CameraSpacePoint*>(&pointCloud[0]));

	if(FAILED(hr))
		throw std::runtime_error("KinectService2::GeneratePointCloud: failed to generate point cloud.");
}

}	// namespace CSRT

#endif	// WIN32