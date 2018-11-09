//
// Provide native core methods for RealSense device.
//

#include "RealSense.h"
#include "librealsense2/rsutil.h"
#include "librealsense2/rs_advanced_mode.hpp"
#include <exception>

namespace CSRT {

RealSense::RealSense(bool dpPostProcessing, int dimReduce) :
	depthWidth(0),
	depthHeight(0),
	colorWidth(0),
	colorHeight(0),
	fps(0),
	color_format(RS2_FORMAT_BGRA8),
	depth_format(RS2_FORMAT_Z16),
	depthScale(0.0f),
	maxDepth(0.0f),
	minDepth(0.0f),
	postProcessing(dpPostProcessing),
	dec_filter_step(dimReduce),
	colorBuffer(nullptr),
	depthBuffer(nullptr),
	pointCloud(nullptr),
	enableColor(false),
	enableDepth(false) { }

RealSense::~RealSense() {
	pipe.stop();
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;
}

void RealSense::Release() {
	pipe.stop();
	colorBuffer = nullptr;
	depthBuffer = nullptr;
	pointCloud = nullptr;
}

bool RealSense::Initialize(bool color, bool depth) {
	enableColor = color;
	enableDepth = depth;
	if (!enableColor && !enableDepth) {
		throw std::invalid_argument("RealSense::Initialize: read nothing from Kinect device.");
	}

	// Initial parameters
	colorWidth = 848;
	colorHeight = 480;
	depthWidth = 848;
	depthHeight = 480;
	fps = 30;
	color_format = RS2_FORMAT_BGRA8;
	depth_format = RS2_FORMAT_Z16;
	minDepth = 0.11f;
	maxDepth = 2.0f;

	// Configure RealSense device
	rs2::context ctx;
	auto devices = ctx.query_devices();
	if (devices.size() == 0) {
		throw std::runtime_error("RealSense::Initialize: cannot find ant RealSense device.");
	}
	rs2::device device = devices.front();
	std::string serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
	if (device.is<rs400::advanced_mode>()) {
		auto advanced_device = device.as<rs400::advanced_mode>();
		// Check if advanced-mode is enabled
		if (!advanced_device.is_enabled()) {
			// Enable advanced-mode
			advanced_device.toggle_advanced_mode(true);
		}
		auto depthTable = advanced_device.get_depth_table();
		depthTable.depthClampMax = (int)(maxDepth * depthTable.depthUnits);
		depthTable.depthClampMin = (int)(minDepth * depthTable.depthUnits);
		advanced_device.set_depth_table(depthTable);
	}

	// Configure RealSense pipeline
	rs2::config cfg;
	cfg.enable_device(serial);
	if (enableColor) cfg.enable_stream(RS2_STREAM_COLOR, -1, colorWidth, colorHeight, color_format, fps);
	if (enableDepth) cfg.enable_stream(RS2_STREAM_DEPTH, -1, depthWidth, depthHeight, depth_format, fps);
	auto profile = pipe.start(cfg);
	
	// Control the depth sensor
	if (enableDepth) {
		auto depthSensor = profile.get_device().first<rs2::depth_sensor>();
		if (depthSensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
			depthSensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
		}
		if (depthSensor.supports(RS2_OPTION_LASER_POWER)) {
			auto range = depthSensor.get_option_range(RS2_OPTION_LASER_POWER);
			depthSensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
		}
		rs2_rs400_visual_preset preset = RS2_RS400_VISUAL_PRESET_DEFAULT;
		if(depthSensor.supports(RS2_OPTION_VISUAL_PRESET)) {
			depthSensor.set_option(RS2_OPTION_VISUAL_PRESET, preset);
		}

		// Get depth units
		depthScale = depthSensor.get_depth_scale();

		// Configure post processing filters
		if (postProcessing) {
			dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, dec_filter_step);
			disparity2depth_filter = rs2::disparity_transform(false);
			spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2.0f);
			spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
			spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);
			// [0 - 5] range mapped to[none, 2, 4, 8, 16, unlimited] pixels.
			spat_filter.set_option(RS2_OPTION_HOLES_FILL, 2.0f);
			temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
			temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);
			temp_filter.set_option(RS2_OPTION_HOLES_FILL, 3.0f);
		}
	}

	// Wait for the first frame
	rs2::frameset frames = pipe.wait_for_frames();
	rs2::frame depth_frame, color_frame;
	if (enableDepth) {
		depth_frame = frames.get_depth_frame();
		if (postProcessing && dec_filter_step > 1) depth_frame = dec_filter.process(depth_frame);
		auto video_frame = depth_frame.as<rs2::video_frame>();
		depthWidth = video_frame.get_width();
		depthHeight = video_frame.get_height();
		depthCacheBuffer.resize(depthWidth*depthHeight);
	}
	if (enableColor) {
		color_frame = frames.get_color_frame();
		auto video_frame = color_frame.as<rs2::video_frame>();
		colorWidth = video_frame.get_width();
		colorHeight = video_frame.get_height();
	}

	// Get transformation parameters.
	if (enableColor && enableDepth) {
		auto depth_stream = depth_frame.get_profile();
		auto color_stream = color_frame.get_profile();
		depth2Color = depth_stream.get_extrinsics_to(color_stream);
		color2Depth = color_stream.get_extrinsics_to(depth_stream);
		depthIntrinsics = depth_stream.as<rs2::video_stream_profile>().get_intrinsics();
		colorIntrinsics = color_stream.as<rs2::video_stream_profile>().get_intrinsics();
	}

	return true;
}

void RealSense::Tick() {
	rs2::frameset frames;
	if (!pipe.poll_for_frames(&frames)) return;	// Not ready.

	if (enableColor) {
		rs2::frame color_frame = frames.get_color_frame();
		ProcessColor(reinterpret_cast<const uint8_t*>(color_frame.get_data()));
	}
	if(enableDepth) {
		rs2::frame depth_frame = frames.get_depth_frame();
		if(postProcessing) {
			if (dec_filter_step > 1) depth_frame = dec_filter.process(depth_frame);
			depth_frame = depth2disparity_filter.process(depth_frame);
			depth_frame = spat_filter.process(depth_frame);
			depth_frame = temp_filter.process(depth_frame);
			depth_frame = disparity2depth_filter.process(depth_frame);
		}
		ProcessDepth(reinterpret_cast<uint16_t*>(const_cast<void*>(depth_frame.get_data())));
		if (pointCloud) {
			rs2::points points = pc.calculate(depth_frame);
			memcpy(pointCloud, points.get_vertices(), points.size() * sizeof(CameraPoint3));
		}
	}
}

void RealSense::ProcessColor(const uint8_t* buffer) {
	if (!buffer) {
		throw std::invalid_argument("RealSense::ProcessColor: invalid color frame data.");
		return;
	}

	memcpy(&colorBuffer[0], buffer, colorWidth * colorHeight * 4);
}

void RealSense::ProcessDepth(uint16_t* buffer) {
	if (!buffer) {
		throw std::invalid_argument("RealSense::ProcessDepth: invalid depth frame data.");
		return;
	}

	int len = depthWidth * depthHeight;
	float *dst = depthBuffer;
	float scale = 1.0f / (maxDepth - minDepth);
	float offset = -minDepth * scale;
	memcpy(&depthCacheBuffer[0], buffer, len * sizeof(uint16_t));
	for (int i = 0; i < len; ++i) {
		*(dst++) = (*(buffer++)) * depthScale * scale + offset;
	}

	/*uint16_t* dst0 = &depthCacheBuffer[0];
	float* dst1 = depthBuffer;
	int len = depthWidth * depthHeight;
	float scale = 1.0f / (maxDepth - minDepth);
	float offset = -minDepth * scale;
	uint16_t low = minDepth / depthScale;
	uint16_t up = maxDepth / depthScale;

	for (int i = 0; i < len; ++i) {
		*buffer = CameraClamp(*buffer, low, up);
		*dst0 = *buffer;
		*dst1 = (*dst0) * depthScale * scale + offset;
		++buffer;
		++dst0;
		++dst1;
	}*/
}

void RealSense::MapColor2Depth(const CameraPoint2* colorPoints, CameraPoint2* depthPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("RealSense::MapColor2Depth: cannot do mapping for the lack of data.");
		return;
	}

	for (int i = 0; i < count; ++i) {
		rs2_project_color_pixel_to_depth_pixel(reinterpret_cast<float*>(&depthPoints[i]), &depthCacheBuffer[0], depthScale,
			minDepth, maxDepth, &depthIntrinsics, &colorIntrinsics, &color2Depth, &depth2Color, reinterpret_cast<const float*>(&colorPoints[i]));
	}
}

void RealSense::MapDepth2Color(const CameraPoint2* depthPoints, CameraPoint2* colorPoints, int count) {
	if (!enableColor || !enableDepth) {
		throw std::runtime_error("RealSense::MapDepth2Color: cannot do mapping for the lack of data.");
	}

	float pt0[3], pt1[3];
	for (int i = 0; i < count; ++i) {
		rs2_deproject_pixel_to_point(pt0, &depthIntrinsics, reinterpret_cast<const float*>(&depthPoints[i]),
			depthCacheBuffer[depthPoints[i].Y * depthWidth + depthPoints[i].X] * depthScale);
		rs2_transform_point_to_point(pt1, &depth2Color, pt0);
		rs2_project_point_to_pixel(reinterpret_cast<float*>(&colorPoints[i]), &colorIntrinsics, pt1);
	}
}

}	// namespace CSRT