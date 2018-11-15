// Real time tracker test.

#include "CSRT.h"
#include "Utility/Geometry.h"
#include "Utility/Parallel.h"
#include "Utility/Mat.h"
#include "Utility/FFT.h"
#include "Core/FeaturesExtractor.h"
#include "Core/Filter.h"
#include "Core/InfoProvider.h"
#include "Core/Processor.h"
#include <fstream>
#include "Core/Filter.h"
#include "Camera/KinectService2.h"
#include "Camera/RealSense.h"

#include <opencv2/opencv.hpp>
#include <memory>

using namespace CSRT;

inline void StartSystem() {
	CSRT::CreateLogger();
	CSRT::SetThreadCount(NumSystemCores() + 1);
	CSRT::ParallelInit();
	Eigen::setNbThreads(MaxThreadIndex());
	Eigen::initParallel();
	GFFT.Initialize();
	GImageMemoryArena.Initialize();
	GFeatsExtractor.Initialize();
	GFilter.Initialize();
	GSegment.Initialize();
	GInfoProvider.Initialize();
}

inline void CloseSystem() {
	CSRT::ParallelCleanup();
	CSRT::Info("All work is done!");
	CSRT::ClearLogger();
}

inline  void Rgba2Rgb(unsigned char *srcData, unsigned char *dstData, int rows, int cols) {
	ParallelFor([&](int64_t y) {
		const unsigned char *ps = srcData + y * cols * 4;
		unsigned char *pd = dstData + y * cols * 3;
		for (int i = 0; i < cols; ++i) {
			*(pd++) = *(ps++);
			*(pd++) = *(ps++);
			*(pd++) = *(ps++);
			++ps;
		}
	}, rows, 32);
}

int main() {
	StartSystem();

	float scale = 0.8f;
	int radius = 60;
	int targetCount = 2;

	TrackerParams params;
	//params.UseHOG = true;
	//params.UseCN = true;
	//params.UseGRAY = true;
	//params.UseRGB = false;
	//params.NumHOGChannelsUsed = 18;
	//params.UsePCA = false;
	//params.PCACount = 20;
	//params.UseChannelWeights = true;
	//params.UseSegmentation = true;
	//params.AdmmIterations = 4;
	//params.Padding = 3.0f;
	//params.TemplateSize = 200;
	//params.GaussianSigma = 1.0f;
	//params.WindowFunc = CSRTWindowType::Hann;
	//params.ChebAttenuation = 45.0f;
	//params.KaiserAlpha = 3.75f;
	//params.WeightsLearnRate = 0.02f;
	//params.FilterLearnRate = 0.02f;
	//params.HistLearnRate = 0.04f;
	//params.ScaleLearnRate = 0.025f;
	//params.BackgroundRatio = 2.0f;
	//params.HistogramBins = 16;
	//params.PostRegularCount = 16;
	//params.MaxSegmentArea = 1024.0f;
	//params.ScaleCount = 33;
	//params.ScaleSigma = 0.25f;
	//params.ScaleMaxArea = 512.0f;
	//params.ScaleStep = 1.02f;
	//params.UpdateInterval = 1;
	//params.PeakRatio = 0.1f;
	//params.PSRThreshold = 12.0f;

	params.UseHOG = true;
	params.UseCN = true;
	params.UseGRAY = true;
	params.UseRGB = true;
	params.UseDepthHOG = true;
	params.UseDepthGray = true;
	params.UseDepthNormal = true;
	params.UseDepthHON = true;
	params.UseDepthNormalHOG = true;
	params.NumHOGChannelsUsed = 18;
	params.PCACount = 0;
	params.UseNormalForSegment = true;
	params.UseNormalForDSST = true;
	params.UseChannelWeights = true;
	params.UseSegmentation = true;
	params.CellSize = 4;
	params.AdmmIterations = 4;
	params.Padding = 3.0f;
	params.TemplateSize = 200;
	params.GaussianSigma = 1.0f;
	params.WindowFunc = WindowType::Hann;
	params.ChebAttenuation = 45.0f;
	params.KaiserAlpha = 3.75f;
	params.WeightsLearnRate = 0.02f;
	params.FilterLearnRate = 0.02f;
	params.HistLearnRate = 0.04f;
	params.ScaleLearnRate = 0.025f;
	params.BackgroundRatio = 2.0f;
	params.HistogramBins = 16;
	params.PostRegularCount = 16;
	params.MaxSegmentArea = 1024.0f;
	params.ScaleCount = 33;
	params.ScaleSigma = 0.25f;
	params.ScaleMaxArea = 512.0f;
	params.ScaleStep = 1.02f;
	params.UpdateInterval = 1;
	params.UseScale = false;
	params.UseSmoother = true;

	// Configure camera
	std::unique_ptr<CameraService> camera(new RealSense(true, 2));
	if (!camera->Initialize(true, false)) {
		std::cout << "Cannot initialize camera device." << std::endl;
		CloseSystem();
		return -1;
	}
	int colorWidth = camera->GetColorWidth();
	int colorHeight = camera->GetColorHeight();
	int postWidth = colorWidth * scale;
	int postHeight = colorHeight * scale;
	Mat rawColorFrame(colorHeight, colorWidth, 4);
	Mat colorFrame(colorHeight, colorWidth, 3);
	camera->SetColorData(rawColorFrame.Data());

	// Configure OpenCV.
	cv::Scalar WaitColor = cv::Scalar(0, 0, 255);
	cv::Scalar TrackColor = cv::Scalar(0, 255, 0);
	std::string WindowName = "Tracker";
	cv::namedWindow(WindowName, 1);
	cv::Mat cvImage(postHeight, postWidth, CV_8UC3);

	// Create tracker.
	std::unique_ptr<Processor> pTracker(new Processor(params, targetCount, Vector2i(postWidth, postHeight), TrackMode::RGB));
	Mat postFrame;

	// Create initial bounding boxes.
	std::vector<Bounds2i> initbbs(targetCount, Bounds2i());
	int deltaX = postWidth / (targetCount + 1);
	int centerY = postHeight / 2;
	int centerX = 0;
	for (int i = 0; i < targetCount; ++i) {
		centerX += deltaX;
		initbbs[i].pMin.x = centerX - radius;
		initbbs[i].pMin.y = centerY - radius;
		initbbs[i].pMax.x = centerX + radius;
		initbbs[i].pMax.y = centerY + radius;
	}

	// Main loop.
	bool started = false;
	TimePt startTime = TimeNow();
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27) break;

		// Fetch data
		camera->Tick();
		Rgba2Rgb(rawColorFrame.Data(), colorFrame.Data(), colorHeight, colorWidth);
		colorFrame.Resize(postFrame, postHeight, postWidth);
		memcpy(cvImage.data, postFrame.Data(), postWidth*postHeight * 3 * sizeof(uint8_t));

		if (!started && Duration(startTime, TimeNow()) > 5e6) {
			started = true;
			pTracker->Initialize(postFrame, initbbs);
			for (int i = 0; i < targetCount; ++i) {
				int centerX = (initbbs[i].pMin.x + initbbs[i].pMax.x) / 2;
				int centerY = (initbbs[i].pMin.y + initbbs[i].pMax.y) / 2;
				int newRadius = std::max(initbbs[i].pMax.x - initbbs[i].pMin.x, initbbs[i].pMax.y - initbbs[i].pMin.y) / 2;
				cv::circle(cvImage, cv::Point(centerX, centerY), newRadius, TrackColor, 2);
			}
		} else if (started) {
			std::vector<Bounds2i> outputbbs;
			pTracker->Update(postFrame, outputbbs);
			for (int i = 0; i < targetCount; ++i) {
				int centerX = (outputbbs[i].pMin.x + outputbbs[i].pMax.x) / 2;
				int centerY = (outputbbs[i].pMin.y + outputbbs[i].pMax.y) / 2;
				int newRadius = std::max(outputbbs[i].pMax.x - outputbbs[i].pMin.x, outputbbs[i].pMax.y - outputbbs[i].pMin.y) / 2;
				cv::circle(cvImage, cv::Point(centerX, centerY), newRadius, TrackColor, 2);
			}
		} else {
			for (int i = 0; i < targetCount; ++i) {
				int centerX = (initbbs[i].pMin.x + initbbs[i].pMax.x) / 2;
				int centerY = (initbbs[i].pMin.y + initbbs[i].pMax.y) / 2;
				int newRadius = std::max(initbbs[i].pMax.x - initbbs[i].pMin.x, initbbs[i].pMax.y - initbbs[i].pMin.y) / 2;
				cv::circle(cvImage, cv::Point(centerX, centerY), newRadius, WaitColor, 2);
			}
		}

		cv::flip(cvImage, cvImage, 1);
		cv::imshow(WindowName, cvImage);
	}

	camera->Release();
	pTracker.reset();
	CloseSystem();
	return 0;
}