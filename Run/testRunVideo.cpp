// Produce testing tracker video.

#include "../Source/Core/Processor.h"
#include "../Source/Camera/Camera.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>

using namespace CSRT;

int main() {
	StartSystem();

	float scale = 1.0f;
	int radius = 68;
	int targetCount = 2;

	float learnRates[3] = { 0.02f, 0.08f, 0.16f };
	TrackerParams params;
	params.UseHOG = true;
	params.UseCN = true;
	params.UseGRAY = true;
	params.UseRGB = false;
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
	params.AdmmIterations = 3;
	params.Padding = 3.0f;
	params.TemplateSize = 200;
	params.GaussianSigma = 1.5f;
	params.WindowFunc = WindowType::Hann;
	params.ChebAttenuation = 45.0f;
	params.KaiserAlpha = 3.75f;
	params.WeightsLearnRates = &learnRates[0];
	params.FilterLearnRates = &learnRates[0];
	params.LearnRateNum = sizeof(learnRates) / sizeof(learnRates[0]);
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
	params.UpdateInterval = 0;
	params.UseScale = false;
	params.UseSmoother = false;
	params.UseFastScale = false;
	params.FailThreshold = 0.08f;

	// Configure camera
	std::unique_ptr<CameraService> camera = CreateCameraService(CameraType::RealSense, true, 2);
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
	std::vector<Bounds2f> initbbs(targetCount, Bounds2f());
	float deltaX = ((float)postWidth) / (targetCount + 1);
	float centerY = postHeight / 2.0f;
	float centerX = 0.0f;
	for (int i = 0; i < targetCount; ++i) {
		centerX += deltaX;
		initbbs[i].pMin.x = centerX - radius;
		initbbs[i].pMin.y = centerY - radius;
		initbbs[i].pMax.x = centerX + radius;
		initbbs[i].pMax.y = centerY + radius;
	}

	TimePt startTime;
	TimePt endTime;
	int frames = 0;

	// Create video writer
	cv::VideoWriter outputVideo;
	outputVideo.open("trackRGB.avi", CV_FOURCC('M', 'J', 'P', 'G'), 20, cv::Size(postWidth, postHeight), true);

	// Main loop.
	bool started = false;
	startTime = TimeNow();
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27) break;

		// Fetch data
		camera->Tick();
		Rgba2Rgb(rawColorFrame.Data(), colorFrame.Data(), colorHeight, colorWidth);
		colorFrame.Resize(postFrame, postHeight, postWidth);
		memcpy(cvImage.data, postFrame.Data(), postWidth*postHeight * 3 * sizeof(uint8_t));

		if (!started && Duration(startTime, TimeNow()) > 6e6) {
			started = true;
			pTracker->Initialize(postFrame, initbbs);
			for (int i = 0; i < targetCount; ++i) {
				int centerx = (int)std::floor((initbbs[i].pMin.x + initbbs[i].pMax.x) / 2);
				int centery = (int)std::floor((initbbs[i].pMin.y + initbbs[i].pMax.y) / 2);
				int newRadius = (int)std::floor(std::max(initbbs[i].pMax.x - initbbs[i].pMin.x, initbbs[i].pMax.y - initbbs[i].pMin.y) / 2);
				cv::circle(cvImage, cv::Point(centerx, centery), newRadius, TrackColor, 2);
			}
			startTime = TimeNow();
			cv::flip(cvImage, cvImage, 1);
			outputVideo.write(cvImage);
		} else if (started) {
			std::vector<Bounds2f> outputbbs;
			pTracker->Update(postFrame, outputbbs);
			for (int i = 0; i < targetCount; ++i) {
				int centerx = (int)std::floor((outputbbs[i].pMin.x + outputbbs[i].pMax.x) / 2);
				int centery = (int)std::floor((outputbbs[i].pMin.y + outputbbs[i].pMax.y) / 2);
				int newRadius = (int)std::floor(std::max(outputbbs[i].pMax.x - outputbbs[i].pMin.x, outputbbs[i].pMax.y - outputbbs[i].pMin.y) / 2);
				cv::circle(cvImage, cv::Point(centerx, centery), newRadius, TrackColor, 2);
			}
			//std::cout << score << std::endl;
			++frames;
			cv::flip(cvImage, cvImage, 1);
			outputVideo.write(cvImage);
		} else {
			for (int i = 0; i < targetCount; ++i) {
				int centerx = (int)std::floor((initbbs[i].pMin.x + initbbs[i].pMax.x) / 2);
				int centery = (int)std::floor((initbbs[i].pMin.y + initbbs[i].pMax.y) / 2);
				int newRadius = (int)std::floor(std::max(initbbs[i].pMax.x - initbbs[i].pMin.x, initbbs[i].pMax.y - initbbs[i].pMin.y) / 2);
				cv::circle(cvImage, cv::Point(centerx, centery), newRadius, WaitColor, 2);
			}
			cv::flip(cvImage, cvImage, 1);
		}

		cv::imshow(WindowName, cvImage);
	}

	endTime = TimeNow();
	float fps = frames / (Duration(startTime, endTime) * 1e-6f);
	std::cout << "fps: " << fps << std::endl;

	// the camera will be deinitialized automatically in VideoCapture destructor
	camera->Release();
	outputVideo.release();
	pTracker.reset();
	CloseSystem();
	return 0;
}