#include "Export/CSRTracker.h"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <memory>

#include "CSRT.h"

using namespace CSRT;

int main() {
	SetThread();
	StartSystem();

	float scale = 0.5f;
	int radius = 80;

	CSRTrackerParams params;
	params.UseHOG = true;
	params.UseCN = true;
	params.UseGRAY = true;
	params.UseRGB = false;
	params.NumHOGChannelsUsed = 18;
	params.UsePCA = false;
	params.PCACount = 12;
	params.UseChannelWeights = true;
	params.UseSegmentation = true;
	params.AdmmIterations = 4;
	params.Padding = 3.0f;
	params.TemplateSize = 200;
	params.GaussianSigma = 1.0f;
	params.WindowFunc = CSRTWindowType::Hann;
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
	params.PeakRatio = 0.1f;
	params.PSRThreshold = 20.0f;
	params.UseScale = true;
	params.UseSmoother = true;

	// Configure OpenCV.
	cv::Scalar WaitColor = cv::Scalar(0, 0, 255);
	cv::Scalar TrackColor = cv::Scalar(0, 255, 0);
	std::string WindowName = "Tracker";
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cout << "cannot open camera" << std::endl;
		CloseSystem();
		return -1;
	}
	int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int cameraFps = (int)cap.get(CV_CAP_PROP_FPS);
	if (std::min(width, height) / 2 <= radius) {
		std::cout << "please reduce radius" << std::endl;
		CloseSystem();
		return -1;
	}
	std::cout << "camera: " << width << " x " << height << ", fps " << cameraFps << std::endl;

	cv::Mat cvImage;
	cv::namedWindow(WindowName, 1);

	// Create tracker.
	std::unique_ptr<CSRTracker> pTracker = std::unique_ptr<CSRTracker>(new CSRTracker(height, width, scale, params));
	//pTracker->SetDrawMode(true, 255, 0, 0, 0.5f);

	TimePt startTime;
	TimePt endTime;
	int frames = 0;

	// Main loop.
	bool started = false;
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27) break;

		cap >> cvImage;
		cv::flip(cvImage, cvImage, 1);

		cv::cvtColor(cvImage, cvImage, CV_BGR2RGB);
		if (cvImage.step != cvImage.cols * 3 || !cvImage.isContinuous()) {
			std::cout << "Image data must be continous without any padding" << std::endl;
			break;
		}

		if (key == 32 && !started) {
			started = true;
			Bounds initbb;
			initbb.x0 = width / 2 - radius;
			initbb.y0 = height / 2 - radius;
			initbb.x1 = initbb.x0 + radius * 2;
			initbb.y1 = initbb.y0 + radius * 2;
			pTracker->Initialize(cvImage.data, 3, initbb);
			int centerX = (initbb.x0 + initbb.x1) / 2;
			int centerY = (initbb.y0 + initbb.y1) / 2;
			int newRadius = std::max(initbb.x1 - initbb.x0, initbb.y1 - initbb.y0) / 2;
			cv::circle(cvImage, cv::Point(centerX, centerY), newRadius, TrackColor, 2);
			startTime = TimeNow();
		} else if (started) {
			Bounds outputbb;
			float score = 0.0f;
			bool res = pTracker->Update(cvImage.data, 3, outputbb, score);
			int centerX = (outputbb.x0 + outputbb.x1) / 2;
			int centerY = (outputbb.y0 + outputbb.y1) / 2;
			int newRadius = std::max(outputbb.x1 - outputbb.x0, outputbb.y1 - outputbb.y0) / 2;
			cv::circle(cvImage, cv::Point(centerX, centerY), newRadius, TrackColor, 2);
			//std::cout << score << std::endl;
			++frames;
		} else {
			cv::circle(cvImage, cv::Point(width / 2, height / 2), radius, WaitColor, 2);
		}

		cv::cvtColor(cvImage, cvImage, CV_RGB2BGR);
		cv::imshow(WindowName, cvImage);
	}

	endTime = TimeNow();
	float fps = frames / (Duration(startTime, endTime) * 1e-6f);
	std::cout << "fps: " << fps << std::endl;

	// the camera will be deinitialized automatically in VideoCapture destructor
	cap.release();
	pTracker.reset();
	CloseSystem();
	return 0;
}