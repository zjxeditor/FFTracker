//
// Different camera device test. Current support devices are: RealSense, KinectV1 and KinectV2.
//

#include "Camera/Camera.h"
#include <opencv2/opencv.hpp>
#include <memory>

using namespace CSRT;

int main() {
	// Create logger
	float colorScale = 1.0f;

	// Configure camera
	std::unique_ptr<CameraService> camera = CreateCameraService(CameraType::RealSense, true, 2);
	if(!camera->Initialize(true, true)) {
		std::cout << "Cannot initialize camera device." << std::endl;
		return -1;
	}
	int colorWidth = camera->GetColorWidth();
	int colorHeight = camera->GetColorHeight();
	int depthWidth = camera->GetDepthWidth();
	int depthHeight = camera->GetDepthHeight();

	std::vector<uint8_t> colorData(colorWidth * colorHeight * 4);
	std::vector<float> depthData(depthWidth * depthHeight);
	camera->SetColorData(&colorData[0]);
	camera->SetDepthData(&depthData[0]);

	// Configure OpenCV.
	std::string ColorWindowName = "ColorFrame";
	std::string DepthWindowName = "DepthFrame";
	cv::namedWindow(ColorWindowName, 1);
	cv::namedWindow(DepthWindowName, 1);
	cv::Mat cvImage(colorHeight, colorWidth, CV_8UC4);
	cv::Mat cvScaledImage;
	cv::Mat cvDepthImage(depthHeight, depthWidth, CV_32FC1);
	cv::Mat cvDepthGrayImage, cvDepthShowImage;

	std::vector<CameraPoint2> colorPoints;
	std::vector<CameraPoint2> depthPoints;
	// Main loop.
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27) break;
		
		camera->Tick();
		memcpy(cvImage.data, &colorData[0], colorWidth*colorHeight * 4 * sizeof(uint8_t));
		memcpy(cvDepthImage.data, &depthData[0], depthWidth*depthHeight * sizeof(float));
		cv::resize(cvImage, cvScaledImage, cv::Size(), colorScale, colorScale);
		cvDepthImage.convertTo(cvDepthGrayImage, CV_8UC1, 255.0, 0.0);
		cv::cvtColor(cvDepthGrayImage, cvDepthShowImage, CV_GRAY2BGR);

		cv::imshow(ColorWindowName, cvScaledImage);
		cv::imshow(DepthWindowName, cvDepthShowImage);

		// Color to depth
		//cv::Rect roi = cv::selectROI("roi", cvScaledImage, true, false);
		//cv::rectangle(cvScaledImage, roi, cv::Scalar(0, 0, 255));
		//roi.x = (int)(roi.x / colorScale);
		//roi.y = (int)(roi.y / colorScale);
		//roi.width = (int)(roi.width / colorScale);
		//roi.height = (int)(roi.height / colorScale);
		//colorPoints.resize(2);
		//depthPoints.resize(2);
		//colorPoints[0].X = roi.x;
		//colorPoints[0].Y = roi.y;
		//colorPoints[1].X = roi.x + roi.width;
		//colorPoints[1].Y = roi.y + roi.height;
		//camera->MapColor2Depth(&colorPoints[0], &depthPoints[0], 2);
		//roi.x = depthPoints[0].X;
		//roi.y = depthPoints[0].Y;
		//roi.width = depthPoints[1].X - roi.x;
		//roi.height = depthPoints[1].Y - roi.y;
		//cv::rectangle(cvDepthShowImage, roi, cv::Scalar(255, 0, 0));

		// Depth to color
		/*cv::Rect roi = cv::selectROI("roi", cvDepthShowImage, true, false);
		cv::rectangle(cvDepthShowImage, roi, cv::Scalar(0, 0, 255));
		colorPoints.resize(2);
		depthPoints.resize(2);
		depthPoints[0].X = roi.x;
		depthPoints[0].Y = roi.y;
		depthPoints[1].X = roi.x + roi.width;
		depthPoints[1].Y = roi.y + roi.height;
		camera->MapDepth2Color(&depthPoints[0], &colorPoints[0], 2);
		roi.x = colorPoints[0].X;
		roi.y = colorPoints[0].Y;
		roi.width = colorPoints[1].X - roi.x;
		roi.height = colorPoints[1].Y - roi.y;
		roi.x = (int)(roi.x * colorScale);
		roi.y = (int)(roi.y * colorScale);
		roi.width = (int)(roi.width * colorScale);
		roi.height = (int)(roi.height * colorScale);
		cv::rectangle(cvScaledImage, roi, cv::Scalar(255, 0, 0));*/

		/*cv::imshow(ColorWindowName, cvScaledImage);
		cv::imshow(DepthWindowName, cvDepthShowImage);*/
	}

	return 0;
}