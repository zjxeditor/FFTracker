// Feature extraction test.

#include <opencv2/opencv.hpp>
#include "Camera/KinectService2.h"
#include "Camera/RealSense.h"

#include "CSRT.h"
#include "Utility/Parallel.h"
#include "Utility/Mat.h"
#include "Core/FeaturesExtractor.h"

using namespace CSRT;

int main() {
	CSRT::CreateLogger();
	CSRT::SetThreadCount(NumSystemCores() + 1);
	CSRT::ParallelInit();
	GFeatsExtractor.Initialize();

	// Configure camera
	std::unique_ptr<CameraService> camera(new RealSense(false, 2));
	if (!camera->Initialize(false, true)) {
		std::cout << "Cannot initialize camera device." << std::endl;
		return -1;
	}
	int depthWidth = camera->GetDepthWidth();
	int depthHeight = camera->GetDepthHeight();
	float minDepth = camera->GetMinDepth();
	float maxDepth = camera->GetMaxDepth();

	std::vector<float> depthData(depthWidth * depthHeight);
	camera->SetDepthData(&depthData[0]);
	std::vector<Vector3f> pointCloud(depthWidth * depthHeight);
	camera->SetPointCloud(reinterpret_cast<CameraPoint3*>(&pointCloud[0]));

	// Configure openCV
	std::string FeatureWindowName = "FeatureFrame";
	cv::namedWindow(FeatureWindowName, 1);
	cv::Mat cvFeatureImage(depthHeight, depthWidth, CV_8UC3);

	// Data elements
	MatF depth(depthHeight, depthWidth);
	MatF normal(depthHeight, depthWidth * 3);
	Mat feature(depthHeight, depthWidth, 3);
	std::vector<MatF> features;

	// Main loop
	bool started = false;
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27) break;

		camera->Tick();
		memcpy(depth.Data(), &depthData[0], depth.Size() * sizeof(float));

		GFeatsExtractor.ComputeNormalFromPointCloud(&pointCloud[0], reinterpret_cast<Vector3f*>(normal.Data()), depthWidth, depthHeight);
		normal.ToMat(feature, 3, 255.0f / 2.0f, 255.0f / 2.0f);

		//GFeatsExtractor.GetDepthFeaturesHON(normal, features, 4, 5, 10);
		/*GFeatsExtractor.GetDepthFeaturesHOG(depth, features, 4);

		for (int i = 0; i < features.size(); ++i) {
			SaveToFile("dhog" + std::to_string(i), features[i]);
		}*/

		memcpy(cvFeatureImage.data, feature.Data(), feature.Size() * sizeof(uint8_t));
		cv::flip(cvFeatureImage, cvFeatureImage, 1);
		cv::imshow(FeatureWindowName, cvFeatureImage);
	}

	CSRT::ParallelCleanup();
	CSRT::Info("All work is done!");
	CSRT::ClearLogger();

	return 0;
}