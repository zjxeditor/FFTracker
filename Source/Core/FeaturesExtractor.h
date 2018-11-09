//
// Image features extraction.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_FEATURES_H
#define CSRT_FEATURES_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"

namespace CSRT {

// Standard color names features.
extern const float ColorNames[][10];

class FeaturesExtractor {
public:
	FeaturesExtractor();
	~FeaturesExtractor();
	void Initialize();

	//
	// RGB image features
	//

	// Extract 32 dimension HOG features.
	void GetFeaturesHOG(const Mat &im, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);
	// Extract color names features.
	void GetFeaturesCN(const Mat &img, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);
	// Extract RGB features.
	void GetFeaturesRGB(const Mat &img, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);

	//
	// Depth image features
	//

	// Extract 1 dimension depth gray feature.
	void GetDepthFeaturesGray(const MatF &depth, MatF &feature, const Vector2i &OutSize);

	// Extract 32 dimension depth HOG features.
	void GetDepthFeaturesHOG(const MatF &depth, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);

	// Calculate normal information from depth map.
	void ComputeNormalFromPointCloud(const Vector3f *pointCloud, Vector3f *normal, int width, int height);

	// Calculate polar normal information from depth map. First component is phi [0, Pi/2], second component is theta [0, 2Pi], both are normalized to [0, 1].
	void ComputePolarNormalFromPointCloud(const Vector3f *pointCloud, Vector2f *polarNormal, int width, int height);

	// Convert normalized normal to normalized polar representation.
	void ConvertNormalToPolarNormal(const Vector3f *normal, Vector2f *polarNormal, int width, int height);

	// Convert normalized polar normal to normalized normal representation.
	void ConvertPolarNormalToNormal(const Vector2f *polarNormal, Vector3f *normal, int width, int height);

	// Extract polar normal features.
	void GetDepthFeaturesPolarNormal(const MatF &normal, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);

	// Calculate normal histogram features. Note the normal value must has been normalized and is in polar representation.
	void GetDepthFeaturesHON(const MatF &normal, std::vector<MatF> &features, int binSize, int phiBinNum, int thetaBinNum, MemoryArena *targetArena = nullptr);

	// Extract 32 dimension depth polar normal HOG features.
	void GetDepthFeaturesNormalHOG(const MatF &normal, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);
	
private:
	// Compute the 32 dimensional hog features.
	void ComputeHOG32D(const Mat& img, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena = nullptr);

	// Compute the 32 dimensional hog features for depth map.
	void ComputeDepthHOG32D(const MatF& depth, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena = nullptr);

	// Compute the 32 dimensional hon features for depth map. phi -> [0, PI/2], theta -> [0, 2PI].
	void ComputeDepthHON(const MatF &normal, std::vector<MatF> &features, int binSize, int phiBinNum, int thetaBinNum, 
		int padx, int pady, MemoryArena *targetArena = nullptr);

	// Compute the 32 dimensional normal hog features for polar normal map.
	void ComputeNormalHOG32D(const MatF& normal, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena = nullptr);

	bool initialized;
	static bool SingleInstance;
};

// Global Feature Extractor
extern FeaturesExtractor GFeatsExtractor;

}	// namespace CSRT

#endif	// CSRT_FEATURES_H