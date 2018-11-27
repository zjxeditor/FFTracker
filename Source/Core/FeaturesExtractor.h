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
	void GetFeaturesHOG(const MatF &im, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);
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

	// Extract polar normal features.
	void GetDepthFeaturesPolarNormal(const MatF &normal, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);

	// Calculate normal histogram features. Note the normal value must has been normalized and is in polar representation.
	void GetDepthFeaturesHON(const MatF &normal, std::vector<MatF> &features, int binSize, int phiBinNum, int thetaBinNum, MemoryArena *targetArena = nullptr);

	// Extract 32 dimension depth polar normal HOG features.
	void GetDepthFeaturesNormalHOG(const MatF &normal, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);
	
private:
	// Compute the 32 dimensional hog features.
	void ComputeHOG32D(const MatF& img, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena = nullptr);

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