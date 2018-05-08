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

	// Extract 32 dimension HOG features.
	void GetFeaturesHOG(const Mat &im, std::vector<MatF> &features, int binSize, MemoryArena *targetArena = nullptr);
	// Extract color names features.
	void GetFeaturesCN(const Mat &img, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);
	// Extract RGB features.
	void GetFeaturesRGB(const Mat &img, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena = nullptr);

private:
	// Computer the 32 dimensional hog features.
	void ComputeHOG32D(const Mat& img, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena = nullptr);

	bool initialized;
	static bool SingleInstance;
};

// Global Feature Extractor
extern FeaturesExtractor GFeatsExtractor;

}	// namespace CSRT

#endif	// CSRT_FEATURES_H