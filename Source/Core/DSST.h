//
// Scale estimation using DSST method.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_DSST_H
#define CSRT_DSST_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"

namespace CSRT {

class DSST {
public:
	// DSST Public Methods
	DSST() : initialized(false), currentScaleFactor(0.0f), minScaleFactor(0.0f), maxScaleFactor(0.0f),
		scaleCount(0), scaleStep(0.0f), scaleSigma(0.0f), learnRate(0.0f), maxModelArea(0.0f), arenas(nullptr), 
		bk_scale(0.0f), backgroundUpdateFlag(false) {}
	
	~DSST() {
		if(arenas) {
			delete[] arenas;
			arenas = nullptr;
		}
	}

	// Initialize this DSST.
	void Initialize(const Mat &image, const Bounds2i &bb, int numberOfScales, float stepOfScale, float sigmaFactor, float scaleLearnRate,
		float maxArea, const Vector2i &templateSize, float scaleModelFactor = 1.0f);

	// Set initialize flag to re-initialize.
	inline void SetReinitialize() { initialized = false; }

	// Get the estimated scale value.
	float GetScale(const Mat &image, const Vector2i &objectCenter);

	// Update the DSST model.
	void Update(const Mat &image, const Vector2i &objectCenter);

	// Background update logic.
	void StartBackgroundUpdate();
	void FetchUpdateResult();
	void BackgroundUpdate(const Mat& image, const Vector2i& objectCenter);

private:
	// Get image features used for DSST. Currently, only HOG features are applied.
	void GetScaleFeatures(const Mat &img, MatF &feat, const Vector2i &pos, const Vector2i &orgSize, float currentScale,
		const std::vector<float> &factors, const MatF &window, const Vector2i &modelSize) const;

	void ResetArenas(bool background = false) const;

	// DSST Private Data
	bool initialized;				// Initialize flag.

	Vector2i orgTargetSize;		// Original target size.	
	Vector2i scaleModelSize;	// Final calculation size used for DSST. 
	float currentScaleFactor;	// Current scale factor for this DSST.
	float minScaleFactor;		// Minimum scale factor for this DSST.
	float maxScaleFactor;		// Maximum scale factor for this DSST.

	MatF gs;			// Spatial response.
	MatCF gsf;			// Frequency response.
	MatF scaleWindow;	// Signal window.
	MatCF hfNum;		// Filter numerator.
	MatCF hfDen;		// Filter denominator.

	int scaleCount;
	float scaleStep;
	std::vector<float> scaleFactors;
	float scaleSigma;	// Sigma value to calculate standard gaussian response.
	float learnRate;

	float maxModelArea;	// Maximum supported area.

	// DSST memory arena management.
	MemoryArena *arenas;

	// Background thread backup.
	float bk_scale;
	MatCF bk_hfNum;
	MatCF bk_hfDen;
	bool backgroundUpdateFlag;
};

}	// namespace CSRT

#endif	// CSRT_DSST_H