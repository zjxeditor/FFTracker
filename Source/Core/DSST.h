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
#include "../Utility/Mat.h"

namespace CSRT {

class DSST {
public:
	// DSST Public Methods
	DSST(float learnRateOfScale) :
		initialized(false), scaleLearnRate(learnRateOfScale), 
		arenas(nullptr), backgroundUpdateFlag(false) {}

	~DSST() {
		if (arenas) {
			delete[] arenas;
			arenas = nullptr;
		}
	}

	// Set initialize flag to re-initialize.
	void SetReinitialize() { initialized = false; }

	// Initialize this DSST. 
	// feats: the scale representation features.
	// gsfRow: ideal 1 dimension gaussian response.
	void Initialize(const MatF &feats, const MatCF &gsfRow);

	// Get the estimated scale value. The feats matrix are extracted at target object previous location.
	void GetScale(const MatF &feats, const std::vector<float> &scaleFactors, const float &currentScale, 
		float minScaleFactor, float maxScaleFactor, float &newScale);

	// Update the DSST model. The feats matrix are extracted at target object current location.
	void Update(const MatF &feats);

	// Background update logic.
	void StartBackgroundUpdate();
	void FetchUpdateResult();
	void BackgroundUpdate(const MatF& feats);	// The feats matrix are extracted at target object current location.

private:
	void ResetArenas(bool background = false) const;

	// DSST Private Data
	bool initialized;				// Initialize flag.
	float scaleLearnRate;			// Learning rate of the scale model.

	MatCF gsf;			// Frequency response.
	MatCF hfNum;		// Filter numerator.
	MatCF hfDen;		// Filter denominator.

	// DSST memory arena management.
	MemoryArena *arenas;

	// Background thread backup.
	MatCF bk_hfNum;
	MatCF bk_hfDen;
	bool backgroundUpdateFlag;
};

}	// namespace CSRT

#endif	// CSRT_DSST_H