//
// Scale estimation using RotationEstimator method.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_ROTATION_ESTIMATOR_H
#define CSRT_ROTATION_ESTIMATOR_H

#include "../CSRT.h"
#include "../Utility/Mat.h"

namespace CSRT {

class RotationEstimator {
public:
    // RotationEstimator Public Methods
    RotationEstimator(float learnRateOfRotation) :
            initialized(false), rotationLearnRate(learnRateOfRotation),
            arenas(nullptr), backgroundUpdateFlag(false) {}

    ~RotationEstimator() {
        if (arenas) {
            delete[] arenas;
            arenas = nullptr;
        }
    }

    // Set initialize flag to re-initialize.
    void SetReinitialize() { initialized = false; }

    // Initialize this RotationEstimator.
    // feats: the rotation representation features.
    // gsfRow: ideal 1 dimension gaussian response.
    void Initialize(const MatF &feats, const MatCF &gsfRow);

    // Get the estimated rotation value. The feats matrix are extracted at target object previous location.
    void GetRotation(const MatF &feats, const std::vector<float> &rotationFactors, float currentRotation, float &newRotation);

    // Update the RotationEstimator model. The feats matrix are extracted at target object current location.
    void Update(const MatF &feats);

    // Background update logic.
    void StartBackgroundUpdate();
    void FetchUpdateResult();
    void BackgroundUpdate(const MatF& feats);	// The feats matrix are extracted at target object current location.

private:
    void ResetArenas(bool background = false) const;

    // RotationEstimator Private Data
    bool initialized;				// Initialize flag.
    float rotationLearnRate;		// Learning rate of the rotation model.

    MatCF gsf;			// Frequency response.
    MatCF hfNum;		// Filter numerator.
    MatCF hfDen;		// Filter denominator.

    // RotationEstimator memory arena management.
    MemoryArena *arenas;

    // Background thread backup.
    MatCF bk_hfNum;
    MatCF bk_hfDen;
    bool backgroundUpdateFlag;
};

}	// namespace CSRT

#endif	// CSRT_ROTATION_ESTIMATOR_H