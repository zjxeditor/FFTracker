//
// Scale estimation using fast DSST method.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_FDSST_H
#define CSRT_FDSST_H

#include "../CSRT.h"
#include "../Utility/Mat.h"

namespace CSRT {

class FDSST {
public:
    // DSST Public Methods
    FDSST(float learnRateOfScale) :
            initialized(false), scaleLearnRate(learnRateOfScale),
            arenas(nullptr), backgroundUpdateFlag(false) {}

    ~FDSST() {
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
    void GetScale(const MatF &feats, const std::vector<float> &scaleInterpFactors, const float &currentScale,
                  float minScaleFactor, float maxScaleFactor, float &newScale);

    // Update the DSST model. The feats matrix are extracted at target object current location.
    void Update(const MatF &feats);

    // Background update logic.
    void StartBackgroundUpdate();
    void FetchUpdateResult();
    void BackgroundUpdate(const MatF& feats);	// The feats matrix are extracted at target object current location.

private:
    void ResetArenas(bool background = false) const;

    void ResizeDFT(const MatCF& indft, MatCF& outdft, int desiredLength);

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

}   // namespace CSRT

#endif //CSRT_FDSST_H
