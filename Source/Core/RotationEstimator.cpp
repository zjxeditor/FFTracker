//
// Scale estimation using RotationEstimator method.
//

#include "RotationEstimator.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"
#include "Filter.h"

namespace CSRT {

void RotationEstimator::ResetArenas(bool background) const {
    if (!arenas) return;
    int len = MaxThreadIndex() - 1;
    if (background) {
        arenas[len].Reset();
    } else {
        for (int i = 0; i < len; ++i)
            arenas[i].Reset();
    }
}

void RotationEstimator::Initialize(const MatF &feats, const MatCF &gsfRow) {
    if (initialized) return;
    initialized = true;
    if (!arenas)
        arenas = new MemoryArena[MaxThreadIndex()];
    backgroundUpdateFlag = false;

    // Verify scale representation features.
    if(feats.Rows() == 0 || feats.Cols() != gsfRow.Cols()) {
        Critical("RotationEstimator::Initialize: invalid scale features.");
        return;
    }

    // Calculate the initial filter.
    gsfRow.Repeat(gsf, feats.Rows(), 1);
    MatCF srf(&arenas[ThreadIndex]);
    GFFT.FFT2Row(feats, srf);
    GFilter.MulSpectrums(gsf, srf, hfNum, true);
    MatCF hfDenAll(&arenas[ThreadIndex]);
    GFilter.MulSpectrums(srf, srf, hfDenAll, true);
    hfDenAll.Reduce(hfDen, ReduceMode::Sum, true);

    ResetArenas(false);
    ResetArenas(true);
}

void RotationEstimator::GetRotation(const MatF &feats, const std::vector<float> &rotationFactors, float currentRotation, float &newRotation) {
    if (!initialized) {
        Error("RotationEstimator::GetRotation: RotationEstimator is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("RotationEstimator::GetScale: invalid scale features.");
        return;
    }

    // Calculate reponse for the previoud filter.
    MatCF srf(&arenas[ThreadIndex]), tempf(&arenas[ThreadIndex]);
    GFFT.FFT2Row(feats, srf);
    GFilter.MulSpectrums(srf, hfNum, tempf, false);
    MatCF scaleRepresent(&arenas[ThreadIndex]), respf(&arenas[ThreadIndex]);
    tempf.Reduce(scaleRepresent, ReduceMode::Sum, true);
    GFilter.DivideComplexMatrices(scaleRepresent, hfDen, respf, ComplexF(0.01f, 0.01f));
    MatF resp(&arenas[ThreadIndex]);
    GFFT.FFTInv2Row(respf, resp, true);

    // Find max reponse location.
    float maxValue = 0.0f;
    Vector2i maxLoc;
    resp.MaxLoc(maxValue, maxLoc);

    newRotation = currentRotation - rotationFactors[maxLoc.x];
    if(std::abs(newRotation) < 1e-5f) newRotation = 0.0f;
    if(newRotation > 2.0f * Pi) newRotation -= 2.0f * Pi;
    if(newRotation < -2.0f * Pi) newRotation += 2.0f * Pi;
    ResetArenas(false);
}

void RotationEstimator::Update(const MatF& feats) {
    if (!initialized) {
        Error("RotationEstimator::Update: RotationEstimator is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("RotationEstimator::Update: invalid scale features.");
        return;
    }

    // Calculate new filter model.
    MatCF srf(&arenas[ThreadIndex]);
    GFFT.FFT2Row(feats, srf);
    MatCF newHfNum(&arenas[ThreadIndex]), newHfDen(&arenas[ThreadIndex]), newHfDenAll(&arenas[ThreadIndex]);
    GFilter.MulSpectrums(gsf, srf, newHfNum, true);
    GFilter.MulSpectrums(srf, srf, newHfDenAll, true);
    newHfDenAll.Reduce(newHfDen, ReduceMode::Sum, true);

    // Lerp by learn rate.
    hfNum.Lerp(newHfNum, rotationLearnRate);
    hfDen.Lerp(newHfDen, rotationLearnRate);

    ResetArenas(false);
}

void RotationEstimator::StartBackgroundUpdate() {
    if (backgroundUpdateFlag) return;
    bk_hfNum = hfNum;
    bk_hfDen = hfDen;
    backgroundUpdateFlag = true;
}

void RotationEstimator::FetchUpdateResult() {
    if (!backgroundUpdateFlag) return;
    hfNum = bk_hfNum;
    hfDen = bk_hfDen;
    backgroundUpdateFlag = false;
}

void RotationEstimator::BackgroundUpdate(const MatF& feats) {
    if (!initialized) {
        Error("RotationEstimator::BackgroundUpdate: RotationEstimator is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("RotationEstimator::BackgroundUpdate: invalid scale features.");
        return;
    }

    // Calculate new filter model.
    MatCF srf(&arenas[ThreadIndex]);
    GFFT.FFT2Row(feats, srf);
    MatCF newHfNum(&arenas[ThreadIndex]), newHfDen(&arenas[ThreadIndex]), newHfDenAll(&arenas[ThreadIndex]);
    GFilter.MulSpectrums(gsf, srf, newHfNum, true);
    GFilter.MulSpectrums(srf, srf, newHfDenAll, true);
    newHfDenAll.Reduce(newHfDen, ReduceMode::Sum, true);

    // Lerp by learn rate.
    bk_hfNum.Lerp(newHfNum, rotationLearnRate);
    bk_hfDen.Lerp(newHfDen, rotationLearnRate);

    ResetArenas(true);
}

}	// namespace CSRT