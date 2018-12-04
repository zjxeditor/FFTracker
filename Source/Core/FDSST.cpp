//
// Scale estimation using fast FDSST method.
//


#include "FDSST.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"
#include "Filter.h"

namespace CSRT {

void FDSST::ResetArenas(bool background) const {
    if (!arenas) return;
    int len = MaxThreadIndex() - 1;
    if (background) {
        arenas[len].Reset();
    } else {
        for (int i = 0; i < len; ++i)
            arenas[i].Reset();
    }
}

void FDSST::ResizeDFT(const MatCF& indft, MatCF& outdft, int desiredLength) {
    int len = indft.Cols();
    int minsz = std::min(len, desiredLength);
    float scaling = (float)desiredLength/len;

    outdft.Reshape(1, desiredLength, true);
    int mids = (int)std::ceil(minsz/2.0f);
    int mide = (int)std::floor((minsz-1)/2.0f) - 1;

    const ComplexF *psrc = indft.Data();
    ComplexF *pdest = outdft.Data();
    for(int i = 0; i < mids; ++i) {
        pdest[i] = psrc[i] * scaling;
    }
    int s = len - 1;
    int t = desiredLength - 1;
    for(int i = 0; i <= mide; ++i) {
        pdest[t] = psrc[s] * scaling;
        --t;
        --s;
    }
}

void FDSST::Initialize(const MatF &feats, const MatCF &gsfRow) {
    if (initialized) return;
    initialized = true;
    if (!arenas)
        arenas = new MemoryArena[MaxThreadIndex()];
    backgroundUpdateFlag = false;

    // Verify scale representation features.
    if(feats.Rows() == 0 || feats.Cols() != gsfRow.Cols()) {
        Critical("DSST::Initialize: invalid scale features.");
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

void FDSST::GetScale(const MatF &feats, const std::vector<float> &scaleInterpFactors, const float &currentScale,
                    float minScaleFactor, float maxScaleFactor, float &newScale) {
    if (!initialized) {
        Error("FDSST::GetScale: FDSST is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("FDSST::GetScale: invalid scale features.");
        return;
    }

    // Calculate reponse for the previoud filter.
    MatCF srf(&arenas[ThreadIndex]), tempf(&arenas[ThreadIndex]);
    GFFT.FFT2Row(feats, srf);
    GFilter.MulSpectrums(srf, hfNum, tempf, false);
    MatCF scaleRepresent(&arenas[ThreadIndex]), respf(&arenas[ThreadIndex]);
    tempf.Reduce(scaleRepresent, ReduceMode::Sum, true);
    GFilter.DivideComplexMatrices(scaleRepresent, hfDen, respf, ComplexF(0.01f, 0.01f));

    // Do interpolation
    MatCF interpf(&arenas[ThreadIndex]);
    ResizeDFT(respf, interpf, scaleInterpFactors.size());
    MatF resp(&arenas[ThreadIndex]);
    GFFT.FFTInv2Row(interpf, resp, true);

//    MatF resp(&arenas[ThreadIndex]);
//    GFFT.FFTInv2Row(respf, resp, true);

    // Find max reponse location.
    float maxValue = 0.0f;
    Vector2i maxLoc;
    resp.MaxLoc(maxValue, maxLoc);

    // Update the current scale factor
    newScale = Clamp(currentScale * scaleInterpFactors[maxLoc.x], minScaleFactor, maxScaleFactor);
    ResetArenas(false);
}

void FDSST::Update(const MatF& feats) {
    if (!initialized) {
        Error("DSST::Update: DSST is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("DSST::Update: invalid scale features.");
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
    hfNum.Lerp(newHfNum, scaleLearnRate);
    hfDen.Lerp(newHfDen, scaleLearnRate);

    ResetArenas(false);
}

void FDSST::StartBackgroundUpdate() {
    if (backgroundUpdateFlag) return;
    bk_hfNum = hfNum;
    bk_hfDen = hfDen;
    backgroundUpdateFlag = true;
}

void FDSST::FetchUpdateResult() {
    if (!backgroundUpdateFlag) return;
    hfNum = bk_hfNum;
    hfDen = bk_hfDen;
    backgroundUpdateFlag = false;
}

void FDSST::BackgroundUpdate(const MatF& feats) {
    if (!initialized) {
        Error("DSST::BackgroundUpdate: DSST is not initialized.");
        return;
    }

    // Verify scale representation features.
    if (feats.Rows() == 0 || feats.Cols() != gsf.Cols()) {
        Critical("DSST::BackgroundUpdate: invalid scale features.");
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
    bk_hfNum.Lerp(newHfNum, scaleLearnRate);
    bk_hfDen.Lerp(newHfDen, scaleLearnRate);

    ResetArenas(true);
}

}   // namespace CSRT