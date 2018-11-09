//
// Provide smooth filters.
//

#include "Smoother.h"

namespace CSRT {

//
// Holt Double Exponential Smoothing filter for filtering Joints provided by Microsoft.
//

//TransformSmoothParamter CommonSmoothParam(0.25f, 0.25f, 0.25f, 0.03f, 0.05f);
TransformSmoothParamter CommonSmoothParam(0.25f, 0.25f, 0.25f, 0.03f, 0.05f);
TransformSmoothParamter GestureSmoothParam(0.5f, 0.5f, 0.5f, 0.05f, 0.04f);
TransformSmoothParamter MenuSmoothParam(0.5f, 0.1f, 0.5f, 0.1f, 0.1f);
TransformSmoothParamter StationSmoothParam(0.7f, 0.3f, 1.0f, 1.0f, 1.0f);

inline bool PointPositionIsValid(const float point) {
    return point >= 0.0f;
}

inline bool PointPositionIsValid(const Vector2f &point) {
    return (point.x >= 0.0f && point.y >= 0.0f);
}

float FilterDoubleExponential1D::Update(const float point) {
    float vRawPosition = point;
    float vFilteredPosition, vTrend, vPredictedPosition;

    float vPrevRawPosition = history.vRawPosition;
    float vPrevFilteredPosition = history.vFilteredPosition;
    float vPrevTrend = history.vTrend;

    // If joint is invalid, reset the filter
    bool bJointIsValid = PointPositionIsValid(vRawPosition);
    if (!bJointIsValid)
        history.mFrameCount = 0;

    float vDiff;
    float fDiff;

    // Initial start values
    if (history.mFrameCount == 0) {
        vFilteredPosition = vRawPosition;
        vTrend = 0.0f;
        history.mFrameCount++;
    } else if (history.mFrameCount == 1) {
        vFilteredPosition = (vRawPosition + vPrevRawPosition) * 0.5f;
        vDiff = vFilteredPosition - vPrevFilteredPosition;
        vTrend = vDiff * filterParameter.fCorrection + vPrevTrend * (1.0f - filterParameter.fCorrection);
        history.mFrameCount++;
    } else {
        // First apply jitter filter
        vDiff = vRawPosition - vPrevFilteredPosition;
        fDiff = std::abs(vDiff);

        if (fDiff <= filterParameter.fJitterRadius)
            vFilteredPosition = vRawPosition * (fDiff / filterParameter.fJitterRadius) +
                                vPrevFilteredPosition * (1.0f - fDiff / filterParameter.fJitterRadius);
        else
            vFilteredPosition = vRawPosition;

        // Now the double exponential smoothing filter
        vFilteredPosition = vFilteredPosition * (1 - filterParameter.fSmoothing) +
                            (vPrevFilteredPosition + vPrevTrend) * filterParameter.fSmoothing;
        vDiff = vFilteredPosition - vPrevFilteredPosition;
        vTrend = vDiff * filterParameter.fCorrection + vPrevTrend * (1.0f - filterParameter.fCorrection);
    }

    // Predict into the future to reduce latency
    vPredictedPosition = vFilteredPosition + vTrend * filterParameter.fPrediction;

    // Check that we are not too far away from raw data
    vDiff = vPredictedPosition - vRawPosition;
    fDiff = std::abs(vDiff);

    if (fDiff > filterParameter.fMaxDeviationRadius)
        vPredictedPosition = vPredictedPosition * (filterParameter.fMaxDeviationRadius / fDiff) +
                             vRawPosition * (1.0f - filterParameter.fMaxDeviationRadius / fDiff);

    // Save the data from this frame
    history.vRawPosition = vRawPosition;
    history.vFilteredPosition = vFilteredPosition;
    history.vTrend = vTrend;

    // Output the data
    return vPredictedPosition;
}

Vector2f FilterDoubleExponential2D::Update(const CSRT::Vector2f &point) {
    Vector2f vRawPosition = point;
    Vector2f vFilteredPosition, vTrend, vPredictedPosition;

    Vector2f vPrevRawPosition = history.vRawPosition;
    Vector2f vPrevFilteredPosition = history.vFilteredPosition;
    Vector2f vPrevTrend = history.vTrend;

    // If joint is invalid, reset the filter
    bool bJointIsValid = PointPositionIsValid(vRawPosition);
    if (!bJointIsValid)
        history.mFrameCount = 0;

    Vector2f vDiff;
    float fDiff;

    // Initial start values
    if (history.mFrameCount == 0) {
        vFilteredPosition = vRawPosition;
        vTrend = Vector2f();
        history.mFrameCount++;
    } else if (history.mFrameCount == 1) {
        vFilteredPosition = (vRawPosition + vPrevRawPosition) * 0.5f;
        vDiff = vFilteredPosition - vPrevFilteredPosition;
        vTrend = vDiff * filterParameter.fCorrection + vPrevTrend * (1.0f - filterParameter.fCorrection);
        history.mFrameCount++;
    } else {
        // First apply jitter filter
        vDiff = vRawPosition - vPrevFilteredPosition;
        fDiff = vDiff.Length();

        if (fDiff <= filterParameter.fJitterRadius)
            vFilteredPosition = vRawPosition * (fDiff / filterParameter.fJitterRadius) +
                                vPrevFilteredPosition * (1.0f - fDiff / filterParameter.fJitterRadius);
        else
            vFilteredPosition = vRawPosition;

        // Now the double exponential smoothing filter
        vFilteredPosition = vFilteredPosition * (1 - filterParameter.fSmoothing) +
                            (vPrevFilteredPosition + vPrevTrend) * filterParameter.fSmoothing;
        vDiff = vFilteredPosition - vPrevFilteredPosition;
        vTrend = vDiff * filterParameter.fCorrection + vPrevTrend * (1.0f - filterParameter.fCorrection);
    }

    // Predict into the future to reduce latency
    vPredictedPosition = vFilteredPosition + vTrend * filterParameter.fPrediction;

    // Check that we are not too far away from raw data
    vDiff = vPredictedPosition - vRawPosition;
    fDiff = vDiff.Length();

    if (fDiff > filterParameter.fMaxDeviationRadius)
        vPredictedPosition = vPredictedPosition * (filterParameter.fMaxDeviationRadius / fDiff) +
                             vRawPosition * (1.0f - filterParameter.fMaxDeviationRadius / fDiff);

    // Save the data from this frame
    history.vRawPosition = vRawPosition;
    history.vFilteredPosition = vFilteredPosition;
    history.vTrend = vTrend;

    // Output the data
    return vPredictedPosition;
}

}   // namespace CSRT