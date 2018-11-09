//
// Provide smooth filters.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_SMOOTHER_H
#define CSRT_SMOOTHER_H

#include "../CSRT.h"
#include "Geometry.h"

namespace CSRT {

//
// A very simple Kalman filter. Note it is only support a single double data. You need verious
// filters to process 3D position data for instance.
//

class SimpleKalmanFilter
{
public:
    float Update(float measurement)
    {
        measurementUpdate();
        float result = X + (measurement - X) * K;
        X = result;
        return result;
    }

private:
    void measurementUpdate()
    {
        K = (P + Q) / (P + Q + R);
        P = R * (P + Q) / (R + P + Q);
    }

private:
    float Q = 0.000001f;
    float R = 0.01f;	// test 0.0001
    float P = 1.0f, X = 0.0f, K;
};

//
// The following filter is a Holt Double Exponential Smoothing filter for filtering Joints
// referenced to Microsoft.
//

struct TransformSmoothParamter
{
    TransformSmoothParamter() : fSmoothing(0.25f), fCorrection(0.25f), fPrediction(0.25f), fJitterRadius(0.03f), fMaxDeviationRadius(0.05f) {}
    TransformSmoothParamter(float smoothing, float correction, float prediction, float jitterRadius, float maxDeviationRadius)
            : fSmoothing(smoothing), fCorrection(correction), fPrediction(prediction), fJitterRadius(jitterRadius), fMaxDeviationRadius(maxDeviationRadius) {}

    float fSmoothing;             // [0..1], lower values closer to raw data. Will lag when too high.
    float fCorrection;            // [0..1], lower values slower to correct towards the raw data. Can make things springy.
    float fPrediction;            // [0..n], the number of frames to predict into the future. Can over shoot when too high.
    float fJitterRadius;          // The radius in meters for jitter reduction. Can do too much smoothing when too high.
    float fMaxDeviationRadius;    // The maximum radius in meters that filtered positions are allowed to deviate from raw data. Can snap back to noisy data when too high.
};

extern TransformSmoothParamter CommonSmoothParam;	// Good for common movement tracking.
extern TransformSmoothParamter GestureSmoothParam;	// Good for gesture recognition.
extern TransformSmoothParamter MenuSmoothParam;		// Good for menu operation.
extern TransformSmoothParamter StationSmoothParam;	// Good for very smooth situation.

//
// Holt Double Exponential Smoothing filter
//

struct FilterDoubleExponentialData1D
{
    FilterDoubleExponentialData1D() {
        vRawPosition = 0.0f;
        vFilteredPosition = 0.0f;
        vTrend = 0.0f;
        mFrameCount = 0;
    }

    float vRawPosition;
    float vFilteredPosition;
    float vTrend;
    int mFrameCount;
};

struct FilterDoubleExponentialData2D
{
    FilterDoubleExponentialData2D() { mFrameCount = 0; }

    Vector2f vRawPosition;
    Vector2f vFilteredPosition;
    Vector2f vTrend;
    int mFrameCount;
};

class FilterDoubleExponential1D
{
public:
    FilterDoubleExponential1D() { Init(TransformSmoothParamter()); }

    void Init(const TransformSmoothParamter &param)
    {
        filterParameter = param;
        filterParameter.fJitterRadius = std::max(0.0001f, filterParameter.fJitterRadius);
        history = FilterDoubleExponentialData1D();
    }

    float Update(const float point);

private:
    FilterDoubleExponentialData1D history;
    TransformSmoothParamter filterParameter;
};

class FilterDoubleExponential2D
{
public:
    FilterDoubleExponential2D() { Init(TransformSmoothParamter()); }

    void Init(const TransformSmoothParamter &param)
    {
        filterParameter = param;
        filterParameter.fJitterRadius = std::max(0.0001f, filterParameter.fJitterRadius);
        history = FilterDoubleExponentialData2D();
    }

    Vector2f Update(const Vector2f& point);

private:
    FilterDoubleExponentialData2D history;
    TransformSmoothParamter filterParameter;
};


}   // namespace CSRT

#endif //CSRT_SMOOTHER_H
