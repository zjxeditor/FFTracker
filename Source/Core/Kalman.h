//
// Kalman motion controller
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_KALMAN_H
#define CSRT_KALMAN_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace CSRT {

class Kalman2D {
public:
    void Initialize(const float *initialState);

    void Predict(float *res);

    void Correct(const float *measure, float *res);
private:

#ifdef WITH_OPENCV
    cv::KalmanFilter filter;
#endif
};

}   // namespace CSRT

#endif //CSRT_KALMAN_H
