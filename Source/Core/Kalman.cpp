//
// Kalman motion controller
//

#include "Kalman.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"

namespace CSRT {

#ifdef WITH_OPENCV

void Kalman2D::Initialize(const float *initialState) {
    filter.init(4, 2, 0, CV_32F);
    filter.transitionMatrix = cv::Mat::eye(4, 4, CV_32F);
    filter.transitionMatrix.at<float>(0, 2) = 1.0f;
    filter.transitionMatrix.at<float>(1, 3) = 1.0f;
    filter.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    filter.measurementMatrix.at<float>(0, 0) = 1.0f;
    filter.measurementMatrix.at<float>(1, 1) = 1.0f;
    filter.processNoiseCov = 0.16f * cv::Mat::eye(4, 4, CV_32F);
    filter.measurementNoiseCov = 1e-2f * cv::Mat::eye(2, 2, CV_32F);
    filter.errorCovPost = 0.01f * cv::Mat::eye(4, 4, CV_32F);
    filter.errorCovPre = 0.01f * cv::Mat::eye(4, 4, CV_32F);

    // set initial state
    filter.statePre = cv::Mat::zeros(4, 1, CV_32F);
    filter.statePost = cv::Mat::zeros(4, 1, CV_32F);
    memcpy(filter.statePre.data, initialState, sizeof(float) * 4);
    memcpy(filter.statePost.data, initialState, sizeof(float) * 4);
}

void Kalman2D::Predict(float *res) {
    cv::Mat m = filter.predict();
    memcpy(res, m.data, sizeof(float) * 4);
}

void Kalman2D::Correct(const float *measure, float *res) {
    cv::Mat measureM(2, 1, CV_32F);
    memcpy(measureM.data, measure, sizeof(float) * 2);
    cv::Mat m = filter.correct(measureM);
    memcpy(res, m.data, sizeof(float) * 4);
}

#endif

}   // namespace CSRT
