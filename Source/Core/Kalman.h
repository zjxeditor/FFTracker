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

namespace CSRT {

class KalmanFilter {
public:
    KalmanFilter() : DP(0), MP(0), CP(0) {}

    // Initialize the internal data.
    void Initialize(int dynamParams, int measureParams, int controlParams = 0);

    // Prediction process. Output the predicted system state.
    void Predict(MatF &res, const MatF &control = MatF());

    // Correction process. Output the corrected system state.
    void Correct(MatF &res, const MatF &measure);

    // Internal matrix data.
    MatF statePre;          // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    MatF statePost;         // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    MatF transitionMatrix;  // state transition matrix (A)
    MatF controlMatrix;     // control matrix (B) (not used if there is no control)
    MatF measureMatrix;     // measurement matrix (H)
    MatF processNoiseCov;   // process noise covariance matrix (Q)
    MatF measureNoiseCov;   // measurement noise covariance matrix (R)
    MatF errorCovPre;       // prior error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)
    MatF errorCovPost;      // posterior error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
    MatF gain;              // Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)

private:
    // temporary matrices
    MatF temp1;
    MatF temp2;
    MatF temp3;
    MatF temp4;

    int DP, MP, CP;
};

class Kalman2D {
public:
    void Initialize(const float *initialState);

    // Predict process.
    void Predict(float *res);

    // Correct process.
    void Correct(const float *measure, float *res);

private:
    KalmanFilter filter;
};

}   // namespace CSRT

#endif //CSRT_KALMAN_H
