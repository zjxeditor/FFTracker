//
// Kalman motion controller
//

#include "Kalman.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"

namespace CSRT {

//
// Kalman filter implementation.
//

void KalmanFilter::Initialize(int dynamParams, int measureParams, int controlParams) {
    DP = dynamParams;
    MP = measureParams;
    CP = controlParams;
    if(DP <= 0 || MP <= 0)
        Critical("KalmanFilter::Initialize: invalid kalman filter configuration!");
    CP = std::max(CP, 0);
    float *pdata = nullptr;

    statePre.Reshape(DP, 1);
    statePost.Reshape(DP, 1);
    transitionMatrix.Reshape(DP, DP);
    pdata = transitionMatrix.Data();
    for(int i = 0; i <  DP; ++i) {
        *pdata = 1.0f;
        pdata += (DP + i);
    }
    measureMatrix.Reshape(MP, DP);
    if(CP > 0) controlMatrix.Reshape(DP, CP);
    else controlMatrix.Reshape(0, 0);

    processNoiseCov.Reshape(DP, DP);
    pdata = processNoiseCov.Data();
    for(int i = 0; i <  DP; ++i) {
        *pdata = 1.0f;
        pdata += (DP + i);
    }
    measureNoiseCov.Reshape(MP, MP);
    pdata = measureNoiseCov.Data();
    for(int i = 0; i <  MP; ++i) {
        *pdata = 1.0f;
        pdata += (MP + i);
    }
    errorCovPre.Reshape(DP, DP);
    errorCovPost.Reshape(DP, DP);
    gain.Reshape(DP, MP);

    temp1.Reshape(MP, DP);
    temp2.Reshape(MP, MP);
    temp3.Reshape(MP, DP);
    temp4.Reshape(MP, 1);
}

void KalmanFilter::Predict(CSRT::MatF &res, const CSRT::MatF &control) {
    if(control.Size() != 0 && (control.Rows() != CP || control.Cols() != 1))
        Critical("KalmanFilter::Predict: invalid control parameter.");

    Eigen::Map<Eigen::MatrixXf> egTransitionM(transitionMatrix.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egStatePreM(statePre.Data(), DP, 1);
    Eigen::Map<Eigen::MatrixXf> egStatePostM(statePost.Data(), DP, 1);
    Eigen::Map<Eigen::MatrixXf> egErrorCovPreM(errorCovPre.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egErrorCovPostM(errorCovPost.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egProcessNoiseCovM(processNoiseCov.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egControlM(controlMatrix.Data(), DP, CP);
    Eigen::Map<Eigen::MatrixXf> egControlInputM(control.Data(), CP, 1);

    // update the state: x'(k) = A*x(k)
    egStatePreM.noalias() = egTransitionM * egStatePostM;
    if(control.Size() != 0) {
        // x'(k) = x'(k) + B*u(k)
        egStatePreM.noalias() = egStatePreM + egControlM * egControlInputM;
    }

    // P'(k) = A*P(k)*At + Q
    egErrorCovPreM.noalias() = egTransitionM * egErrorCovPostM * egTransitionM.transpose() + egProcessNoiseCovM;

    // handle the case when there will be no measurement before the next predict.
    statePost = statePre;
    errorCovPost = errorCovPre;

    res = statePre;
}

void KalmanFilter::Correct(CSRT::MatF &res, const CSRT::MatF &measure) {
    if(measure.Rows() != MP || measure.Cols() != 1)
        Critical("KalmanFilter::Correct: invlaid measure parameter.");

    Eigen::Map<Eigen::MatrixXf> egTemp1M(temp1.Data(), MP, DP);
    Eigen::Map<Eigen::MatrixXf> egTemp2M(temp2.Data(), MP, MP);
    Eigen::Map<Eigen::MatrixXf> egTemp3M(temp3.Data(), MP, DP);
    Eigen::Map<Eigen::MatrixXf> egTemp4M(temp4.Data(), MP, 1);
    Eigen::Map<Eigen::MatrixXf> egMeasureM(measureMatrix.Data(), MP, DP);
    Eigen::Map<Eigen::MatrixXf> egStatePreM(statePre.Data(), DP, 1);
    Eigen::Map<Eigen::MatrixXf> egStatePostM(statePost.Data(), DP, 1);
    Eigen::Map<Eigen::MatrixXf> egErrorCovPreM(errorCovPre.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egErrorCovPostM(errorCovPost.Data(), DP, DP);
    Eigen::Map<Eigen::MatrixXf> egMeasureNoiseCovM(measureNoiseCov.Data(), MP, MP);
    Eigen::Map<Eigen::MatrixXf> egMeasureInputM(measure.Data(), MP, 1);
    Eigen::Map<Eigen::MatrixXf> egGainM(gain.Data(), DP, MP);

    // temp1 = H*P'(k)
    egTemp1M.noalias() = egMeasureM * egErrorCovPreM;

    // temp2 = temp1*Ht + R
    egTemp2M.noalias() = egTemp1M * egMeasureM.transpose() + egMeasureNoiseCovM;

    // temp3 = inv(temp2)*temp1 = Kt(k)
    // K(k) = temp3t
    egTemp3M.noalias() = egTemp2M.inverse() * egTemp1M;
    egGainM.noalias() = egTemp3M.transpose();

    //egGainM.noalias() = egErrorCovPreM * egMeasureM.transpose() * egTemp2M.inverse();

    // temp4 = z(k) - H*x'(k)
    egTemp4M.noalias() = egMeasureInputM - egMeasureM * egStatePreM;

    // x(k) = x'(k) + K(k)*temp5
    egStatePostM.noalias() = egStatePreM + egGainM * egTemp4M;

    // P(k) = P'(k) - K(k)*temp2
    egErrorCovPostM.noalias() = egErrorCovPreM - egGainM * egTemp1M;

    res = statePost;
}

//
// Kalman2D implementation.
//

void Kalman2D::Initialize(const float *initialState) {
    filter.Initialize(4, 2, 0);
    float *pdata = nullptr;

    // config system
    pdata = filter.transitionMatrix.Data();
    memset(pdata, 0, filter.transitionMatrix.Size() * sizeof(float));
    pdata[0] = pdata[5] = pdata[10] = pdata[15] = 1.0f;
    pdata[2] = pdata[7] = 1.0f;
    pdata = filter.measureMatrix.Data();
    memset(pdata, 0, filter.measureMatrix.Size() * sizeof(float));
    pdata[0] = pdata[5] = 1.0f;
    pdata = filter.processNoiseCov.Data();
    memset(pdata, 0, filter.processNoiseCov.Size() * sizeof(float));
    pdata[0] = pdata[5] = pdata[10] = pdata[15] = 0.16f;
    pdata = filter.measureNoiseCov.Data();
    memset(pdata, 0, filter.measureNoiseCov.Size() * sizeof(float));
    pdata[0] = pdata[3] = 1e-2f;
    pdata = filter.errorCovPost.Data();
    memset(pdata, 0, filter.errorCovPost.Size() * sizeof(float));
    pdata[0] = pdata[5] = pdata[10] = pdata[15] = 0.01f;
    pdata = filter.errorCovPre.Data();
    memset(pdata, 0, filter.errorCovPre.Size() * sizeof(float));
    pdata[0] = pdata[5] = pdata[10] = pdata[15] = 0.01f;

    // set initial state
    memcpy(filter.statePre.Data(), initialState, sizeof(float) * 4);
    memcpy(filter.statePost.Data(), initialState, sizeof(float) * 4);
}

void Kalman2D::Predict(float *res) {
    MatF resM;
    resM.ManageExternalData(res, 4, 1);
    filter.Predict(resM);
}

void Kalman2D::Correct(const float *measure, float *res) {
    MatF measureM, resM;
    measureM.ManageExternalData(const_cast<float*>(measure), 2, 1);
    resM.ManageExternalData(res, 4, 1);
    filter.Correct(resM, measureM);
}

}   // namespace CSRT
