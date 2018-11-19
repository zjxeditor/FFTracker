//
// Provide common utilities for camera operations.
//

#include "Camera.h"
#include "KinectService1.h"
#include "KinectService2.h"
#include "RealSense.h"

namespace CSRT {

std::unique_ptr<CameraService> CreateCameraService(CameraType type, bool dpPostProcessing, int dimReduce) {
#ifdef _WIN32
    switch(type) {
        case CameraType::RealSense:
            return std::unique_ptr<CameraService>(new RealSense(dpPostProcessing, dimReduce));
        case CameraType::KinectV1:
            return std::unique_ptr<CameraService>(new KinectV1());
        case CameraType::KinectV2:
            return std::unique_ptr<CameraService>(new KinectV2());
        default:
            throw std::runtime_error("CreateCameraService: unsupported camera service type.");
    }
#else
    switch(type) {
        case CameraType::RealSense:
            return std::unique_ptr<CameraService>(new RealSense(dpPostProcessing, dimReduce));
        default:
            throw std::runtime_error("CreateCameraService: unsupported camera service type.");
    }
#endif
}

}   // namespace CSRT
