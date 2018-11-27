#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "../Resampler/vnImagine.h"
#include "../Source/Utility/Mat.h"

int main() {
    CSRT::CreateLogger();

    CSRT::Mat image("/Users/jxzhang/Desktop/test1.jpg");
    int rows = image.Rows();
    int cols = image.Cols();
    float scaleRatio[1] = { 4.0f };
    VN_IMAGE_KERNEL_TYPE kernel[4] = {VN_IMAGE_KERNEL_NEAREST, VN_IMAGE_KERNEL_BILINEAR, VN_IMAGE_KERNEL_BICUBIC, VN_IMAGE_KERNEL_LANCZOS};
    std::string kernelName[4] = { "nearest", "bilinear", "bicubic", "lanczos" };

    CVImage source_image;
    if (VN_FAILED(vnCreateImage(VN_IMAGE_FORMAT_R8G8B8, cols, rows, &source_image))) {
        std::cout << "Error! Unable to create the source image." << std::endl;
        return -1;
    }
    memcpy(source_image.QueryData(), image.Data(), image.Size() * sizeof(uint8_t));

    CVImage resampled_image;
    CSRT::Mat resized;
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 1; ++j) {
            if (VN_FAILED(vnResizeImage(source_image, kernel[i], cols * scaleRatio[j], rows * scaleRatio[j], 0, &resampled_image))) {
                std::cout << "Error! Unable to resample the source image." << std::endl;
                return -1;
            }
            resized.Reshape(resampled_image.QueryHeight(), resampled_image.QueryWidth(), resampled_image.QueryChannelCount());
            memcpy(resized.Data(), resampled_image.QueryData(), resized.Size() * sizeof(uint8_t));
            CSRT::SaveToFile(kernelName[i] + "_" + std::to_string(scaleRatio[j]), resized);
        }
    }

    CSRT::ClearLogger();
    return 0;
}