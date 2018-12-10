#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "../Source/Utility/Mat.h"
#include "../Source/Utility/Parallel.h"

int main() {
    CSRT::CreateLogger();
    CSRT::ParallelInit();
    CSRT::GImageMemoryArena.Initialize();

    CSRT::Mat image("/Users/jxzhang/Desktop/test1.jpg");
    int rows = image.Rows();
    int cols = image.Cols();
    float scaleRatio[4] = { 0.25f, 0.5f, 2.0f, 4.0f };

    CSRT::Mat resized;
    for(int i = 0; i < 4; ++i) {
        image.Resize(resized, scaleRatio[i], scaleRatio[i], CSRT::ResizeMode::Bilinear);
        CSRT::SaveToFile("bilinear_" + std::to_string(scaleRatio[i]), resized);
        image.Resize(resized, scaleRatio[i], scaleRatio[i], CSRT::ResizeMode::Nearest);
        CSRT::SaveToFile("nearest" + std::to_string(scaleRatio[i]), resized);
    }

    CSRT::ParallelCleanup();
    CSRT::ClearLogger();
    return 0;
}