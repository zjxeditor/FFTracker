#include "../Source/Core/Processor.h"
#include "../Source/Camera/Camera.h"
#include "../Source/Core/Filter.h"
#include <opencv2/opencv.hpp>
#include <iostream>

inline int modul(int a, int b)
{
    // function calculates the module of two numbers and it takes into account also negative numbers
    return ((a % b) + b) % b;
}

cv::Mat circshift(cv::Mat matrix, int dx, int dy)
{
    cv::Mat matrix_out = matrix.clone();
    int idx_y = 0;
    int idx_x = 0;
    for(int i=0; i<matrix.rows; i++) {
        for(int j=0; j<matrix.cols; j++) {
            idx_y = modul(i+dy+1, matrix.rows);
            idx_x = modul(j+dx+1, matrix.cols);
            matrix_out.at<float>(idx_y, idx_x) = matrix.at<float>(i,j);
        }
    }
    return matrix_out;
}

cv::Mat gaussian_shaped_labels(const float sigma, const int w, const int h)
{
    // create 2D Gaussian peak, convert to Fourier space and stores it into the yf
    cv::Mat y = cv::Mat::zeros(h, w, CV_32F);
    float w2 = static_cast<float>(cvFloor(w / 2));
    float h2 = static_cast<float>(cvFloor(h / 2));

    // calculate for each pixel separatelly
    for(int i=0; i<y.rows; i++) {
        for(int j=0; j<y.cols; j++) {
            y.at<float>(i,j) = (float)exp((-0.5 / pow(sigma, 2)) * (pow((i+1-h2), 2) + pow((j+1-w2), 2)));
        }
    }
    // wrap-arround with the circulat shifting
    y = circshift(y, -cvFloor(y.cols / 2), -cvFloor(y.rows / 2));
    cv::Mat yf;
    dft(y, yf, cv::DFT_COMPLEX_OUTPUT);
    return yf;
}

int main() {
    CSRT::StartSystem();

    CSRT::Vector2i size(6, 6);
    CSRT::MatCF res;
    CSRT::GFilter.GaussianShapedLabels(res, 1.0f, size.x, size.y);
    CSRT::Info("csrt:");
    CSRT::PrintMat(res);

    cv::Mat cvRes = gaussian_shaped_labels(1.0f, size.x, size.y);
    CSRT::Info("cv:");
    std::cout << cvRes <<std::endl;

    CSRT::CloseSystem();
    return 0;
}