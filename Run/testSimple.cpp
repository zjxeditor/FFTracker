#include "../Source/Core/Processor.h"
#include "../Source/Camera/Camera.h"
#include "../Source/Core/Filter.h"
#include <iostream>
#include <fstream>

using namespace CSRT;

void RotateBounds(Bounds2f& bb, float radians) {
    float s = sin(radians);
    float c = cos(radians);
    float cx = (bb.pMin.x + bb.pMax.x) / 2.0f;
    float cy = (bb.pMin.y + bb.pMax.y) / 2.0f;
    float l = std::numeric_limits<float>::max();
    float r = std::numeric_limits<float>::min();
    float t = std::numeric_limits<float>::max();
    float b = std::numeric_limits<float>::min();
    float nx, ny;

    float xs[4] = {bb.pMin.x, bb.pMax.x, bb.pMax.x, bb.pMin.x};
    float ys[4] = {bb.pMin.y, bb.pMin.y, bb.pMax.y, bb.pMax.y};
    for(int i = 0; i < 4; ++i) {
        nx = cx+(xs[i]-cx)*c+(ys[i]-cy)*s;
        ny = cy-(xs[i]-cx)*s+(ys[i]-cy)*c;
        if(nx < l) l = nx;
        if(nx > r) r = nx;
        if(ny < t) t = ny;
        if(ny > b) b = ny;
    }

    bb.pMin.x = l;
    bb.pMin.y = t;
    bb.pMax.x = r;
    bb.pMax.y = b;
}


int main() {
    StartSystem();

//    Mat img("/Users/jxzhang/Desktop/test.jpg");
//    Mat roted;
//
//    int rotationCount = 8;
//    std::vector<float> rotationFactors(rotationCount);
//    float delta = 2.0f * Pi / rotationCount;
//    float rr = 0.0f;
//    for(int i = 0; i <rotationCount; ++i) {
//        rotationFactors[i] = rr;
//        rr += delta;
//    }
//
//    for(int i = 0; i < rotationCount; ++i) {
//        img.Rotate(roted, rotationFactors[i]);
//        SaveToFile(StringPrintf("rot%d", i), roted);
//    }

//    Bounds2f bb;
//    bb.pMin.x = 0.0f;
//    bb.pMin.y = 0.0f;
//    bb.pMax.x = 4.0f;
//    bb.pMax.y = 4.0f;
//
//    int rotationCount = 8;
//    std::vector<float> rotationFactors(rotationCount);
//    float delta = 2.0f * Pi / rotationCount;
//    float rr = 0.0f;
//    for(int i = 0; i <rotationCount; ++i) {
//        rotationFactors[i] = rr;
//        rr += delta;
//    }
//
//    for(int i = 0; i < rotationCount; ++i) {
//        Bounds2f rbb = bb;
//        RotateBounds(rbb, rotationFactors[i]);
//        std::cout << rbb << std::endl;
//    }

    CloseSystem();
    return 0;
}