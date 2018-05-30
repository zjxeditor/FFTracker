//
// Export the CSRDCF tracker as a library.
//

#include "CSRTracker.h"

#include "../CSRT.h"
#include "../Utility/FFT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"
#include "../Utility/Memory.h"
#include "../Utility/Parallel.h"
#include "../Utility/StringPrint.h"
#include "../Core/DSST.h"
#include "../Core/FeaturesExtractor.h"
#include "../Core/Filter.h"
#include "../Core/Segment.h"
#include "../Core/Tracker.h"

namespace CSRT {

void SetThread(int count) {
    if(count <= 0)
        CSRT::SetThreadCount(NumSystemCores() + 1);
    else {
        if(count == 1) ++count;
        CSRT::SetThreadCount(count);
    }
}

void StartSystem(const char *logPath) {
    if (logPath == nullptr)
        CSRT::CreateLogger();
    else
        CSRT::CreateLogger(std::string(logPath));
    CSRT::ParallelInit();
    Eigen::setNbThreads(MaxThreadIndex());
    Eigen::initParallel();
    GFFT.Initialize();
    GImageMemoryArena.Initialize();
    GFeatsExtractor.Initialize();
    GFilter.Initialize();
    GSegment.Initialize();
}

void CloseSystem() {
    CSRT::ParallelCleanup();
    CSRT::Info("All work is done!");
    CSRT::ClearLogger();
}


CSRTrackerParams::CSRTrackerParams() {
    UseHOG = true;
    UseCN = true;
    UseGRAY = true;
    UseRGB = false;
    NumHOGChannelsUsed = 18;
    UsePCA = false;
    PCACount = 8;

    UseChannelWeights = true;
    UseSegmentation = true;

    AdmmIterations = 4;
    Padding = 3.0f;
    TemplateSize = 200;
    GaussianSigma = 1.0f;

    WindowFunc = CSRTWindowType::Hann;
    ChebAttenuation = 45.0f;
    KaiserAlpha = 3.75f;

    WeightsLearnRate = 0.02f;
    FilterLearnRate = 0.02f;
    HistLearnRate = 0.04f;
    ScaleLearnRate = 0.025f;

    BackgroundRatio = 2.0f;
    HistogramBins = 16;
    PostRegularCount = 10;
    MaxSegmentArea = 1024.0f;

    ScaleCount = 33;
    ScaleSigma = 0.25f;
    ScaleMaxArea = 512.0f;
    ScaleStep = 1.02f;

    UpdateInterval = 4;
    PeakRatio = 0.1f;
    PSRThreshold = 15.0f;
    UseScale = true;
}


CSRTracker::CSRTracker(int rows, int cols, float scale, const CSRT::CSRTrackerParams &trackerParams)
        : rowNum(rows), colNum(cols), scaleFactor(scale), draw(false), red(0), green(0), blue(0), alpha(0.0f) {
    TrackerParams params;
    memcpy(&params, &trackerParams, sizeof(TrackerParams));
    tracker = new Tracker(params);
    arena = new MemoryArena();
}

CSRTracker::~CSRTracker() {
    if (tracker) {
        delete tracker;
        tracker = nullptr;
    }
    if (arena) {
        delete arena;
        tracker = nullptr;
    }
}

void CSRTracker::Initialize(unsigned char *sourceData, int channels, const CSRT::Bounds &bb) {
    Mat image(arena);
    if (channels == 4) {
        image.Reshape(rowNum, colNum, 3, false);
        Rgba2Rgb(sourceData, image.Data());
    } else if (channels == 3) {
        image = Mat(sourceData, rowNum, colNum, 3, arena);
    } else {
        Critical("CSRTracker::Initialize: unsupported channel count.");
        return;
    }
    Mat scaledImage(arena);
    image.Resize(scaledImage, scaleFactor, scaleFactor);

    Bounds2i initBox;
    memcpy(&initBox, &bb, sizeof(Bounds2i));
    initBox *= scaleFactor;
    tracker->Initialize(scaledImage, initBox);
    arena->Reset();

    if(draw)
        DrawRect(sourceData, channels, bb);
}

bool CSRTracker::Update(unsigned char *sourceData, int channels, CSRT::Bounds &bb, float &score) {
    Mat image(arena);
    if (channels == 4) {
        image.Reshape(rowNum, colNum, 3, false);
        Rgba2Rgb(sourceData, image.Data());
    } else if (channels == 3) {
        image = Mat(sourceData, rowNum, colNum, 3, arena);
    } else {
        Critical("CSRTracker::Initialize: unsupported channel count.");
        return false;
    }
    Mat scaledImage(arena);
    image.Resize(scaledImage, scaleFactor, scaleFactor);

    Bounds2i outputBox;
    bool res = tracker->Update(scaledImage, outputBox, score);
    outputBox /= scaleFactor;
    memcpy(&bb, &outputBox, sizeof(Bounds2i));
    arena->Reset();

    if(draw)
        DrawRect(sourceData, channels, bb);
    return res;
}

void CSRTracker::SetReinitialize() {
    tracker->SetReinitialize();
}

void CSRTracker::SetDrawMode(bool enableDraw, unsigned char r, unsigned char g, unsigned char b, float a) {
    draw = enableDraw;
    red = r;
    green = g;
    blue = b;
    alpha = a;
}

void CSRTracker::Rgba2Rgb(unsigned char *srcData, unsigned char *dstData) {
    ParallelFor([&](int64_t y) {
        const unsigned char *ps = srcData + y * colNum * 4;
        unsigned char *pd = dstData + y * colNum * 3;
        for (int i = 0; i < colNum; ++i) {
            *(pd++) = *(ps++);
            *(pd++) = *(ps++);
            *(pd++) = *(ps++);
            ++ps;
        }
    }, rowNum, 32);
}

void CSRTracker::DrawRect(unsigned char *srcData, int channels, const CSRT::Bounds &bb) {
    if(bb.x0 < 0 || bb.y0 < 0 || bb.x1 > colNum || bb.y1 > rowNum) {
        Error("CSRTracker::DrawRect: invalid bounding box to draw on the image.");
        return;
    }
    int inc = 0;
    if(channels == 3) inc = 1;
    else if(channels == 4) inc = 2;
    else {
        Critical("CSRTracker::DrawRect: unsupported channel count.");
        return;
    }

    ParallelFor([&](int64_t y) {
        y += bb.y0;
        unsigned char *ps = srcData + y * colNum * channels + bb.x0 * channels;
        for (int i = bb.x0; i < bb.x1; ++i) {
            *ps =  (unsigned char)Clamp((int)Lerp(alpha, *ps, red), 0, 255);
            ++ps;
            *ps =  (unsigned char)Clamp((int)Lerp(alpha, *ps, green), 0, 255);
            ++ps;
            *ps =  (unsigned char)Clamp((int)Lerp(alpha, *ps, blue), 0, 255);
            ps += inc;
        }
    }, bb.y1 - bb.y0, 32);
}

}   // namespace CSRT