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

void StartSystem(const char *logPath) {
    if(logPath == nullptr)
        CSRT::CreateLogger();
    else
        CSRT::CreateLogger(std::string(logPath));
    CSRT::SetThreadCount(NumSystemCores() + 1);
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
}


CSRTracker::CSRTracker(int rows, int cols, const CSRT::CSRTrackerParams &trackerParams) : rowNum(rows), colsNum(cols) {
    TrackerParams params;
    memcpy(&params, &trackerParams, sizeof(TrackerParams));
    tracker = new Tracker(params);
    arena = new MemoryArena();
}

CSRTracker::~CSRTracker() {
    if(tracker) {
        delete tracker;
        tracker = nullptr;
    }
    if(arena) {
        delete arena;
        tracker = nullptr;
    }
}

void CSRTracker::Initialize(const unsigned char *sourceData, const CSRT::Bounds &bb) {
    Mat image(sourceData, rowNum, colsNum, 3, arena);
    Bounds2i initBox;
    memcpy(&initBox, &bb, sizeof(Bounds2i));
    tracker->Initialize(image, initBox);
    arena->Reset();
}

bool CSRTracker::Update(const unsigned char *sourceData, CSRT::Bounds &bb, float &score) {
    Mat image(sourceData, rowNum, colsNum, 3, arena);
    Bounds2i outputBox;
    bool res = tracker->Update(image, outputBox, score);
    memcpy(&bb, &outputBox, sizeof(Bounds2i));
    arena->Reset();
    return res;
}

void CSRTracker::SetReinitialize() {
    tracker->SetReinitialize();
}

}   // namespace CSRT