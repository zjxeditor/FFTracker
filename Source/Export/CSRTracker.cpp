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
#include "../Core/InfoProvider.h"
#include "../Core/Processor.h"

namespace CSRT {

void StartTheSystem(int count, const char *logPath) {
    StartSystem(count, logPath);
}

void CloseTheSystem() {
    CloseSystem();
}

bool LoadImage(const char *path, int &width, int &height, int &channels, unsigned char **data) {
    if(path == nullptr) return false;
    Mat img(path);
    width = img.Cols();
    height = img.Rows();
    channels = img.Channels();
    *data = img.Data();
    if (*data == nullptr) return false;
    return true;
}

void FinishLoadImage() {
    GImageMemoryArena.ResetImgLoadArena();
}

void Info(const char *message) {
    Info(std::string(message));
}

void Warning(const char *message) {
    Warning(std::string(message));
}

void Error(const char *message) {
    Error(std::string(message));
}

void Critical(const char *message) {
    Critical(std::string(message));
}

float CSRTrackerParams::DefaultLearnRates[3] = { 0.02f, 0.08f, 0.16f };

CSRTrackerParams::CSRTrackerParams() {
    UseHOG = true;
    UseCN = true;
    UseGRAY = true;
    UseRGB = false;
    UseDepthHOG = false;
    UseDepthGray = true;
    UseDepthNormal = true;
    UseDepthHON = true;
    UseDepthNormalHOG = false;
    NumHOGChannelsUsed = 18;
    PCACount = 0;

    UseNormalForSegment = true;
    UseNormalForDSST = false;
    UseChannelWeights = true;
    UseSegmentation = true;

    AdmmIterations = 4;
    Padding = 3.0f;
    TemplateSize = 200.0f;
    GaussianSigma = 1.0f;

    WindowFunc = CSRTWindowType::Hann;
    ChebAttenuation = 45.0f;
    KaiserAlpha = 3.75f;

    WeightsLearnRates = &DefaultLearnRates[0];
    FilterLearnRates = &DefaultLearnRates[0];
    LearnRateNum = 3;
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
    UseScale = true;
    FailThreshold = 0.08f;
}

CSRTracker::CSRTracker(int rows, int cols, int targetCount, CSRTrackMode mode, const CSRT::CSRTrackerParams &trackerParams)
        : rowNum(rows), colNum(cols), targets(targetCount) {
    TrackerParams params;
    memcpy(&params, &trackerParams, sizeof(TrackerParams));
    Vector2i moveSize(cols, rows);
    processor = new Processor(params, targetCount, moveSize, (TrackMode)mode);
    arena = new MemoryArena();
}

CSRTracker::~CSRTracker() {
    if (processor) {
        delete processor;
        processor = nullptr;
    }
    if (arena) {
        delete arena;
        arena = nullptr;
    }
}

void CSRTracker::Initialize(unsigned char *sourceData, int channels, const CSRT::Bounds *bbs) {
    Mat image(arena);
    if (channels == 4) {
        image.Reshape(rowNum, colNum, 3, false);
        Rgba2Rgb(sourceData, image.Data());
    } else if (channels == 3) {
        image.ManageExternalData(sourceData, rowNum, colNum, 3);
    } else {
        Critical("CSRTracker::Initialize: unsupported channel count.");
        return;
    }
    std::vector<Bounds2f> initBoxes(targets);
    memcpy(&initBoxes[0], bbs, targets * sizeof(Bounds2f));

    processor->Initialize(image, initBoxes);
    arena->Reset();
}

void CSRTracker::Initialize(float *depthData, float *normalData, const CSRT::Bounds *bbs) {
    MatF depthM, normalM;
    depthM.ManageExternalData(depthData, rowNum, colNum);
    if(normalData != nullptr) normalM.ManageExternalData(normalData, rowNum, colNum * 2);
    std::vector<Bounds2f> initBoxes(targets);
    memcpy(&initBoxes[0], bbs, targets * sizeof(Bounds2f));

    processor->Initialize(depthM, normalM, initBoxes);
    arena->Reset();
}

bool CSRTracker::Update(unsigned char *sourceData, int channels, CSRT::Bounds *bbs) {
    Mat image(arena);
    if (channels == 4) {
        image.Reshape(rowNum, colNum, 3, false);
        Rgba2Rgb(sourceData, image.Data());
    } else if (channels == 3) {
        image.ManageExternalData(sourceData, rowNum, colNum, 3);
    } else {
        Critical("CSRTracker::Initialize: unsupported channel count.");
        return false;
    }

    std::vector<Bounds2f> outputBoxes;
    processor->Update(image, outputBoxes);
    memcpy(bbs, &outputBoxes[0], targets * sizeof(Bounds2f));
    arena->Reset();

    return true;
}

bool CSRTracker::Update(float *depthData, float *normalData, CSRT::Bounds *bbs) {
    MatF depthM, normalM;
    depthM.ManageExternalData(depthData, rowNum, colNum);
    if(normalData != nullptr) normalM.ManageExternalData(normalData, rowNum, colNum * 2);

    std::vector<Bounds2f> outputBoxes;
    processor->Update(depthM, normalM, outputBoxes);
    memcpy(bbs, &outputBoxes[0], targets * sizeof(Bounds2f));
    arena->Reset();

    return true;
}

void CSRTracker::SetReinitialize() {
    processor->SetReinitialize();
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

}   // namespace CSRT