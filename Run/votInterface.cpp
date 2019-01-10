#include "../Source/Core/Processor.h"
#include "../Source/Camera/Camera.h"
#include <memory>

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include "vot.h"

using namespace CSRT;

int main(int argc, char* argv[]) {
    StartSystem();

    std::vector<std::string> argues;
    for(int i = 1; i < argc; ++i) argues.push_back(argv[i]);

    float learnRates[3] = { 0.02f, 0.08f, 0.16f };
    TrackerParams params;
    params.UseHOG = true;
    params.UseCN = true;
    params.UseGRAY = true;
    params.UseRGB = false;
    params.UseDepthHOG = true;
    params.UseDepthGray = true;
    params.UseDepthNormal = true;
    params.UseDepthHON = true;
    params.UseDepthNormalHOG = true;
    params.NumHOGChannelsUsed = 18;
    params.PCACount = 0;
    if(argues.size() > 0) params.PCACount = std::stoi(argues[0]);
    params.UseNormalForSegment = true;
    params.UseNormalForDSST = true;
    params.UseChannelWeights = true;
    params.UseSegmentation = true;
    params.AdmmIterations = 3;
    params.Padding = 3.0f;
    params.TemplateSize = 200;
    //if(argues.size() > 2) params.TemplateSize = std::stof(argues[2]);
    params.GaussianSigma = 1.5f;
    params.WindowFunc = WindowType::Hann;
    params.ChebAttenuation = 45.0f;
    params.KaiserAlpha = 3.75f;
    params.WeightsLearnRates = &learnRates[0];
    params.FilterLearnRates = &learnRates[0];
    params.LearnRateNum = sizeof(learnRates) / sizeof(learnRates[0]);
    params.HistLearnRate = 0.04f;
    params.ScaleLearnRate = 0.025f;
    params.BackgroundRatio = 2.0f;
    //if(argues.size() > 1) params.BackgroundRatio = std::stof(argues[1]);
    params.HistogramBins = 16;
    params.PostRegularCount = 16;
    params.MaxSegmentArea = 1024.0f;
    params.ScaleCount = 33;
    params.ScaleSigma = 0.25f;
    params.ScaleMaxArea = 512.0f;
    params.ScaleStep = 1.02f;
    params.UpdateInterval = 0;
    //if(argues.size() > 0) params.UpdateInterval = std::stoi(argues[0]);
    params.UseScale = true;
    params.UseSmoother = false;
    params.UseFastScale = false;
    params.FailThreshold = 0.08f;
    //if(argues.size() > 0) params.FailThreshold = std::stof(argues[0]);

    VOT vot;
    std::vector<Bounds2f> bbs(1, Bounds2f());
    VOTRegion region = vot.region();
    bbs[0] = Bounds2f(region.get_x(), region.get_y(), region.get_width(), region.get_height());
    string path = vot.frame();
    Mat image(path);
    int rows = image.Rows();
    int cols = image.Cols();
    std::unique_ptr<Processor> pTracker(new Processor(params, 1, Vector2i(cols, rows), TrackMode::RGB));
    pTracker->Initialize(image, bbs);
    GImageMemoryArena.ResetImgLoadArena();

    while (true) {
        path = vot.frame();
        if (path.empty()) break;

        image = Mat(path);
        float confidence = 1.0f;
        pTracker->Update(image, bbs);
        GImageMemoryArena.ResetImgLoadArena();
        region.set_x(bbs[0].pMin.x);
        region.set_y(bbs[0].pMin.y);
        region.set_width(bbs[0].pMax.x - bbs[0].pMin.x);
        region.set_height(bbs[0].pMax.y - bbs[0].pMin.y);
        vot.report(region, confidence);
    }

    CloseSystem();
    return 0;
}