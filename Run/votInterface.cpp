#include "../Source/Core/Processor.h"
#include "../Source/Camera/Camera.h"
#include <memory>

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include "vot.h"

using namespace CSRT;

int main(int argc, char* argv[]) {
    StartSystem();

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
    params.NumHOGChannelsUsed = 31;
    params.PCACount = 0;
    params.UseNormalForSegment = true;
    params.UseNormalForDSST = true;
    params.UseChannelWeights = true;
    params.UseSegmentation = true;
    params.CellSize = 4;
    params.AdmmIterations = 4;
    params.Padding = 3.0f;
    params.TemplateSize = 200;
    params.GaussianSigma = 1.0f;
    params.WindowFunc = WindowType::Hann;
    params.ChebAttenuation = 45.0f;
    params.KaiserAlpha = 3.75f;
    params.WeightsLearnRate = 0.02f;
    params.FilterLearnRate = 0.02f;
    params.HistLearnRate = 0.04f;
    params.ScaleLearnRate = 0.025f;
    params.BackgroundRatio = 2.0f;
    params.HistogramBins = 16;
    params.PostRegularCount = 16;
    params.MaxSegmentArea = 1024.0f;
    params.ScaleCount = 33;
    params.ScaleSigma = 0.25f;
    params.ScaleMaxArea = 512.0f;
    params.ScaleStep = 1.02f;
    params.UpdateInterval = 0;
    params.UseScale = true;
    params.UseSmoother = true;

    VOT vot;
    std::vector<Bounds2i> bbs(1, Bounds2i());
    VOTRegion region = vot.region();
    bbs[0] = Bounds2i(region.get_x(), region.get_y(), region.get_width(), region.get_height());
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