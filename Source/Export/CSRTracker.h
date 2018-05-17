//
// Export the CSRDCF tracker as a library.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_CSRTRACKER_H
#define CSRT_CSRTRACKER_H

#ifdef _WIN32
#ifdef CSRT_EXPORT
#define CSRT_API __declspec(dllexport)
#else
#define CSRT_API __declspec(dllimport)
#endif
#else
#define CSRT_API
#endif

namespace CSRT {

//
// Global Functions
//

extern CSRT_API void StartSystem(const char *logPath = nullptr);
extern CSRT_API void CloseSystem();


//
// Global Structs
//

enum class CSRTWindowType {
    Hann = 0,
    Cheb = 1,
    Kaiser = 2
};

struct CSRT_API CSRTrackerParams {
    // Default constructor.
    CSRTrackerParams();

    // Feature related parameters.
    bool UseHOG;
    bool UseCN;
    bool UseGRAY;
    bool UseRGB;
    int NumHOGChannelsUsed;
    bool UsePCA;	// PCA features.
    int PCACount;	// Feature PCA count.
    //float HogOrientations;
    //float HogClip;

    bool UseChannelWeights;	// Use different weights for each channel.
    bool UseSegmentation;	// Use segmentation for spatial constraint.

    int AdmmIterations;		// Iteration number for optimized filter solver.
    float Padding;			// Padding used to calculate template size. Affect how much area to search.
    int TemplateSize;		// Specify the target template size.
    float GaussianSigma;	// Guassian lable sigma factor.

    CSRTWindowType WindowFunc;	// Filter window function type.
    float ChebAttenuation;	// Attenuation when use Cheb window.
    float KaiserAlpha;		// Alpha value when use Kaiser window.

    float WeightsLearnRate;	// Filter weights learn rate.
    float FilterLearnRate;	// Filter learn rate.
    float HistLearnRate;	// Histogram model learn rate.
    float ScaleLearnRate;	// DSST learn rate.

    float BackgroundRatio;	// Background extend ratio.
    int HistogramBins;		// Bins number for the hisogram extraction.
    int PostRegularCount;	// Iteration count for post regularization count in segment process.
    float MaxSegmentArea;	// Controls the max area used for segment probability computation.

    int ScaleCount;			// Scale numbers for DSST.
    float ScaleSigma;		// DSST scale sigma factor.
    float ScaleMaxArea;		// Max model area for DSST.
    float ScaleStep;		// Scale step for DSST.

    int UpdateInterval;		// Update frame interval. Set to 0 or negative to disable background update mode.
    float PeakRatio;		// Specify the peak occupation ratio to calculate PSR. PeakSize = ResponseSize * PeakRatio * 2 + 1.
    float PSRThreshold;		// PSR threshold used for failure detection. 20.0 ~ 60.0 is indicates strong peaks.
};

struct CSRT_API Bounds {
    Bounds() : x0(0), y0(0), x1(0), y1(0) {}
    Bounds(int xx0, int yy0, int w, int h) : x0(xx0), y0(yy0), x1(xx0 + w), y1(yy0 + h) {}

    int x0;
    int y0;
    int x1;
    int y1;
};


//
// Global Classes
//

class Tracker;
class MemoryArena;

class CSRT_API CSRTracker {
public:
    CSRTracker(int rows, int cols) : CSRTracker(rows, cols, CSRTrackerParams()) {}
    CSRTracker(int rows, int cols, const CSRTrackerParams &trackerParams);
    ~CSRTracker();

    // Input image data is continuous unsigned char memory block, in RGB interleaved format.
    void Initialize(unsigned char *sourceData, int channels, const Bounds &bb);
    bool Update(unsigned char  *sourceData, int channels, Bounds &bb, float &score);
    void SetReinitialize();
    void SetDrawMode(bool enableDraw, unsigned char r, unsigned char g, unsigned char b, float a);

private:
    void Rgba2Rgb(unsigned char *srcData, unsigned char *dstData);
    void DrawRect(unsigned char *srcData, int channels, const Bounds &bb);

    int rowNum;
    int colNum;
    Tracker *tracker;
    MemoryArena *arena;

    bool draw;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    float alpha;
};

}   // namespace CSRT

#endif //CSRT_CSRTRACKER_H
