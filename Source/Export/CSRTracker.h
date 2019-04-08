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
#elif CSRT_IMPORT
#define CSRT_API __declspec(dllimport)
#else
#define CSRT_API
#endif
#else
#define CSRT_API
#endif

namespace CSRT {

//
// Global Functions
//

// Must call 'StartSystem' at the first and call 'CloseSystem' at the end.
CSRT_API void StartTheSystem(int count = 0, const char *logPath = nullptr);
CSRT_API void CloseTheSystem();

// Build in image loader
// Load image from file path. You cannot free the 'data' pointer manually.
// Call 'FinishLoadImage' will free all the current loading image data.
CSRT_API bool LoadImage(const char *path, int &width, int &height, int &channels, unsigned char **data);
CSRT_API void FinishLoadImage();

// Build in logging
CSRT_API void Info(const char *message);
CSRT_API void Warning(const char *message);
CSRT_API void Error(const char *message);
CSRT_API void Critical(const char *message);

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
    bool UseDepthHOG;
    bool UseDepthGray;
    bool UseDepthNormal;
    bool UseDepthHON;
    bool UseDepthNormalHOG;
    int NumHOGChannelsUsed;
    int PCACount;		        // Feature PCA count.

    bool UseNormalForSegment;	// Use normal data for segmentation.
    bool UseNormalForDSST;		// Use normal data for DSST.
    bool UseChannelWeights;	    // Use different weights for each channel.
    bool UseSegmentation;	    // Use segmentation for spatial constraint.

    int AdmmIterations;		    // Iteration number for optimized filter solver.
    float Padding;			    // Padding used to calculate template size. Affect how much area to search.
    float TemplateSize;		    // Specify the target template size.
    float GaussianSigma;	    // Guassian lable sigma factor.

    CSRTWindowType WindowFunc;	// Filter window function type.
    float ChebAttenuation;	    // Attenuation when use Cheb window.
    float KaiserAlpha;		    // Alpha value when use Kaiser window.

    float *WeightsLearnRates;	// Filter weights learn rate.
    float *FilterLearnRates;	// Filter learn rate.
    int LearnRateNum;			// The number of weight learn rate and filter learn weight.

    float HistLearnRate;	    // Histogram model learn rate.
    float ScaleLearnRate;	    // DSST learn rate.

    float BackgroundRatio;	    // Background extend ratio.
    int HistogramBins;		    // Bins number for the hisogram extraction.
    int PostRegularCount;	    // Iteration count for post regularization count in segment process.
    float MaxSegmentArea;	    // Controls the max area used for segment probability computation.

    int ScaleCount;			    // Scale numbers for DSST.
    float ScaleSigma;		    // DSST scale sigma factor.
    float ScaleMaxArea;		    // Max model area for DSST.
    float ScaleStep;		    // Scale step for DSST.

    int UpdateInterval;		    // Update frame interval. Set to 0 or negative to disable background update mode.
    bool UseScale;              // Whether use DSST to perform scale estimation.
    float FailThreshold;        // [0, 1] value. Below this threshold will be considered as track failure.

private:
    static float DefaultLearnRates[3];
};

enum class CSRTrackMode {
    RGB = 0,
    Depth = 1
};

struct Bounds {
    Bounds() : x0(0), y0(0), x1(0), y1(0) {}
    Bounds(float xx0, float yy0, float w, float h) : x0(xx0), y0(yy0), x1(xx0 + w), y1(yy0 + h) {}

    float x0;
    float y0;
    float x1;
    float y1;
};


//
// Global Classes
//

class Processor;
class MemoryArena;

class CSRT_API CSRTracker {
public:
    CSRTracker(int rows, int cols, int targetCount, CSRTrackMode mode)
    : CSRTracker(rows, cols, targetCount, mode, CSRTrackerParams()) {}
    CSRTracker(int rows, int cols, int targetCount, CSRTrackMode mode, const CSRTrackerParams &trackerParams);
    ~CSRTracker();

    // Input image data is continuous unsigned char memory block, in RGB interleaved format.
    void Initialize(unsigned char *sourceData, int channels, const Bounds *bbs);
    // Input image data is continuous float memory block. Normal data can be set to nullptr to disable the according features.
    // Depth data size should be equal to rowNum * colNum. Normal data is in polar representation and should be equal torowNum * 2colNum.
    void Initialize(float *depthData, float *normalData, const Bounds *bbs);

    // Update in RGB mode.
    bool Update(unsigned char *sourceData, int channels, Bounds *bbs);
    // Update in Depth mode.
    bool Update(float *depthData, float *normalData, Bounds *bbs);

    // Set the reinitialize flag to prepare the reinitialization work.
    void SetReinitialize();

private:
    void Rgba2Rgb(unsigned char *srcData, unsigned char *dstData);

    int rowNum;
    int colNum;
    int targets;
    Processor *processor;
    MemoryArena *arena;
};

}   // namespace CSRT

#endif //CSRT_CSRTRACKER_H
