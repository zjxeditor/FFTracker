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

extern CSRT_API void StartSystem(int count = 0, const char *logPath = nullptr);
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

    int CellSize;			    // Feature extraction cell size.
    int AdmmIterations;		    // Iteration number for optimized filter solver.
    float Padding;			    // Padding used to calculate template size. Affect how much area to search.
    int TemplateSize;		    // Specify the target template size.
    float GaussianSigma;	    // Guassian lable sigma factor.

    CSRTWindowType WindowFunc;	    // Filter window function type.
    float ChebAttenuation;	    // Attenuation when use Cheb window.
    float KaiserAlpha;		    // Alpha value when use Kaiser window.

    float WeightsLearnRate;	    // Filter weights learn rate.
    float FilterLearnRate;	    // Filter learn rate.
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
    bool UseSmoother;           // Whether use smoother to filter the tracker outputs.
};

enum class CSRTrackMode {
    RGB = 0,
    Depth = 1
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
