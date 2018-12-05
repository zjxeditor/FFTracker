//
// Main interface for the tracking system
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_PROCESSOR_H
#define CSRT_PROCESSOR_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"
#include "InfoProvider.h"

namespace CSRT {

// Must call 'StartSystem' at the first and call 'CloseSystem' at the end.
void StartSystem(int count = 0, const char *logPath = nullptr);
void CloseSystem();

// Change from rgba data format to rgb data format.
void Rgba2Rgb(unsigned char *srcData, unsigned char *dstData, int rows, int cols);

// Calculate normal information from depth map.
void ComputeNormalFromPointCloud(const Vector3f *pointCloud, Vector3f *normal, int width, int height);
// Calculate polar normal information from depth map. First component is phi [0, Pi/2], second component is theta [0, 2Pi], both are normalized to [0, 1].
void ComputePolarNormalFromPointCloud(const Vector3f *pointCloud, Vector2f *polarNormal, int width, int height);
// Convert normalized normal to normalized polar representation.
void ConvertNormalToPolarNormal(const Vector3f *normal, Vector2f *polarNormal, int width, int height);
// Convert normalized polar normal to normalized normal representation.
void ConvertPolarNormalToNormal(const Vector2f *polarNormal, Vector3f *normal, int width, int height);


struct TrackerParams {
	// Default constructor.
	TrackerParams();

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
	int PCACount;		// Feature PCA count.

	bool UseNormalForSegment;	// Use normal data for segmentation.
	bool UseNormalForDSST;		// Use normal data for DSST.
	bool UseChannelWeights;	// Use different weights for each channel.
	bool UseSegmentation;	// Use segmentation for spatial constraint.

	int AdmmIterations;		// Iteration number for optimized filter solver.
	float Padding;			// Padding used to calculate template size. Affect how much area to search.
	float TemplateSize;		// Specify the target template size.
	float GaussianSigma;	// Guassian lable sigma factor.

	WindowType WindowFunc;	// Filter window function type.
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
	bool UseScale;          // Whether use DSST to perform scale estimation.
	bool UseSmoother;       // Whether use smoother to filter the tracker outputs.
	bool UseFastScale;		// Whether use fast dsst method to estimate scale.
};

enum class TrackMode {
	RGB = 0,
	Depth = 1
};

class Processor {
public:
	Processor(const TrackerParams &trackParams, int count, const Vector2i &moveSize, TrackMode mode);
	~Processor();

	// Prepare for reinitialize work.
	void SetReinitialize();

	// Initialize the system with rgb image and initial bounding boxes.
	void Initialize(const Mat &rgbImage, const std::vector<Bounds2f> &bbs);

	// Initialize the system with depth image, polar normal image and initial bounding boxes. Polar normal image can be empty.
	void Initialize(const MatF &depthImage, const MatF& normalImage, const std::vector<Bounds2f> &bbs);

	// Update the system with rgb image and output the result bounding boxes.
	void Update(const Mat &rgbImage, std::vector<Bounds2f> &bbs);

	// Update the system with depth image, polar normal image and output the result bounding boxes.
	void Update(const MatF &depthImage, const MatF& normalImage, std::vector<Bounds2f> &bbs);

private:
	// Reset internal arenas.
	void ResetArenas(bool background = false) const;

	// Background update logic.
	void StartBackgroundUpdate(const Mat &rgbImage);
	void StartBackgroundUpdate(const MatF &depthImage, const MatF &normalImage);
	void FetchUpdateResult();
	void BackgroundUpdateRGB();
	void BackgroundUpdateDepth();

	// Private members.
	TrackerParams params;
	int targetCount;
	Vector2i moveRegion;
	TrackMode trackMode;
	uint8_t featMask;
	bool initialized;
	MemoryArena *arenas;

	// Members for background update.
	Mat bk_rgbImage;
	MatF bk_depthImage;
	MatF bk_normalImage;
	std::vector<Vector2f> bk_positions;
	std::vector<float> bk_scales;
	std::vector<Bounds2f> bk_bounds;
	bool backgroundUpdateFlag;
	int updateCounter;
};

}	// namespace CSRT

#endif	// CSRT_PROCESSOR_H
