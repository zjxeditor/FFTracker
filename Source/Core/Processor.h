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

	int CellSize;			// Feature extraction cell size.
	int AdmmIterations;		// Iteration number for optimized filter solver.
	float Padding;			// Padding used to calculate template size. Affect how much area to search.
	int TemplateSize;		// Specify the target template size.
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
	void Initialize(const Mat &rgbImage, const std::vector<Bounds2i> &bbs);

	// Initialize the system with depth image, polar normal image and initial bounding boxes. Polar normal image can be empty.
	void Initialize(const MatF &depthImage, const MatF& normalImage, const std::vector<Bounds2i> &bbs);

	// Update the system with rgb image and output the result bounding boxes.
	void Update(const Mat &rgbImage, std::vector<Bounds2i> &bbs);

	// Update the system with depth image, polar normal image and output the result bounding boxes.
	void Update(const MatF &depthImage, const MatF& normalImage, std::vector<Bounds2i> &bbs);

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
	std::vector<Vector2i> bk_positions;
	std::vector<float> bk_scales;
	std::vector<Bounds2i> bk_bounds;
	bool backgroundUpdateFlag;
	int updateCounter;
};

}	// namespace CSRT

#endif	// CSRT_PROCESSOR_H
