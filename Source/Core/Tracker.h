//
// CSRDCF Tracker.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_TRACKER_H
#define CSRT_TRACKER_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"
#include "DSST.h"
#include "Segment.h"

namespace CSRT {

enum class WindowType {
	Hann = 0,
	Cheb = 1,
	Kaiser = 2
};

struct TrackerParams {
	// Default constructor.
	TrackerParams();

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
	float PeakRatio;		// Specify the peak occupation ratio to calculate PSR. PeakSize = ResponseSize * PeakRatio * 2 + 1.
	float PSRThreshold;		// PSR threshold used for failure detection. 20.0 ~ 60.0 is indicates strong peaks.
}; 

class Tracker {
public:
	// Tracker Public Methods
	Tracker() : Tracker(TrackerParams()) {}
	Tracker(const TrackerParams &trackerParams);
	~Tracker();

	void SetInitialMask(const MatF &mask);
	void Initialize(const Mat &image, const Bounds2i &bb);
	bool Update(const Mat &image, Bounds2i &bb, float &score);

	void SetReinitialize();

private:
	// Tracker Private Methods
	void GetFeatures(const Mat &patch, const Vector2i &featureSize, std::vector<MatF> &feats, MemoryArena *targetArena = nullptr) const;

	// Optimization solver for the CSRDCF tracker.
	// Y is the initial response. P is the spatial mask. All the input features has been trasformed int to frequency space.
	void CreateFilters(const std::vector<MatCF> &feats, const MatCF &Y, const MatF &P, std::vector<MatCF> &filters, MemoryArena *targetArena = nullptr) const;
	// G = (Sxy + mu * H - L) / (Sxx + mu).
	void CalcGMHelper(const MatCF &Sxy, const MatCF &Sxx, const MatCF &H, const MatCF &L, MatCF &G, float mu) const;
	// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
	void CalcHMHelper(const MatCF &G, const MatCF &L, const MatF &P, MatCF &H, MatF &h, float mu, float lambda) const;
	// L = L + mu * (G - H).
	void CalcLMHelper(const MatCF &G, const MatCF &H, MatCF &L, float mu) const;

	// Calculate response based on all the filters and channels weights.
	void CalculateResponse(const Mat &image, const std::vector<MatCF> &filter, MatF &response) const;

	// Update the filters.
	void UpdateFilter(const Mat &image, const MatF &mask);

	// Calculate spatial location prior.
	void GetLocationPrior(const Vector2i &targetSize, const Vector2i &imgSize, MatF &prior) const;

	// Extract foreground histogram and background histogram.
	void ExtractHistograms(const Mat &image, const Bounds2i& bb, Histogram &hf, Histogram &hb);

	// Update histogram model.
	void UpdateHistograms(const Mat &image, const Bounds2i& bb);

	// Sgement the target object foreground.
	void SegmentRegion(const Mat &image, const Vector2i &objCenter, 
		const Vector2i &tempSize, const Vector2i &tarSize, float scaleFactor, MatF &mask) const;

	// Estimate new target position.
	Vector2i EstimateNewPos(const Mat &image, float &score) const;

	// Check mask area for small sum value.
	bool CheckMaskArea(const MatF &mat, int area) const;

	// Reset all arenas.
	void ResetArenas(bool background = false) const;

	// Background update logic.
	void StartBackgroundUpdate(const Mat& image);
	void FetchUpdateResult();
	void BackgroundUpdate();


	// Tracker Private Members
	bool initialized;					// Initialize flag.
	TrackerParams params;				// Algorithm parameters.
	MatF window;						// Filter window function.
	int cellSize;						// Image cell size for related calculation.

	Vector2i objectCenter;				// Target object center.
	Bounds2i objbb;						// Target object bounding box.
	Vector2i orgTargetSize;				// Original target object size.
	Vector2i imageSize;					// Input image size.
	Vector2i templateSize;				// Template size for final calculation.
	Vector2i rescaledTemplateSize;		// Rescale of template size.
	float currentScaleFactor;			// Current scale factor for this tracker.
	float rescaleRatio;					// Rescale factor.

	MatCF yf;							// Response in the frequency space.
	
	MatF project;						// Feature PCA project matrix.
	std::vector<MatCF> filters;			// Calculated CSRDCF filters.
	std::vector<float> filterWeights;	// Filter channel weights.

	DSST dsst;							// DSST for scale estimation.
	Histogram histFore;					// Foreground histogram model.
	Histogram histBack;					// Background histogram model.
	float pFore;						// Probobility for the foreground. Area ratio.

	int defaultMaskArea;				// Default mask area.
	MatF defaultMask;					// Default mask for pipeline without segmentation.
	MatF filterMask;					// Segmented filter mask.
	MatF presetMask;					// Preset target object mask to help segmentation.
	MatF erodeKernel;					// 3x3 erode kernel.

	MemoryArena *arenas;				// Memory arena management

	// Background thread backup.
	Mat bk_image;
	Vector2i bk_center;
	Bounds2i bk_bb;
	float bk_scale;
	std::vector<MatCF> bk_filters;
	std::vector<float> bk_filterWeights;

	bool backgroundUpdateFlag;
	int updateCounter;
};

}	// namespace CSRT

#endif	// CSRT_TRACKER_H