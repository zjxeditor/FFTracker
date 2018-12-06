//
// Provide features from various kinds of image to the tracking system. It can be customized for different features configuration.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_INFO_PROVIDER_H
#define CSRT_INFO_PROVIDER_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"
#include "../Utility/Smoother.h"
#include "Segment.h"
#include "DSST.h"
#include "FDSST.h"
#include "Tracker.h"
#include "RotationEstimator.h"

namespace CSRT {

enum class WindowType {
	Hann = 0,
	Cheb = 1,
	Kaiser = 2
};

class Processor;

class InfoProvider {
	friend class Processor;

public:
	InfoProvider();
	~InfoProvider();
	void Initialize();

	// Get image features used for DSST. Currently, only HOG features are applied.
	void GetScaleFeatures(const Mat &img, MatF &feat, const Vector2f &pos, const Vector2f &orgSize, float currentScale,
		const std::vector<float> &factors, const MatF &window, const Vector2i &modelSize) const;
	// Get depth features used for DSST. Currently, only HOG features are applied. Normal data is in normalized polar representation.
	void GetScaleFeatures(const MatF &depth, const MatF &normal, MatF &feat, const Vector2f &pos, const Vector2f &orgSize, float currentScale,
		const std::vector<float> &factors, const MatF &window, const Vector2i &modelSize, bool useNormal) const;

	// Get image features used for rotation estimation, only HOG features are applied.
    void GetRotationFeatures(const Mat& img, MatF& feat, const Vector2f& pos, const Vector2f& orgSize, float currentScale,
            const std::vector<float>& factors, const MatF& window, const Vector2i& modelSize) const;

	// Get image features used for tracker. Features types are stored in mask. 
	// 1 bit -> HOG; 2 bit -> CN; 3 bit -> GRAY; 4 bit -> RGB.
	// cellSize and hogNum are only used for HOG features.
	void GetTrackFeatures(const Mat &patch, const Vector2i &featureSize, std::vector<MatF> &feats, const MatF &window, 
		uint8_t mask, int cellSize, int hogNum, MemoryArena *targetArena = nullptr) const;
	// Get depth features used for tracker. Features types are stored in mask. Normal patch data is in normalized polar representation.
	// 1 bit -> Depth HOG; 2 bit -> GRAY; 3 bit -> Normal; 4 bit -> Normal HON; 5 bit -> Normal HOG. 
	// cellSize and hogNum are only used for HOG features.
	void GetTrackFeatures(const MatF &depthPatch, const MatF &normalPatch, const Vector2i &featureSize, std::vector<MatF> &feats, const MatF &window,
		uint8_t mask, int cellSize, int hogNum, MemoryArena *targetArena = nullptr) const;

	// Modules configuration. Must be configured in order.
	void ConfigureTargets(int count, const Vector2i &moveRegion, const std::vector<Bounds2f> &bbs);
	void ConfigureTracker(float localPadding, float referTemplateSize, float gaussianSigma,
		bool channelWeights, int pca, int iteration, float learnRateOfChannel, float learnRateOfFilter,
		WindowType windowFunc = WindowType::Hann, float chebAttenuation = 45.0f, float kaiserAlpha = 3.75f);
	void ConfigureDSST(int numberOfScales, float stepOfScale, float sigmaFactor, float learnRateOfScale, float maxArea);
    void ConfigureFDSST(int numberOfInterpScales, float stepOfScale, float sigmaFactor, float learnRateOfScale, float maxArea);
	void ConfigureSegment(int histDim, int histBin, float bgRatio, float learnRateOfHist, int regularCount, float maxSegArea);
	void ConfigureSmoother(float stepOfScale);
    void ConfigureRotation(int numberOfRotation, float sigmaFactor, float learnRateOfRotation, float maxArea);

	// Extract foreground histogram and background histogram.
	void ExtractHistograms(const Mat &image, const std::vector<Bounds2f> &bbs, std::vector<Histogram> &hfs, std::vector<Histogram> &hbs);
	// Update histogram model.
	void UpdateHistograms(const Mat &image, const std::vector<Bounds2f> &bbs);
	// Segment the target object foreground.
	void SegmentRegion(const Mat &image, const std::vector<Vector2f> &objPositions, const std::vector<float> &objScales);

	// Get current target's information.
	Vector2f GetCurrentPosition(int i) { return currentPositions[i]; }
	float GetCurrentScale(int i) { return currentScales[i]; }
	Bounds2f GetCurrentBounds(int i) { return currentBounds[i]; }
	float GetCurrentRotation(int i) { return currentRotations[i]; }

private:
	// Calculate spatial location prior.
	void GetLocationPrior(const Vector2f &targetSize, const Vector2i &imgSize, MatF &prior) const;
	// Check mask area for small sum value.
	bool CheckMaskArea(const MatF &mat, float area) const;
	// Reset internal arenas.
	void ResetArenas() const;

	bool initialized;
	static bool SingleInstance;

	int targetCount;							// Target number.
	int cellSize;								// Image cell size for related calculation, choose from [2, 4].
	Vector2i moveSize;							// Movable region size.
	std::vector<Vector2f> orgTargetSizes;		// Original targets' sizes.
	std::vector<Bounds2f> currentBounds;		// Current targets' bounding box.
	std::vector<Vector2f> currentPositions;		// Current targets' positions.
	std::vector<float> currentScales;			// Current targets' scale factors.
	std::vector<float> currentRotations;        // Current targets' rotation factors in radians.

	// Members used for Tracker.
	Vector2f templateSize;						// Template size for tracking local region.
	Vector2i rescaledTemplateSize;				// Rescale of template size for final calculation.
	float rescaleRatio;							// Rescale factor.
	MatCF yf;									// Ideal response in the frequency space.
	MatF window;								// Filter window function.

	// Members used for DSST.
	int scaleCount;								// Total scale count.
	Vector2i scaleModelSize;					// Final calculation size. 
	MatCF scaleGSF;								// Ideal frequency response, a single row.
	MatF scaleWindow;							// Signal window.
	std::vector<float> scaleMinFactors;			// Minimum scale factors.
	std::vector<float> scaleMaxFactors;			// Maximum scale factors.
	std::vector<float> scaleFactors;			// Calculated scale factors of each level.
	// Additional members for fast DSST
	int scaleInterpCount;                       // Total interpolation scale count.
    std::vector<float> scaleInterpFactors;		// Calculated interpolation scale factors of each level.

	// Members used for segmentation.
	std::vector<Histogram> histFores;			// Foreground histogram model.
	std::vector<Histogram> histBacks;			// Background histogram model 
	std::vector<float> pFores;					// Probability for the foreground, area ratio.
	std::vector<float> defaultMaskAreas;			// Default mask area.
	std::vector<MatF> defaultMasks;				// Default mask for pipeline without segmentation.
	std::vector<MatF> filterMasks;				// Segmented filter mask.
	Mat erodeKernel;							// 3x3 erode kernel.
	float backgroundRatio;						// Background extension ratio for histogram calculation.
	float histLearnRate;						// Histogram learning rate.
	int postRegularCount;						// Iteration count for post regularization.
	float maxSegmentArea;						// The maximum area used for segmentation.

	// Members used for rotation estimation.
    int rotationCount;							// Total rotation count.
    Vector2i rotationModelSize;					// Final calculation size.
    MatCF rotationGSF;							// Ideal frequency response, a single row.
    MatF rotationWindow;						// Signal window.
    std::vector<float> rotationFactors;			// Calculated scale factors of each level.

	// Smooth filter for each tracker.
	std::vector<FilterDoubleExponential1D> smoother1Ds;
	std::vector<FilterDoubleExponential2D> smoother2Ds;

	// Track components.
	std::vector<Tracker> trackers;
	std::vector<DSST> dssts;
    std::vector<FDSST> fdssts;
    std::vector<RotationEstimator> rotors;
};

// Global Feature Provider
extern InfoProvider GInfoProvider;

}	// namespace CSRT

#endif	// CSRT_INFO_PROVIDER_H