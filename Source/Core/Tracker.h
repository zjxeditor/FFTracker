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

namespace CSRT {

class Tracker {
public:
	// Tracker Public Methods
	Tracker(bool channelWeights, int pca, int iter, float learnRateOfChannel, float learnRateOfFilter);
	~Tracker();

	// Set initialize flag to re-initialize.
	void SetReinitialize() { initialized = false; }

	// Initialize this tracker with extracted features, ideal response and spatial filter mask.
	void Initialize(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask, float orgArea);

	// Get the new position according to previous position.
	void GetPosition(const std::vector<MatF> &feats, float currentScale, int cellSize, float rescaleRatio,
		const Vector2f &currentPos, const Vector2i &moveSize, Vector2f &newPos, float &score) const;

	// Update the tracker model.
	void Update(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask);

	// Background update logic.
	void StartBackgroundUpdate();
	void FetchUpdateResult();
	void BackgroundUpdate(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask);

private:
	// Tracker Private Methods

	// Optimization solver for the CSRDCF tracker.
	// Y is the initial response. P is the spatial mask. All the input features has been trasformed int to frequency space.
	void CreateFilters(const std::vector<MatCF> &feats, const MatCF &Y, const MatF &P, std::vector<MatCF> &filters, MemoryArena *targetArena = nullptr) const;
    void CreateFilters1(const std::vector<MatCF> &feats, const std::vector<MatCF> &HTS, const MatCF &Y, const MatF &P, std::vector<MatCF> &filters, MemoryArena *targetArena = nullptr) const;
	// G = (Sxy + mu * H - L) / (Sxx + mu).
	void CalcGMHelper(const MatCF &Sxy, const MatCF &Sxx, const MatCF &H, const MatCF &L, MatCF &G, float mu) const;
	// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
	void CalcHMHelper(const MatCF &G, const MatCF &L, const MatF &P, MatCF &H, MatF &h, float mu, float lambda) const;
    // H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
    void CalcHMHelper1(const MatCF &G, const MatCF &L, const MatCF &HT, const MatCF &L1, const MatF &P, MatCF &H, MatF &h, float mu, float mu1, float lambda) const;
	// L = L + mu * (G - H).
	void CalcLMHelper(const MatCF &G, const MatCF &H, MatCF &L, float mu) const;

	// Calculate response based on all the filters and channels weights. Feats are extracted at previous location and scale.
	void CalculateResponse(const std::vector<MatF> &feats, const std::vector<MatCF> &filter, MatF &response) const;

	// Calculate the response score.
	float CalculateScore(const MatF &response, const Vector2i &pos, float maxResp) const;

	// Find sub peak.
	void FindSubPeak(const MatF &response, const Vector2i &peakPos, Vector2i &subPos, float &subValue) const;

	// Reset all arenas.
	void ResetArenas(bool background = false) const;

	// Tracker Private Members
	bool initialized;					// Initialize flag.
	bool useChannelWeights;				// Whether to use channel weights.
	int pcaCount;						// Feature PCA count.
	int iterations;						// Filter solver iteration count.
	float weightsLearnRate;				// Channel weights learning weight.
	float filterLearnRate;				// Filter model learning weight.

	MatF project;						// Feature PCA project matrix.
	std::vector<MatCF> filters;			// Calculated CSRDCF filters.
	std::vector<float> filterWeights;	// Filter channel weights.
	MemoryArena *arenas;				// Memory arena management

	// Background thread backup.
	std::vector<MatCF> bk_filters;
	std::vector<float> bk_filterWeights;
	bool backgroundUpdateFlag;

	// Target area
	float targetArea;
	float k;
};

}	// namespace CSRT

#endif	// CSRT_TRACKER_H