//
// Target object foreground segmentation.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_SEGMENT_H
#define CSRT_SEGMENT_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"

namespace CSRT {

class CSRT_API Histogram {
public:
	Histogram(MemoryArena *storage = nullptr) : dim(0), bins(0), size(0), arena(storage), data(nullptr), dimCoe(nullptr) { }
	Histogram(int dimNum, int binsNum, MemoryArena *storage = nullptr);
	Histogram(const Histogram &hist);
	Histogram(Histogram &&hist) noexcept;
	Histogram &operator=(const Histogram &hist);
	Histogram &operator=(Histogram &&hist) noexcept;
	Histogram &Reshape(int dimNum, int binsNum = 8);
	~Histogram();

	inline int Dim() const { return dim; }
	inline int Bins() const { return bins; }
	inline int Size() const { return size; }
	inline float *Data() const { return data; }

	// Extract the foreground histogram. Note: the bounding box must be located in the image.
	// No boundary check will be performed in this method.
	void ExtractForeHist(const Mat &imgM, const Bounds2i &bb);

	// Extract the background histogram. Note: the bounding box must be located in the image.
	// No boundary check will be performed in this method.
	void ExtractBackHist(const Mat &imgM, const Bounds2i &inbb, const Bounds2i &outbb);

	// Bakc project the histogram.
	void BackProject(const Mat &inM, MatF &outM) const;

private:
	float KernelProfileEpanechnikov(float x) const {
		return (x <= 1.0f) ? (2.0f / Pi)*(1 - x) : 0.0f;
	}

	// Histogram Private Data
	int dim, bins, size;
	MemoryArena *arena;
	float *data;
	int *dimCoe;
};

class CSRT_API Segment {
public:
	Segment();
	~Segment();
	void Initialize();

	void ComputePosteriors(const Mat &imgM, MatF &forePostM, MatF &backPostM, float pf,
		const MatF &fgPriorr, const Histogram &foreHist, const Histogram &backHist, int maxIter, float maxArea);

private:
	static void RegularizeSegmentation(const MatF &inForePost, const MatF &inBackPost, MatF &inForePrior, MatF &inBackPrior,
		MatF &outForePost, MatF &outBackPost, int maxIter);

	static float Gaussian(float x2, float y2, float sigma2) {
		return exp(-(x2 + y2) / (2 * sigma2)) / (2 * Pi*sigma2);
	}

	bool initialized;
	static bool SingleInstance;
};

// Global Segment Helper.
extern CSRT_API Segment GSegment;

}	// namespace CSRT

#endif	// CSRT_SEGMENT_H