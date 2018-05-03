//
// Target object foreground segmentation.
//

#include "Segment.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "Filter.h"

namespace CSRT {

Segment GSegment;

// Filter memory arena management.
static MemoryArena *GSegmentArenas = nullptr;

//
// Histogram Implementation
//

Histogram::Histogram(int dimNum, int binsNum, MemoryArena *storage) : dim(dimNum), bins(binsNum), arena(storage) {
	size = (int)std::pow(bins, dim);
	if (arena) {
		data = arena->Alloc<float>(size, false);
		dimCoe = arena->Alloc<int>(dim, false);
	} else {
		data = AllocAligned<float>(size);
		dimCoe = AllocAligned<int>(dim);
	}
	memset(data, 0, size * sizeof(float));
	dimCoe[dim - 1] = 1;
	for (int i = 0; i < dim - 1; ++i)
		dimCoe[i] = (int)std::pow(bins, dim - 1 - i);
}

Histogram::Histogram(const Histogram& hist) : dim(hist.dim), bins(hist.bins), size(hist.size), arena(hist.arena) {
	if (arena) {
		data = arena->Alloc<float>(size, false);
		dimCoe = arena->Alloc<int>(dim, false);
	} else {
		data = AllocAligned<float>(size);
		dimCoe = AllocAligned<int>(dim);
	}
	memcpy(data, hist.data, size * sizeof(float));
	memcpy(dimCoe, hist.dimCoe, dim * sizeof(int));
}

Histogram::Histogram(Histogram&& hist) noexcept  : dim(hist.dim), bins(hist.bins), size(hist.size),
arena(hist.arena), data(hist.data), dimCoe(hist.dimCoe) {
	hist.dim = 0;
	hist.bins = 0;
	hist.size = 0;
	hist.arena = nullptr;
	hist.data = nullptr;
	hist.dimCoe = nullptr;
}

Histogram& Histogram::operator=(const Histogram& hist) {
	dim = hist.dim;
	bins = hist.bins;
	size = hist.size;
	if (!arena) {
		if (data) FreeAligned(data);
		if (dimCoe) FreeAligned(dimCoe);
	}
	if (arena) {
		data = arena->Alloc<float>(size, false);
		dimCoe = arena->Alloc<int>(dim, false);
	} else {
		data = AllocAligned<float>(size);
		dimCoe = AllocAligned<int>(dim);
	}
	memcpy(data, hist.data, size * sizeof(float));
	memcpy(dimCoe, hist.dimCoe, dim * sizeof(int));

	return *this;
}

Histogram& Histogram::operator=(Histogram&& hist) noexcept {
	dim = hist.dim;
	bins = hist.bins;
	size = hist.size;
	if (!arena) {
		if (data) FreeAligned(data);
		if (dimCoe) FreeAligned(dimCoe);
	}
	if (arena == hist.arena) {
		data = hist.data;
		dimCoe = hist.dimCoe;
		hist.dim = 0;
		hist.bins = 0;
		hist.size = 0;
		hist.arena = nullptr;
		hist.data = nullptr;
		hist.dimCoe = nullptr;
	} else {
		if (arena) {
			data = arena->Alloc<float>(size, false);
			dimCoe = arena->Alloc<int>(dim, false);
		} else {
			data = AllocAligned<float>(size);
			dimCoe = AllocAligned<int>(dim);
		}
		memcpy(data, hist.data, size * sizeof(float));
		memcpy(dimCoe, hist.dimCoe, dim * sizeof(int));
	}

	return *this;
}

Histogram& Histogram::Reshape(int dimNum, int binsNum) {
	dim = dimNum;
	bins = binsNum;
	size = (int)std::pow(bins, dim);
	if (!arena) {
		if (data) FreeAligned(data);
		if (dimCoe) FreeAligned(dimCoe);
	}
	if (arena) {
		data = arena->Alloc<float>(size, false);
		dimCoe = arena->Alloc<int>(dim, false);
	} else {
		data = AllocAligned<float>(size);
		dimCoe = AllocAligned<int>(dim);
	}
	memset(data, 0, size * sizeof(float));
	dimCoe[dim - 1] = 1;
	for (int i = 0; i < dim - 1; ++i)
		dimCoe[i] = (int)std::pow(bins, dim - 1 - i);

	return *this;
}

Histogram::~Histogram() {
	if (!arena) {
		if (data) FreeAligned(data);
		if (dimCoe) FreeAligned(dimCoe);
	}
}

void Histogram::ExtractForeHist(const Mat &imgM, const Bounds2i &bb) {
	if (imgM.Channels() != dim) {
		Critical("Histogram::ExtractForeHist: image channels number isn't match the histogram dimension.");
		return;
	}

	// Manual calculate the spatial weights.
	// Weights are epanechnikov distributed, with peek at the center of the image.
	Vector2i vd = bb.Diagonal();
	float cx = bb.pMin.x + (vd.x - 1) / 2.0f;
	float cy = bb.pMin.y + (vd.y - 1) / 2.0f;
	float kernelWidth = sqrt(2.0f) / vd.x;
	float kernelHeight = sqrt(2.0f) / vd.y;
	MatF kernelWeightsM(vd.y, vd.x, &GSegmentArenas[ThreadIndex]);
	ParallelFor([&](int64_t y) {
		float *pd = kernelWeightsM.Data() + y * vd.x;
		float tmpy = std::pow((cy - y - bb.pMin.y) * kernelHeight, 2);
		for (int x = 0; x < vd.x; ++x) {
			float tmpx = std::pow((cx - x - bb.pMin.x) * kernelWidth, 2);
			pd[x] = KernelProfileEpanechnikov(tmpx + tmpy);
		}
	}, vd.y, 32);

	// Extract histogram.
	float div = bins / 256.0f;
	float sum = 0.0f;
	int imgStride = imgM.Cols() * imgM.Channels();
	for (int y = 0; y < vd.y; ++y) {
		const uint8_t *ps = imgM.Data() + (y + bb.pMin.y) * imgStride + bb.pMin.x * dim;
		const float *pw = kernelWeightsM.Data() + y * vd.x;
		for (int x = 0; x < vd.x; ++x) {
			int id = 0;
			for (int k = 0; k < dim; ++k)
				id += dimCoe[k] * (int)(*(ps++) * div);
			data[id] += *pw;
			sum += *pw;
			++pw;
		}
	}

	// Normalize
	div = 1.0f / sum;
	float *ptemp = data;
	for (int i = 0; i < size; ++i)
		*(ptemp++) *= div;

	GSegmentArenas[ThreadIndex].Reset();
}

void Histogram::ExtractBackHist(const Mat& imgM, const Bounds2i& inbb, const Bounds2i& outbb) {
	if (imgM.Channels() != dim) {
		Critical("Histogram::ExtractBackHist: image channels number isn't match the histogram dimension.");
		return;
	}

	// Extract histogram.
	float div = bins / 256.0f;
	float sum = 0.0f;
	int imgStride = imgM.Cols() * imgM.Channels();
	for (int y = outbb.pMin.y; y < outbb.pMax.y; ++y) {
		const uint8_t *ps = imgM.Data() + y * imgStride;
		for (int x = outbb.pMin.x; x < outbb.pMax.x; ++x) {
			if (y >= inbb.pMin.y && y < inbb.pMax.y && x >= inbb.pMin.x && x < inbb.pMax.x) {
				ps += dim;
				continue;
			}
			int id = 0;
			for (int k = 0; k < dim; ++k)
				id += dimCoe[k] * (int)(*(ps++) * div);
			data[id] += 1.0f;
			sum += 1.0f;
		}
	}

	// Normalize.
	div = 1.0f / sum;
	float *ptemp = data;
	for (int i = 0; i < size; ++i)
		*(ptemp++) *= div;
}

void Histogram::BackProject(const Mat& inM, MatF& outM) const {
	if (inM.Channels() != dim) {
		Critical("Histogram::BackProject: input image channels isn't match the histogram's dimension.");
		return;
	}

	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(rows, cols);
	float div = bins / 256.0f;
	int imgStride = inM.Cols() * inM.Channels();

	ParallelFor([&](int64_t y) {
		const uint8_t *ps = inM.Data() + imgStride * y;
		float *pd = outM.Data() + cols * y;
		for (int x = 0; x < cols; ++x) {
			int id = 0;
			for (int k = 0; k < dim; ++k)
				id += dimCoe[k] * (int)(*(ps++) * div);
			*(pd++) = data[id];
		}
	}, rows, 32);
}

//
// Segment Implementation.
//

bool Segment::SingleInstance = false;

Segment::Segment() : initialized(false) {
	if (SingleInstance) {
		Critical("Only one \"Segment\" can exists.");
		return;
	}
	SingleInstance = true;
}

Segment::~Segment() {
	if (GSegmentArenas) {
		delete[] GSegmentArenas;
		GSegmentArenas = nullptr;
	}
	SingleInstance = false;
}

void Segment::Initialize() {
	if (initialized) return;
	initialized = true;
	GSegmentArenas = new MemoryArena[MaxThreadIndex()];
}

void Segment::ComputePosteriors(const Mat& imgM, MatF& forePostM, MatF& backPostM, float pf,
	const MatF& fgPrior, const Histogram& foreHist, const Histogram& backHist, int maxIter, float maxArea) {
	// Perform input data check.
	if (imgM.Size() == 0) {
		Critical("Segment::ComputePosteriors: cannot compute posteriors on empty image.");
		return;
	}
	int rows = imgM.Rows();
	int cols = imgM.Cols();
	if (fgPrior.Rows() != rows || fgPrior.Cols() != cols) {
		Critical("Segment::ComputePosteriors: input foreground spatial prior size not match image size.");
		return;
	}
	if (foreHist.Dim() != imgM.Channels() || backHist.Dim() != imgM.Channels()) {
		Critical("Segment::ComputePosteriors: input histogram dimension not match image channels.");
		return;
	}

	// Compute resize factor so that the max area is 1024 (32x32).
	float factor = sqrt(maxArea / (rows * cols));
	if (factor > 1.0f) factor = 1.0f;
	Vector2i newSize(cols * factor, rows * factor);

	// Rescale the input data.
	Mat imgScaledM(&GSegmentArenas[ThreadIndex]);
	imgM.Resize(imgScaledM, newSize.y, newSize.x);

	// Initialize priors.
	MatF fgPriorScaled(&GSegmentArenas[ThreadIndex]), bgPriorScaled(&GSegmentArenas[ThreadIndex]);
	if (fgPrior.Size() == 0) {
		fgPriorScaled.Reshape(newSize.y, newSize.x);
		fgPriorScaled.ReValue(0.5f);
		bgPriorScaled = fgPriorScaled;
	} else {
		fgPrior.Resize(fgPriorScaled, newSize.y, newSize.x);
		bgPriorScaled = fgPriorScaled;
		bgPriorScaled.ReValue(-1.0f, 1.0f);
	}

	// Back project pixels likelihood. First do normal project, then muliply the spatial prior.
	MatF foreLikelihood(&GSegmentArenas[ThreadIndex]), backLikelihood(&GSegmentArenas[ThreadIndex]);
	foreHist.BackProject(imgScaledM, foreLikelihood);
	backHist.BackProject(imgScaledM, backLikelihood);
	foreLikelihood.CMul(fgPriorScaled);
	backLikelihood.CMul(bgPriorScaled);

	// Calculate posterior according to bayes rules.
	float pb = 1 - pf;
	MatF ppf(newSize.y, newSize.x, &GSegmentArenas[ThreadIndex]);
	MatF ppb(newSize.y, newSize.x, &GSegmentArenas[ThreadIndex]);
	float *psrc0 = foreLikelihood.Data();
	float *psrc1 = backLikelihood.Data();
	float *pdest0 = ppf.Data();
	float *pdest1 = ppb.Data();
	for (int index = 0; index < ppf.Size(); ++index) {
		float temp = pb * (*(psrc1++)) / (pf * (*(psrc0++)));
		if (std::isnan(temp))
			*pdest0 = 0.0f;
		else
			*pdest0 = 1.0f / (1.0f + temp);
		*(pdest1++) = 1.0f - *(pdest0++);
	}

	// Regularize.
	MatF rppf(&GSegmentArenas[ThreadIndex]), rppb(&GSegmentArenas[ThreadIndex]);
	RegularizeSegmentation(ppf, ppb, fgPriorScaled, bgPriorScaled, rppf, rppb, maxIter);

	// Recover to original size.
	rppf.Resize(forePostM, rows, cols);
	rppb.Resize(backPostM, rows, cols);

	GSegmentArenas[ThreadIndex].Reset();
}

void Segment::RegularizeSegmentation(const MatF& inForePost, const MatF& inBackPost,
	MatF& inForePrior, MatF& inBackPrior, MatF& outForePost, MatF& outBackPost, int maxIter) {
	int rows = inForePost.Rows();
	int cols = inForePost.Cols();
	int hsize = std::max(1, (int)(cols * 3.0f / 50.0f + 0.5f));
	int lambdaSize = 2 * hsize + 1;

	// Computer gaussian kernel.
	MatF lambdaM(lambdaSize, lambdaSize, &GSegmentArenas[ThreadIndex]);
	float std2 = std::pow(hsize / 3.0f, 2);
	float sumLambda = 0.0f;
	float *plambda = lambdaM.Data();
	for (int y = -hsize; y <= hsize; ++y) {
		float tempy = y * y;
		for (int x = -hsize; x <= hsize; ++x) {
			float tempgauss = Gaussian(x*x, tempy, std2);
			*(plambda++) = tempgauss;
			sumLambda += tempgauss;
		}
	}
	int kcIdx = lambdaM.Size() / 2;
	sumLambda -= lambdaM.Data()[kcIdx];
	lambdaM.Data()[kcIdx] = 0.0f;	// Set center of kernel to 0.
	sumLambda = 1.0f / sumLambda;
	lambdaM.ReValue(sumLambda, 0.0f);	// Normalize kernel to sum to 1.

	// Create lambda2 kernel.
	MatF lambda2M = lambdaM;
	lambda2M.Data()[kcIdx] = 1.0f;

	// Reshape output matrix.
	outForePost.Reshape(rows, cols);
	outBackPost.Reshape(rows, cols);

	// Create temporal matrix.
	MatF Si_o(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF Si_b(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF Ssum_o(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF Ssum_b(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF Qi_o(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF Qi_b(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF P_Io(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF P_Ib(rows, cols, &GSegmentArenas[ThreadIndex]);
	MatF normi(rows, cols, &GSegmentArenas[ThreadIndex]);

	// Start iteration.
	float terminateThr = 0.1f;
	float logLike = std::numeric_limits<float>::max();
	for (int i = 0; i < maxIter; ++i) {
		CwiseMul(inForePrior, inForePost, P_Io, 1.0f, std::numeric_limits<float>::epsilon());
		CwiseMul(inBackPrior, inBackPost, P_Ib, 1.0f, std::numeric_limits<float>::epsilon());

		GFilter.Correlation2D(inForePrior, lambdaM, Si_o, BorderType::Reflect);
		GFilter.Correlation2D(inBackPrior, lambdaM, Si_b, BorderType::Reflect);
		Si_o.CMul(inForePrior);
		Si_b.CMul(inBackPrior);
		CWiseAddRecip(Si_o, Si_b, normi);
		Si_o.CMul(normi);
		Si_b.CMul(normi);

		GFilter.Correlation2D(Si_o, lambda2M, Ssum_o, BorderType::Reflect);
		GFilter.Correlation2D(Si_b, lambda2M, Ssum_b, BorderType::Reflect);

		GFilter.Correlation2D(P_Io, lambdaM, Qi_o, BorderType::Reflect);
		GFilter.Correlation2D(P_Ib, lambdaM, Qi_b, BorderType::Reflect);
		Qi_o.CMul(P_Io);
		Qi_b.CMul(P_Ib);
		CWiseAddRecip(Qi_o, Qi_b, normi);
		Qi_o.CMul(normi);
		Qi_b.CMul(normi);

		GFilter.Correlation2D(Qi_o, lambda2M, outForePost, BorderType::Reflect);
		GFilter.Correlation2D(Qi_b, lambda2M, outBackPost, BorderType::Reflect);

		CWiseAdd(outForePost, Ssum_o, inForePrior, 0.25f);
		CWiseAdd(outBackPost, Ssum_b, inBackPrior, 0.25f);
		CWiseAddRecip(inForePrior, inBackPrior, normi);
		inForePrior.CMul(normi);
		inBackPrior.CMul(normi);

		// Judge converge.
		CWiseLog(outForePost, Ssum_o);
		CWiseLog(outBackPost, Ssum_b);
		CWiseAdd(Ssum_o, Ssum_b, normi);
		float mean = normi.Mean();
		float logLikeNew = -mean / 2;
		if (std::abs(logLike - logLikeNew) < terminateThr)
			break;
		logLike = logLikeNew;
	}
}

}	// namespace CSRT