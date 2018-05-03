//
// CSRDCF Tracker.
//

#include "Tracker.h"
#include "DSST.h"
#include "Features.h"
#include "Filter.h"
#include "Segment.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"

namespace CSRT {

// Default parameters.
TrackerParams::TrackerParams() {
	UseHOG = true;
	UseCN = true;
	UseGRAY = true;
	UseRGB = false;
	NumHOGChannelsUsed = 18;
	UsePCA = false;
	PCACount = 8;

	UseChannelWeights = true;
	UseSegmentation = true;

	AdmmIterations = 4;
	Padding = 3.0f;
	TemplateSize = 200;
	GaussianSigma = 1.0f;

	WindowFunc = WindowType::Hann;
	ChebAttenuation = 45.0f;
	KaiserAlpha = 3.75f;

	WeightsLearnRate = 0.02f;
	FilterLearnRate = 0.02f;
	HistLearnRate = 0.04f;
	ScaleLearnRate = 0.025f;

	BackgroundRatio = 2.0f;
	HistogramBins = 16;
	PostRegularCount = 10;
	MaxSegmentArea = 1024.0f;

	ScaleCount = 33;
	ScaleSigma = 0.25f;
	ScaleMaxArea = 512.0f;
	ScaleStep = 1.02f;

	UpdateInterval = 4;
	PeakRatio = 0.1f;
	PSRThreshold = 15.0f;
}

//
// Tracker Implementation
//

Tracker::Tracker(const TrackerParams &trackerParams) : initialized(false), params(trackerParams), cellSize(0),
	currentScaleFactor(0.0f), rescaleRatio(0.0f), pFore(0.0f), defaultMaskArea(0), arenas(nullptr),
	bk_scale(0.0f), backgroundUpdateFlag(false), updateCounter(0) {
	if (params.NumHOGChannelsUsed > HOG_FEATURE_COUNT)
		Critical("Used hog feature number is greater than hog feature total count.");
}

Tracker::~Tracker() {
	if (params.UpdateInterval > 0) {
		updateCounter = 0;
		backgroundUpdateFlag = false;
		BackgroundWait();
	}
	if (arenas) {
		delete[] arenas;
		arenas = nullptr;
	}
}

void Tracker::GetFeatures(const Mat& patch, const Vector2i& featureSize, std::vector<MatF>& feats, MemoryArena *targetArena) const {
	feats.clear();
	if (params.UseHOG) {
		std::vector<MatF> hog;
		GFeatsExtractor.GetFeaturesHOG(patch, hog, cellSize, targetArena);
		for (int i = 0; i < params.NumHOGChannelsUsed; ++i)
			feats.push_back(std::move(hog[i]));
	}
	if (params.UseCN) {
		std::vector<MatF> cn;
		GFeatsExtractor.GetFeaturesCN(patch, cn, featureSize, targetArena);
		for (int i = 0; i < (int)cn.size(); ++i)
			feats.push_back(std::move(cn[i]));
	}
	if (params.UseGRAY) {
		Mat gray(targetArena), grayResized(targetArena);
		patch.RGBToGray(gray);
		gray.Resize(grayResized, featureSize.y, featureSize.x);
		MatF grayFeat(targetArena);
		grayResized.ToMatF(grayFeat, 1.0f / 255.0f, -0.5f);
		feats.push_back(std::move(grayFeat));
	}
	if (params.UseRGB) {
		std::vector<MatF> rgb;
		GFeatsExtractor.GetFeaturesRGB(patch, rgb, featureSize, targetArena);
		for (int i = 0; i < (int)rgb.size(); ++i)
			feats.push_back(std::move(rgb[i]));
	}

	ParallelFor([&](int64_t index) {
		feats[index].CMul(window);
	}, (int)feats.size(), 4);
}

void Tracker::CreateFilters(const std::vector<MatCF>& feats, const MatCF& Y, const MatF& P, std::vector<MatCF>& filters, MemoryArena *targetArena) const {
	int len = (int)feats.size();
	filters.clear();
	filters.resize(len, MatCF(targetArena));
	for (int i = 0; i < len; ++i)
		filters[i].Reshape(Y.Rows(), Y.Cols(), false);

	ParallelFor([&](int64_t index) {
		float mu = 5.0f;
		float beta = 3.0f;
		float mu_max = 20.0f;
		float lambda = mu / 100.0f;

		const MatCF &F = feats[index];
		MatCF &H = filters[index];
		MatCF Sxy(&arenas[ThreadIndex]), Sxx(&arenas[ThreadIndex]);
		MatF h(&arenas[ThreadIndex]);
		// Initialize.
		GFilter.MulSpectrums(F, Y, Sxy, true);	// Sxy = F * Y.
		GFilter.MulSpectrums(F, F, Sxx, true);	// Sxx = F * F.
		// H = Sxy / [Sxx + lambda].
		GFilter.DivideComplexMatrices(Sxy, Sxx, H, ComplexF(lambda, lambda));
		GFFT.FFTInv2(H, h, true);
		h.CMul(P);
		GFFT.FFT2(h, H);
		MatCF L(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
		MatCF G(H.Rows(), H.Cols(), &arenas[ThreadIndex]);

		for (int iteration = 0; iteration < params.AdmmIterations - 1; ++iteration) {
			// G = (Sxy + mu * H - L) / (Sxx + mu).
			CalcGMHelper(Sxy, Sxx, H, L, G, mu);
			// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
			CalcHMHelper(G, L, P, H, h, mu, lambda);
			// Update variables for next iteration.
			// L = L + mu * (G - H).
			CalcLMHelper(G, H, L, mu);
			mu = std::min(mu_max, beta*mu);
		}
		// G = (Sxy + mu * H - L) / (Sxx + mu).
		CalcGMHelper(Sxy, Sxx, H, L, G, mu);
		// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
		CalcHMHelper(G, L, P, H, h, mu, lambda);
	}, len, 1);
}

void Tracker::CalcGMHelper(const MatCF& Sxy, const MatCF& Sxx, const MatCF& H, const MatCF& L, MatCF& G, float mu) const {
	const ComplexF *psxy = Sxy.Data();
	const ComplexF *psxx = Sxx.Data();
	const ComplexF *ph = H.Data();
	const ComplexF *pl = L.Data();
	ComplexF *pg = G.Data();

	// G = (Sxy + mu * H - L) / (Sxx + mu).
	for (int i = 0; i < G.Size(); ++i)
		*(pg++) = (*(psxy++) + mu * (*(ph++)) - *(pl++)) / (*(psxx++) + mu);
}

void Tracker::CalcHMHelper(const MatCF& G, const MatCF& L, const MatF& P, MatCF& H, MatF& h, float mu, float lambda) const {
	const ComplexF *pG = G.Data();
	const ComplexF *pL = L.Data();
	const float *pP = P.Data();
	ComplexF *pH = H.Data();
	float *ph = h.Data();

	// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).

	// H = mu * G + L.
	for (int i = 0; i < H.Size(); ++i)
		*(pH++) = mu * (*(pG++)) + *(pL++);

	// h = FFTInv2(H).
	GFFT.FFTInv2(H, h, true);

	// h = h * lm * P.
	float lm = 1.0f / (lambda + mu);
	for (int i = 0; i < h.Size(); ++i)
		*(ph++) *= (lm * (*(pP++)));

	// H = FFT2(h).
	GFFT.FFT2(h, H);
}

void Tracker::CalcLMHelper(const MatCF& G, const MatCF& H, MatCF& L, float mu) const {
	const ComplexF *pg = G.Data();
	const ComplexF *ph = H.Data();
	ComplexF *pl = L.Data();

	// L = L + mu * (G - H).
	for (int i = 0; i < L.Size(); ++i)
		*(pl++) += (mu * (*(pg++) - *(ph++)));
}

void Tracker::CalculateResponse(const Mat& image, const std::vector<MatCF>& filter, MatF& response) const {
	// Get ROI.
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(image, orgPatch, objectCenter, (int)(currentScaleFactor * templateSize.x),
		(int)(currentScaleFactor * templateSize.y));
	orgPatch.Resize(patch, rescaledTemplateSize.y, rescaledTemplateSize.x);

	// Extract features and do projection.
	std::vector<MatF> ftrs;
	GetFeatures(patch, Vector2i(yf.Cols(), yf.Rows()), ftrs, &arenas[ThreadIndex]);
	std::vector<MatCF> Ffeatures;
	if (params.UsePCA) {
		int ftrNum = (int)ftrs.size();
		int sampleNum = ftrs[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * params.PCACount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(ftrs);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, params.PCACount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, params.PCACount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, params.PCACount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Ffeatures, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(ftrs, Ffeatures, &arenas[ThreadIndex]);
	}
	if (Ffeatures.size() != filter.size()) {
		Critical("Tracker::CalculateResponse: feature channels not match filter channels.");
		return;
	}

	int len = (int)Ffeatures.size();
	MatCF respf(filter[0].Rows(), filter[0].Cols(), &arenas[ThreadIndex]);
	MatCF tempf(&arenas[ThreadIndex]);
	if (params.UseChannelWeights) {
		for (int i = 0; i < len; ++i) {
			GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
			respf.CAdd(tempf, filterWeights[i], ComplexF(0.0f, 0.0f));
		}
	} else {
		for (int i = 0; i < len; ++i) {
			GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
			respf.CAdd(tempf, 1.0f, ComplexF(0.0f, 0.0f));
		}
	}
	GFFT.FFTInv2(respf, response);
}

void Tracker::UpdateFilter(const Mat& image, const MatF& mask) {
	// Get ROI.
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(image, orgPatch, objectCenter, (int)(currentScaleFactor * templateSize.x),
		(int)(currentScaleFactor * templateSize.y));
	orgPatch.Resize(patch, rescaledTemplateSize.y, rescaledTemplateSize.x);

	// Extract features.
	std::vector<MatF> ftrs;
	GetFeatures(patch, Vector2i(yf.Cols(), yf.Rows()), ftrs, &arenas[ThreadIndex]);
	std::vector<MatCF> Fftrs;
	if (params.UsePCA) {
		int ftrNum = (int)ftrs.size();
		int sampleNum = ftrs[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * params.PCACount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(ftrs);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, params.PCACount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, params.PCACount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, params.PCACount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(ftrs, Fftrs, &arenas[ThreadIndex]);
	}

	std::vector<MatCF> newFilters;
	CreateFilters(Fftrs, yf, mask, newFilters, &arenas[ThreadIndex]);

	// Calculate per channel weights.
	int len = (int)newFilters.size();
	if (params.UseChannelWeights) {
		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			newFilterWeights[index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += newFilterWeights[i];

		// Update filter weights with new values.
		float sumUpdate = 0.0f;
		for (int i = 0; i < len; ++i) {
			filterWeights[i] = Lerp(params.WeightsLearnRate, filterWeights[i], newFilterWeights[i] / sumWeights);
			sumUpdate += filterWeights[i];
		}
		// Normalize weights.
		for (int i = 0; i < len; ++i)
			filterWeights[i] /= sumUpdate;
	}

	// Update filters.
	ParallelFor([&](int64_t i) {
		filters[i].Lerp(newFilters[i], params.FilterLearnRate);
	}, len, 2);
}

void Tracker::GetLocationPrior(const Vector2i& targetSize, const Vector2i& imgSize, MatF& prior) const {
	int ksize = std::min(targetSize.x, targetSize.y);
	float cx = (imgSize.x - 1) / 2.0f;
	float cy = (imgSize.y - 1) / 2.0f;
	float kernelstride = sqrt(2.0f) / ksize;
	prior.Reshape(imgSize.y, imgSize.x, false);

	ParallelFor([&](int64_t y) {
		float *pd = prior.Data() + y * imgSize.x;
		float tmpy = std::pow((cy - y) * kernelstride, 2);
		for (int x = 0; x < imgSize.x; ++x) {
			float tmpx = std::pow((cx - x) * kernelstride, 2);
			pd[x] = KernelEpan(tmpx + tmpy);
		}
	}, imgSize.y, 32);

	float maxValue;
	Vector2i maxLoc;
	prior.MaxLoc(maxValue, maxLoc);

	float *pdata = prior.Data();
	ParallelFor([&](int64_t index) {
		pdata[index] /= maxValue;
		if (pdata[index] < 0.5f) pdata[index] = 0.5f;
		else if (pdata[index] > 0.9f) pdata[index] = 0.9f;
	}, prior.Size(), 2048);
}

void Tracker::ExtractHistograms(const Mat& image, const Bounds2i& bb, Histogram& hf, Histogram& hb) {
	Vector2i vd = bb.Diagonal();
	int offsetX = (int)(vd.x / params.BackgroundRatio);
	int offsetY = (int)(vd.y / params.BackgroundRatio);
	Bounds2i outbb;
	outbb.pMin.x = std::max(0, bb.pMin.x - offsetX);
	outbb.pMin.y = std::max(0, bb.pMin.y - offsetY);
	outbb.pMax.x = std::min(image.Cols(), bb.pMax.x + offsetX);
	outbb.pMax.y = std::min(image.Rows(), bb.pMax.y + offsetY);

	// Calculate probability for the foreground.
	pFore = (float)bb.Area() / outbb.Area();
	hf.ExtractForeHist(image, bb);
	hb.ExtractBackHist(image, bb, outbb);
}

void Tracker::UpdateHistograms(const Mat& image, const Bounds2i& bb) {
	// Create temporary histograms.
	Histogram hf(image.Channels(), params.HistogramBins, &arenas[ThreadIndex]);
	Histogram hb(image.Channels(), params.HistogramBins, &arenas[ThreadIndex]);
	ExtractHistograms(image, bb, hf, hb);

	const float *pnewhf = hf.Data();
	const float *pnewhb = hb.Data();
	float *phf = histFore.Data();
	float *phb = histBack.Data();
	int len = histFore.Size();

	// Update using histogram learning rate.
	for (int i = 0; i < len; ++i) {
		*phf = Lerp(params.HistLearnRate, *phf, *pnewhf);
		*phb = Lerp(params.HistLearnRate, *phb, *pnewhb);
		++phf; ++pnewhf;
		++phb; ++pnewhb;
	}
}

void Tracker::SegmentRegion(const Mat& image, const Vector2i& objCenter,
	const Vector2i& tempSize, const Vector2i& tarSize, float scaleFactor, MatF& mask) const {
	// Calculate posterior of foreground and background.
	Vector2i scaledTemp = tempSize * scaleFactor;
	Vector2i scaledTarget = tarSize * scaleFactor;
	Bounds2i validPixels;
	Mat patch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(image, patch, objCenter, scaledTemp.x, scaledTemp.y, &validPixels);
	MatF fgPrior(&arenas[ThreadIndex]);
	GetLocationPrior(scaledTarget, scaledTemp, fgPrior);
	MatF fgPost(&arenas[ThreadIndex]), bgPost(&arenas[ThreadIndex]);
	GSegment.ComputePosteriors(patch, fgPost, bgPost, pFore, fgPrior, histFore, histBack,
		params.PostRegularCount, params.MaxSegmentArea);

	// Copy valid region.
	int rows = fgPost.Rows();
	int cols = fgPost.Cols();
	mask.Reshape(rows, cols);
	const float *psrc = fgPost.Data();
	float *pdest = mask.Data();
	int offsetX = validPixels.pMin.x;
	int offsetY = validPixels.pMin.y;
	Vector2i vd = validPixels.Diagonal();
	ParallelFor([&](int64_t y) {
		y = (y + offsetY) * cols + offsetX;
		memcpy(pdest + y, psrc + y, vd.x * sizeof(float));
	}, vd.y, 32);

	// Binary threshold.
	float maxValue;
	Vector2i maxLoc;
	mask.MaxLoc(maxValue, maxLoc);
	maxValue /= 2.0f;
	ParallelFor([&](int64_t index) {
		if (pdest[index] > maxValue) pdest[index] = 1.0f;
		else pdest[index] = 0.0f;
	}, mask.Size(), 2048);
}

Vector2i Tracker::EstimateNewPos(const Mat& image, float &score) const {
	MatF response(&arenas[ThreadIndex]);
	CalculateResponse(image, filters, response);
	float maxValue;
	Vector2i maxLoc;
	response.MaxLoc(maxValue, maxLoc);

	// Take into account also subpixel accuracy.
	float x = maxLoc.x + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Horizontal);
	float y = maxLoc.y + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Vertical);
	if (x > (response.Cols() - 1) / 2.0f) x -= response.Cols();
	if (y > (response.Rows() - 1) / 2.0f) y -= response.Rows();

	// Calculate new position.
	float scale = currentScaleFactor * cellSize / rescaleRatio;
	Vector2i newCenter = objectCenter + Vector2i((int)std::round(scale * x), (int)std::round(scale * y));

	// Fit to bound.
	newCenter.x = Clamp(newCenter.x, 0, imageSize.x - 1);
	newCenter.y = Clamp(newCenter.y, 0, imageSize.y - 1);
	
	// Calculate PSR as score (Peak to Sidelobe Ratio)
	int respCenterX = response.Cols() / 2;
	int respCenterY = response.Rows() / 2;
	int offsetX = respCenterX - maxLoc.x;
	int offsetY = respCenterY - maxLoc.y;
	MatF centeredResponse(&arenas[ThreadIndex]);
	GFilter.CircShift(response, centeredResponse, offsetX, offsetY);
	int peakRadius = (int)std::min(response.Cols() * params.PeakRatio, response.Rows() * params.PeakRatio);
	int left = respCenterX - peakRadius;
	int right = respCenterX + peakRadius;
	int top = respCenterY - peakRadius;
	int bottom = respCenterY + peakRadius;
	float mean = 0.0f;
	float sig = 0.0f;
	int count = 0;
	const float *pdata = centeredResponse.Data();
	for (int yidx = 0; yidx < centeredResponse.Rows(); ++yidx) {
		for (int xidx = 0; xidx < centeredResponse.Cols(); ++xidx) {
			if (yidx < top || yidx > bottom || xidx < left || xidx > right) {
				++count;
				mean += *pdata;
			}
			++pdata;
		}
	}
	if(count <= 0) {
		Critical("Tracker::EstimateNewPos: Invalid peak ratio parameter. Peak area is larger than the response area.");
	}
	mean /= count;
	pdata = centeredResponse.Data();
	for (int yidx = 0; yidx < centeredResponse.Rows(); ++yidx) {
		for (int xidx = 0; xidx < centeredResponse.Cols(); ++xidx) {
			if (yidx < top || yidx > bottom || xidx < left || xidx > right) {
				sig += (*pdata - mean) * (*pdata - mean);
			}
			++pdata;
		}
	}
	sig = sqrt(sig / count);
	score = (maxValue - mean) / sig;

	// Return new position.
	return newCenter;
}

bool Tracker::CheckMaskArea(const MatF& mat, int area) const {
	float threshold = 0.05f;
	float maskArea = mat.Mean() * mat.Size();
	return maskArea >= threshold * area;
}

//
// Tracker Main Interface
//

void Tracker::SetReinitialize() {
	if (params.UpdateInterval > 0) {
		updateCounter = 0;
		backgroundUpdateFlag = false;
		BackgroundWait();
	}
	dsst.SetReinitialize();
	initialized = false;
}

void Tracker::SetInitialMask(const MatF& mask) {
	presetMask = mask;
}

void Tracker::Initialize(const Mat& image, const Bounds2i& bb) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];

	// Input check.
	if (image.Size() == 0 || bb.Area() == 0) {
		Critical("Tracker::Init: cannot initialize with empty image data or empty bounding box.");
		return;
	}
	if (image.Channels() != 3) {
		Critical("Tracker::Init: must initialize with 3 channel RGB image.");
		return;
	}

	// Initialize members.
	currentScaleFactor = 1.0f;
	imageSize.x = image.Cols();
	imageSize.y = image.Rows();
	objbb = bb;
	orgTargetSize = objbb.Diagonal();
	cellSize = Clamp((int)std::ceil(orgTargetSize.x * orgTargetSize.y / 400.0f), 1, 4);

	int pad = (int)(params.Padding * sqrt(orgTargetSize.x * orgTargetSize.y));
	templateSize.x = orgTargetSize.x + pad;
	templateSize.y = orgTargetSize.y + pad;
	int squareSize = (templateSize.x + templateSize.y) / 2;
	templateSize.x = templateSize.y = squareSize;

	rescaleRatio = params.TemplateSize / (float)templateSize.x;
	if (rescaleRatio > 1.0f) rescaleRatio = 1.0f;
	rescaledTemplateSize.x = (int)std::round(templateSize.x * rescaleRatio);
	rescaledTemplateSize.y = (int)std::round(templateSize.y * rescaleRatio);
	//rescaledTemplateSize = templateSize * rescaleRatio;

	objectCenter.x = objbb.pMin.x + orgTargetSize.x / 2;
	objectCenter.y = objbb.pMin.y + orgTargetSize.y / 2;

	// Get standard gaussian response.
	GFilter.GaussianShapedLabels(yf, params.GaussianSigma, rescaledTemplateSize.x / cellSize, rescaledTemplateSize.y / cellSize);
	const int rows = yf.Rows();
	const int cols = yf.Cols();

	// Get filter window function.
	const Vector2i windowSize(cols, rows);
	switch (params.WindowFunc) {
	case WindowType::Hann:
		GFilter.HannWin(window, windowSize);
		break;
	case WindowType::Cheb:
		GFilter.ChebyshevWin(window, windowSize, params.ChebAttenuation);
		break;
	case WindowType::Kaiser:
		GFilter.KaiserWin(window, windowSize, params.KaiserAlpha);
		break;
	default:
		Critical("Tracker::Init: unknown window function type.");
		return;
	}

	// Set dummy mask and area;
	const Vector2i scaledObjSize = orgTargetSize * (rescaleRatio / cellSize);
	int x0 = std::max((cols - scaledObjSize.x) / 2, 0);
	int y0 = std::max((rows - scaledObjSize.y) / 2, 0);
	int x1 = std::min(x0 + scaledObjSize.x, cols);
	int y1 = std::min(y0 + scaledObjSize.y, rows);
	defaultMask.Reshape(rows, cols);
	ParallelFor([&](int64_t y) {
		y += y0;
		float *pd = defaultMask.Data() + y * cols + x0;
		for (int x = x0; x < x1; ++x)
			(*pd++) = 1.0f;
	}, y1 - y0, 32);
	defaultMaskArea = (int)(defaultMask.Mean() * defaultMask.Size());

	// Initialize segmentation.
	if (params.UseSegmentation) {
		Mat hsvImg(&arenas[ThreadIndex]);
		image.RGBToHSV(hsvImg);
		histFore.Reshape(hsvImg.Channels(), params.HistogramBins);
		histBack.Reshape(hsvImg.Channels(), params.HistogramBins);
		ExtractHistograms(hsvImg, objbb, histFore, histBack);
		SegmentRegion(hsvImg, objectCenter, templateSize, orgTargetSize, currentScaleFactor, filterMask);
		// Update calculated mask with preset mask.
		if (presetMask.Size() != 0) {
			int srows = filterMask.Rows();
			int scols = filterMask.Cols();
			int sx0 = std::max((scols - presetMask.Cols()) / 2, 0);
			int sy0 = std::max((srows - presetMask.Rows()) / 2, 0);
			int sx1 = std::min(sx0 + presetMask.Cols(), scols);
			int sy1 = std::min(sy0 + presetMask.Rows(), srows);
			int sw = sx1 - sx0;
			MatF presetMaskPadded(srows, scols, &arenas[ThreadIndex]);
			ParallelFor([&](int64_t y) {
				const float *psrc = presetMask.Data() + y * presetMask.Cols();
				float *pdest = presetMaskPadded.Data() + (y + sy0) * scols + sx0;
				memcpy(pdest, psrc, sw * sizeof(float));
			}, sy1 - sy0, 32);
			filterMask.CMul(presetMaskPadded);
		}

		// Manual spefify the ellipse shaped erode kernel.
		erodeKernel.Reshape(3, 3);
		erodeKernel.Data()[1] = 1.0f;
		erodeKernel.Data()[3] = 1.0f;
		erodeKernel.Data()[4] = 1.0f;
		erodeKernel.Data()[5] = 1.0f;
		erodeKernel.Data()[7] = 1.0f;

		MatF resizedFilterMask(&arenas[ThreadIndex]);
		filterMask.Resize(resizedFilterMask, rows, cols);
		if (CheckMaskArea(resizedFilterMask, defaultMaskArea))
			GFilter.Dilate2D(resizedFilterMask, erodeKernel, filterMask, BorderType::Zero);
		else
			filterMask = defaultMask;
	} else {
		filterMask = defaultMask;
	}

	// Initialize filter.
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(image, orgPatch, objectCenter, (int)(currentScaleFactor * templateSize.x),
		(int)(currentScaleFactor * templateSize.y));
	orgPatch.Resize(patch, rescaledTemplateSize.y, rescaledTemplateSize.x);
	std::vector<MatF> ftrs;
	GetFeatures(patch, Vector2i(cols, rows), ftrs, &arenas[ThreadIndex]);
	std::vector<MatCF> Fftrs;

	if (params.PCACount > (int)ftrs.size()) {
		Warning("PCA count is greater than original feature count. Disable PCA.");
		params.UsePCA = false;
	}
	if (params.UsePCA) {
		// PCA features.
		int ftrNum = (int)ftrs.size();
		int sampleNum = ftrs[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF centerdFtrs(sampleNum, ftrNum, &arenas[ThreadIndex]);
		MatF cov(ftrNum, ftrNum, &arenas[ThreadIndex]);
		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * params.PCACount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(ftrs);
		project.Reshape(ftrNum, params.PCACount, false);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egCenteredM(centerdFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egCovM(cov.Data(), ftrNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, params.PCACount);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, params.PCACount);
		egCenteredM = egDataM.rowwise() - egDataM.colwise().mean();
		egCovM.noalias() = egCenteredM.transpose() * egCenteredM;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(egCovM);
		if (eig.info() != Eigen::Success) {
			Critical("Cannot solve eigen values for PCA process.");
			return;
		}
		egProjM = eig.eigenvectors().rightCols(params.PCACount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, params.PCACount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(ftrs, Fftrs, &arenas[ThreadIndex]);
	}

	CreateFilters(Fftrs, yf, filterMask, filters);

	// Initialize channel weights.
	if (params.UseChannelWeights) {
		int len = (int)filters.size();
		filterWeights.resize(len, 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], filters[index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			filterWeights[index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += filterWeights[i];
		for (int i = 0; i < len; ++i)
			filterWeights[i] /= sumWeights;
	}

	// Initialize DSST.
	dsst.Initialize(image, objbb, params.ScaleCount, params.ScaleStep, params.ScaleSigma,
		params.ScaleLearnRate, params.ScaleMaxArea, templateSize);

	ResetArenas(false);
	ResetArenas(true);
}

bool Tracker::Update(const Mat& image, Bounds2i& bb, float &score) {
	if (!initialized) {
		Error("Tracker::Update: tracker is not initialized.");
		return false;
	}

	// Input check.
	if (image.Size() == 0) {
		Critical("Tracker::Init: cannot initialize with empty image data.");
		return false;
	}
	if (image.Channels() != 3) {
		Critical("Tracker::Init: must initialize with 3 channel RGB image.");
		return false;
	}

	// Get translation.
	objectCenter = EstimateNewPos(image, score);
	// Get scale.
	currentScaleFactor = dsst.GetScale(image, objectCenter);

	// Update bounding box.
	Vector2i newSize = currentScaleFactor * orgTargetSize;
	objbb.pMin.x = std::max(objectCenter.x - newSize.x / 2, 0);
	objbb.pMin.y = std::max(objectCenter.y - newSize.y / 2, 0);
	objbb.pMax.x = std::min(objbb.pMin.x + newSize.x, image.Cols());
	objbb.pMax.y = std::min(objbb.pMin.y + newSize.y, image.Rows());

	// Update tracker.
	if(params.UpdateInterval > 0) {
		++updateCounter;
		if (updateCounter >= params.UpdateInterval) {
			updateCounter = 0;
			BackgroundWait();
			FetchUpdateResult();
			StartBackgroundUpdate(image);
			BackgroundProcess([&]() {
				BackgroundUpdate();
			});
		}
	} else {
		if (params.UseSegmentation) {
			Mat hsvImg(&arenas[ThreadIndex]);
			image.RGBToHSV(hsvImg);
			UpdateHistograms(hsvImg, objbb);
			SegmentRegion(hsvImg, objectCenter, templateSize, orgTargetSize, currentScaleFactor, filterMask);
			MatF resizedFilterMask(&arenas[ThreadIndex]);
			filterMask.Resize(resizedFilterMask, yf.Rows(), yf.Cols());
			if (CheckMaskArea(resizedFilterMask, defaultMaskArea))
				GFilter.Dilate2D(resizedFilterMask, erodeKernel, filterMask, BorderType::Zero);
			else
				filterMask = defaultMask;
		} else {
			filterMask = defaultMask;
		}
		UpdateFilter(image, filterMask);
		dsst.Update(image, objectCenter);
	}

	// Return result.
	bb = objbb;
	ResetArenas(false);
	return score > params.PSRThreshold;
}

void Tracker::StartBackgroundUpdate(const Mat& image) {
	if (backgroundUpdateFlag) return;
	bk_image = image;
	bk_center = objectCenter;
	bk_bb = objbb;
	bk_scale = currentScaleFactor;
	bk_filters = filters;
	bk_filterWeights = filterWeights;
	dsst.StartBackgroundUpdate();
	backgroundUpdateFlag = true;
}

void Tracker::FetchUpdateResult() {
	if (!backgroundUpdateFlag) return;
	filters = bk_filters;
	filterWeights = bk_filterWeights;
	dsst.FetchUpdateResult();
	backgroundUpdateFlag = false;
}

void Tracker::BackgroundUpdate() {
	if (params.UseSegmentation) {
		Mat hsvImg(&arenas[ThreadIndex]);
		bk_image.RGBToHSV(hsvImg);
		UpdateHistograms(hsvImg, bk_bb);
		SegmentRegion(hsvImg, bk_center, templateSize, orgTargetSize, bk_scale, filterMask);
		MatF resizedFilterMask(&arenas[ThreadIndex]);
		filterMask.Resize(resizedFilterMask, yf.Rows(), yf.Cols());
		if (CheckMaskArea(resizedFilterMask, defaultMaskArea))
			GFilter.Dilate2D(resizedFilterMask, erodeKernel, filterMask, BorderType::Zero);
		else
			filterMask = defaultMask;
	} else {
		filterMask = defaultMask;
	}

	//
	// Update filter.
	//

	// Get ROI.
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(bk_image, orgPatch, bk_center, (int)(bk_scale * templateSize.x),
		(int)(bk_scale * templateSize.y));
	orgPatch.Resize(patch, rescaledTemplateSize.y, rescaledTemplateSize.x);

	// Extract features.
	std::vector<MatF> ftrs;
	GetFeatures(patch, Vector2i(yf.Cols(), yf.Rows()), ftrs, &arenas[ThreadIndex]);
	std::vector<MatCF> Fftrs;
	if (params.UsePCA) {
		int ftrNum = (int)ftrs.size();
		int sampleNum = ftrs[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * params.PCACount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(ftrs);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, params.PCACount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, params.PCACount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, params.PCACount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(ftrs, Fftrs, &arenas[ThreadIndex]);
	}

	std::vector<MatCF> newFilters;
	CreateFilters(Fftrs, yf, filterMask, newFilters, &arenas[ThreadIndex]);

	// Calculate per channel weights.
	int len = (int)newFilters.size();
	if (params.UseChannelWeights) {
		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			newFilterWeights[index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += newFilterWeights[i];

		// Update filter weights with new values.
		float sumUpdate = 0.0f;
		for (int i = 0; i < len; ++i) {
			bk_filterWeights[i] = Lerp(params.WeightsLearnRate, bk_filterWeights[i], newFilterWeights[i] / sumWeights);
			sumUpdate += bk_filterWeights[i];
		}
		// Normalize weights.
		for (int i = 0; i < len; ++i)
			bk_filterWeights[i] /= sumUpdate;
	}

	// Update filters.
	ParallelFor([&](int64_t i) {
		bk_filters[i].Lerp(newFilters[i], params.FilterLearnRate);
	}, len, 2);

	// Update DSST.
	dsst.BackgroundUpdate(bk_image, bk_center);

	ResetArenas(true);
}

void Tracker::ResetArenas(bool background) const {
	if (!arenas) return;
	int len = MaxThreadIndex() - 1;
	if(background) {
		arenas[len].Reset();
	} else {
		for (int i = 0; i < len; ++i)
			arenas[i].Reset();
	}
}

}	// namespace CSRT