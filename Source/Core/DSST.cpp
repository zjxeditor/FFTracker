//
// Scale estimation using DSST method.
//

#include "DSST.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"
#include "Filter.h"
#include "FeaturesExtractor.h"

namespace CSRT {

void DSST::Initialize(const Mat &image, const Bounds2i &bb, int numberOfScales, float stepOfScale, float sigmaFactor, float scaleLearnRate,
	float maxArea, const Vector2i &templateSize, float scaleModelFactor) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];
	backgroundUpdateFlag = false;

	// Initialize members.
	scaleCount = numberOfScales;
	scaleStep = stepOfScale;
	learnRate = scaleLearnRate;
	maxModelArea = maxArea;
	orgTargetSize = bb.Diagonal();
	Vector2i objectCenter(bb.pMin.x + orgTargetSize.x / 2, bb.pMin.y + orgTargetSize.y / 2);
	currentScaleFactor = 1.0f;
	if (scaleCount % 2 == 0)
		++scaleCount;
	scaleSigma = sqrt((float)scaleCount) * sigmaFactor;
	minScaleFactor = pow(scaleStep, std::ceil(
		log(std::max(5.0f / templateSize.x, 5.0f / templateSize.y)) / log(scaleStep)));
	maxScaleFactor = pow(scaleStep, std::floor(
		log(std::min((float)image.Rows() / orgTargetSize.y, (float)image.Cols() / orgTargetSize.x)) / log(scaleStep)));

	// Calculate standard response, scale factors and scale window.
	gs.Reshape(1, scaleCount);
	scaleFactors.resize(scaleCount);
	for (int i = 0; i < scaleCount; ++i) {
		float ss = i - std::floor(scaleCount / 2.0f);
		gs.Data()[i] = exp(-0.5f * pow(ss, 2) / pow(scaleSigma, 2));
		scaleFactors[i] = pow(scaleStep, -ss);
	}
	GFilter.HannWin(scaleWindow, Vector2i(scaleCount, 1));

	// Calculate scale model size for this DSST.
	if (templateSize.x * templateSize.y * pow(scaleModelFactor, 2) > maxModelArea)
		scaleModelFactor = sqrt(maxModelArea / (templateSize.x * templateSize.y));
	scaleModelSize.x = (int)(templateSize.x * scaleModelFactor);
	scaleModelSize.y = (int)(templateSize.y * scaleModelFactor);

	// Get scale features.
	MatF scaleRepresent(&arenas[ThreadIndex]);
	GetScaleFeatures(image, scaleRepresent, objectCenter, orgTargetSize,
		currentScaleFactor, scaleFactors, scaleWindow, scaleModelSize);

	// Calculate the initial filter.
	MatCF gsfRow(&arenas[ThreadIndex]);
	GFFT.FFT2Row(gs, gsfRow);
	gsfRow.Repeat(gsf, scaleRepresent.Rows(), 1);
	MatCF srf(&arenas[ThreadIndex]);
	GFFT.FFT2Row(scaleRepresent, srf);
	GFilter.MulSpectrums(gsf, srf, hfNum, true);
	MatCF hfDenAll(&arenas[ThreadIndex]);
	GFilter.MulSpectrums(srf, srf, hfDenAll, true);
	hfDenAll.Reduce(hfDen, ReduceMode::Sum, true);

	ResetArenas(false);
	ResetArenas(true);
}

void DSST::GetScaleFeatures(const Mat& img, MatF& feat, const Vector2i& pos, const Vector2i& orgSize, float currentScale,
	const std::vector<float>& factors, const MatF& window, const Vector2i& modelSize) const {
	// Get first scale level features and do initialization work.
	Vector2i patchSize(std::floor(currentScale * factors[0] * orgSize.x),
		std::floor(currentScale * factors[0] * orgSize.y));
	Mat imgPatch(&arenas[ThreadIndex]);
	GFilter.GetSubWindow(img, imgPatch, pos, patchSize.x, patchSize.y);
	Mat resizePatch(&arenas[ThreadIndex]);
	imgPatch.Resize(resizePatch, modelSize.y, modelSize.x);
	std::vector<MatF> hogs;
	GFeatsExtractor.GetFeaturesHOG(resizePatch, hogs, 4, &arenas[ThreadIndex]);
	int flen = (int)hogs[0].Size();
	int fcount = (int)hogs.size();
	MatF featT((int)factors.size(), flen * fcount, &arenas[ThreadIndex]);
	ParallelFor([&](int64_t c) {
		hogs[c].ReValue(window.Data()[0], 0.0f);
		memcpy(featT.Data() + c * flen, hogs[c].Data(), flen * sizeof(float));
	}, fcount, 4);

	// Get other scale level features in parallel.
	ParallelFor([&](int64_t y) {
		++y;
		Vector2i scalePatchSize(std::floor(currentScale * factors[y] * orgSize.x),
			std::floor(currentScale * factors[y] * orgSize.y));
		Mat scaleImgPatch(&arenas[ThreadIndex]);
		GFilter.GetSubWindow(img, scaleImgPatch, pos, scalePatchSize.x, scalePatchSize.y);
		Mat scaleResizePatch(&arenas[ThreadIndex]);
		scaleImgPatch.Resize(scaleResizePatch, modelSize.y, modelSize.x);
		std::vector<MatF> scaleHogs;
		GFeatsExtractor.GetFeaturesHOG(scaleResizePatch, scaleHogs, 4, &arenas[ThreadIndex]);

		float *pd = featT.Data() + y * featT.Cols();
		for (int i = 0; i < fcount; ++i) {
			scaleHogs[i].ReValue(window.Data()[y], 0.0f);
			memcpy(pd + i * flen, scaleHogs[i].Data(), flen * sizeof(float));
		}
	}, factors.size() - 1, 2);

	// Transpose the feat matrix.
	GFFT.Transpose(featT, feat);
}

void DSST::ResetArenas(bool background) const {
	if (!arenas) return;
	int len = MaxThreadIndex() - 1;
	if(background) {
		arenas[len].Reset();
	} else {
		for (int i = 0; i < len; ++i)
			arenas[i].Reset();
	}
}

float DSST::GetScale(const Mat& image, const Vector2i& objectCenter) {
	if (!initialized) {
		Error("DSST::GetScale: DSST is not initialized.");
		return 0.0f;
	}

	// Get image features around the previoud target location.
	MatF scaleFeatures(&arenas[ThreadIndex]);
	GetScaleFeatures(image, scaleFeatures, objectCenter, orgTargetSize,
		currentScaleFactor, scaleFactors, scaleWindow, scaleModelSize);
	// Calculate reponse for the previoud filter.
	MatCF srf(&arenas[ThreadIndex]), tempf(&arenas[ThreadIndex]);
	GFFT.FFT2Row(scaleFeatures, srf);
	GFilter.MulSpectrums(srf, hfNum, tempf, false);
	MatCF scaleRepresent(&arenas[ThreadIndex]), respf(&arenas[ThreadIndex]);
	tempf.Reduce(scaleRepresent, ReduceMode::Sum, true);
	GFilter.DivideComplexMatrices(scaleRepresent, hfDen, respf, ComplexF(0.01f, 0.01f));
	MatF resp(&arenas[ThreadIndex]);
	GFFT.FFTInv2Row(respf, resp, true);

	// Find max reponse location.
	float maxValue = 0.0f;
	Vector2i maxLoc;
	resp.MaxLoc(maxValue, maxLoc);

	// Update the current scale factor.
	currentScaleFactor *= scaleFactors[maxLoc.x];
	if (currentScaleFactor < minScaleFactor)
		currentScaleFactor = minScaleFactor;
	else if (currentScaleFactor > maxScaleFactor)
		currentScaleFactor = maxScaleFactor;

	ResetArenas(false);
	return currentScaleFactor;
}

void DSST::Update(const Mat& image, const Vector2i& objectCenter) {
	if (!initialized) {
		Error("DSST::Update: DSST is not initialized.");
		return;
	}

	// Get image features around the target location.
	MatF scaleFeatures(&arenas[ThreadIndex]);
	GetScaleFeatures(image, scaleFeatures, objectCenter, orgTargetSize,
		currentScaleFactor, scaleFactors, scaleWindow, scaleModelSize);
	MatCF srf(&arenas[ThreadIndex]);
	GFFT.FFT2Row(scaleFeatures, srf);

	// Calculate new filter model.
	MatCF newHfNum(&arenas[ThreadIndex]), newHfDen(&arenas[ThreadIndex]), newHfDenAll(&arenas[ThreadIndex]);
	GFilter.MulSpectrums(gsf, srf, newHfNum, true);
	GFilter.MulSpectrums(srf, srf, newHfDenAll, true);
	newHfDenAll.Reduce(newHfDen, ReduceMode::Sum, true);

	// Lerp by learn rate.
	hfNum.Lerp(newHfNum, learnRate);
	hfDen.Lerp(newHfDen, learnRate);

	ResetArenas(false);
}

void DSST::StartBackgroundUpdate() {
	if (backgroundUpdateFlag) return;
	bk_scale = currentScaleFactor;
	bk_hfNum = hfNum;
	bk_hfDen = hfDen;
	backgroundUpdateFlag = true;
}

void DSST::FetchUpdateResult() {
	if (!backgroundUpdateFlag) return;
	hfNum = bk_hfNum;
	hfDen = bk_hfDen;
	backgroundUpdateFlag = false;
}

void DSST::BackgroundUpdate(const Mat& image, const Vector2i& objectCenter) {
	if (!initialized) {
		Error("DSST::BackgroundUpdate: DSST is not initialized.");
		return;
	}

	// Get image features around the target location.
	MatF scaleFeatures(&arenas[ThreadIndex]);
	GetScaleFeatures(image, scaleFeatures, objectCenter, orgTargetSize,
		bk_scale, scaleFactors, scaleWindow, scaleModelSize);
	MatCF srf(&arenas[ThreadIndex]);
	GFFT.FFT2Row(scaleFeatures, srf);

	// Calculate new filter model.
	MatCF newHfNum(&arenas[ThreadIndex]), newHfDen(&arenas[ThreadIndex]), newHfDenAll(&arenas[ThreadIndex]);
	GFilter.MulSpectrums(gsf, srf, newHfNum, true);
	GFilter.MulSpectrums(srf, srf, newHfDenAll, true);
	newHfDenAll.Reduce(newHfDen, ReduceMode::Sum, true);

	// Lerp by learn rate.
	bk_hfNum.Lerp(newHfNum, learnRate);
	bk_hfDen.Lerp(newHfDen, learnRate);

	ResetArenas(true);
}

}	// namespace CSRT