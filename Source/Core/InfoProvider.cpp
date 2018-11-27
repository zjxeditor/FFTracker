//
// Provide features from various kinds of image to the tracking system. It can be customized for different features configuration.
//

#include "InfoProvider.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"
#include "Filter.h"
#include "FeaturesExtractor.h"

namespace CSRT {

using namespace std;

InfoProvider GInfoProvider;

// Feature memory arena management.
static MemoryArena *GInfoArenas = nullptr;

bool InfoProvider::SingleInstance = false;

InfoProvider::InfoProvider() :
	initialized(false),
	targetCount(0),
	cellSize(0),
	rescaleRatio(0.0f),
	scaleCount(0),
	backgroundRatio(0.0f),
	histLearnRate(0.0f),
	postRegularCount(0),
	maxSegmentArea(0.0f) {
	if (SingleInstance) {
		Critical("Only one \"InfoProvider\" can exists.");
		return;
	}
	SingleInstance = true;
}

InfoProvider::~InfoProvider() {
	if (GInfoArenas) {
		delete[] GInfoArenas;
		GInfoArenas = nullptr;
	}
	SingleInstance = false;
}

void InfoProvider::Initialize() {
	if (initialized) return;
	initialized = true;
	GInfoArenas = new MemoryArena[MaxThreadIndex()];
}

void InfoProvider::ResetArenas() const {
	if (!GInfoArenas) return;
	int len = MaxThreadIndex();
	for (int i = 0; i < len; ++i)
		GInfoArenas[i].Reset();
}

//
// Features used for DSST
//

//void InfoProvider::GetScaleFeatures(
//	const Mat& img,
//	MatF& feat,
//	const Vector2i& pos,
//	const Vector2i& orgSize,
//	float currentScale,
//	const std::vector<float>& factors,
//	const MatF& window,
//	const Vector2i& modelSize) const {
//	if (factors.size() != window.Cols() || window.Rows() != 1) {
//		Critical("InfoProvider::GetScaleFeatures: window and factor sizes are not match.");
//		return;
//	}
//	// Get first scale level features and do initialization work.
//	Vector2i patchSize(std::floor(currentScale * factors[0] * orgSize.x),
//					   std::floor(currentScale * factors[0] * orgSize.y));
//	Mat imgPatch(&GInfoArenas[ThreadIndex]);
//	GFilter.GetSubWindow(img, imgPatch, pos, patchSize.x, patchSize.y);
//	MatF imgPatchF(&GInfoArenas[ThreadIndex]);
//	imgPatch.ToMatF(imgPatchF);
//	MatF resizePatchF(&GInfoArenas[ThreadIndex]);
//	imgPatchF.Resize(resizePatchF, modelSize.y, modelSize.x, 3);
//	resizePatchF.ReValue(1.0f / 255.0f, 0.0f);
//	std::vector<MatF> hogs;
//	GFeatsExtractor.GetFeaturesHOG(resizePatchF, hogs, 4, &GInfoArenas[ThreadIndex]);
//	int flen = (int) hogs[0].Size();
//	int fcount = (int) hogs.size();
//	MatF featT((int) factors.size(), flen * fcount, &GInfoArenas[ThreadIndex]);
//	ParallelFor([&](int64_t c) {
//		hogs[c].ReValue(window.Data()[0], 0.0f);
//		memcpy(featT.Data() + c * flen, hogs[c].Data(), flen * sizeof(float));
//	}, fcount, 4);
//
//	// Get other scale level features in parallel.
//	ParallelFor([&](int64_t y) {
//		++y;
//		Vector2i scalePatchSize(std::floor(currentScale * factors[y] * orgSize.x),
//								std::floor(currentScale * factors[y] * orgSize.y));
//		Mat scaleImgPatch(&GInfoArenas[ThreadIndex]);
//		GFilter.GetSubWindow(img, scaleImgPatch, pos, scalePatchSize.x, scalePatchSize.y);
//		MatF scaleImgPatchF(&GInfoArenas[ThreadIndex]);
//		scaleImgPatch.ToMatF(scaleImgPatchF);
//		MatF scaleResizePatchF(&GInfoArenas[ThreadIndex]);
//		scaleImgPatchF.Resize(scaleResizePatchF, modelSize.y, modelSize.x, 3);
//		scaleResizePatchF.ReValue(1.0f / 255.0f, 0.0f);
//		std::vector<MatF> scaleHogs;
//		GFeatsExtractor.GetFeaturesHOG(scaleResizePatchF, scaleHogs, 4, &GInfoArenas[ThreadIndex]);
//		float *pd = featT.Data() + y * featT.Cols();
//		for (int i = 0; i < fcount; ++i) {
//			scaleHogs[i].ReValue(window.Data()[y], 0.0f);
//			memcpy(pd + i * flen, scaleHogs[i].Data(), flen * sizeof(float));
//		}
//	}, factors.size() - 1, 2);
//
//	// Transpose the feat matrix.
//	GFFT.Transpose(featT, feat);
//	ResetArenas();
//}

void InfoProvider::GetScaleFeatures(
    const Mat& img,
    MatF& feat,
    const Vector2i& pos,
    const Vector2i& orgSize,
    float currentScale,
    const std::vector<float>& factors,
    const MatF& window,
    const Vector2i& modelSize) const {
	if (factors.size() != window.Cols() || window.Rows() != 1) {
		Critical("InfoProvider::GetScaleFeatures: window and factor sizes are not match.");
		return;
	}
	// Get first scale level features and do initialization work.
	Vector2i patchSize(std::floor(currentScale * factors[0] * orgSize.x),
					   std::floor(currentScale * factors[0] * orgSize.y));
	Mat imgPatch(&GInfoArenas[ThreadIndex]);
	GFilter.GetSubWindow(img, imgPatch, pos, patchSize.x, patchSize.y);
	Mat resizePatch(&GInfoArenas[ThreadIndex]);
	imgPatch.Resize(resizePatch, modelSize.y, modelSize.x);
	std::vector<MatF> hogs;
	GFeatsExtractor.GetFeaturesHOG(resizePatch, hogs, 4, &GInfoArenas[ThreadIndex]);
	int flen = (int)hogs[0].Size();
	int fcount = (int)hogs.size();
	MatF featT((int)factors.size(), flen * fcount, &GInfoArenas[ThreadIndex]);
	ParallelFor([&](int64_t c) {
		hogs[c].ReValue(window.Data()[0], 0.0f);
		memcpy(featT.Data() + c * flen, hogs[c].Data(), flen * sizeof(float));
	}, fcount, 4);

	// Get other scale level features in parallel.
	ParallelFor([&](int64_t y) {
		++y;
		Vector2i scalePatchSize(std::floor(currentScale * factors[y] * orgSize.x),
								std::floor(currentScale * factors[y] * orgSize.y));
		Mat scaleImgPatch(&GInfoArenas[ThreadIndex]);
		GFilter.GetSubWindow(img, scaleImgPatch, pos, scalePatchSize.x, scalePatchSize.y);
		Mat scaleResizePatch(&GInfoArenas[ThreadIndex]);
		scaleImgPatch.Resize(scaleResizePatch, modelSize.y, modelSize.x);
		std::vector<MatF> scaleHogs;
		GFeatsExtractor.GetFeaturesHOG(scaleResizePatch, scaleHogs, 4, &GInfoArenas[ThreadIndex]);

		float *pd = featT.Data() + y * featT.Cols();
		for (int i = 0; i < fcount; ++i) {
			scaleHogs[i].ReValue(window.Data()[y], 0.0f);
			memcpy(pd + i * flen, scaleHogs[i].Data(), flen * sizeof(float));
		}
	}, factors.size() - 1, 2);

	// Transpose the feat matrix.
	GFFT.Transpose(featT, feat);
	ResetArenas();
}

void InfoProvider::GetScaleFeatures(
	const MatF& depth,
	const MatF& normal,
	MatF& feat,
	const Vector2i& pos,
	const Vector2i& orgSize,
	float currentScale,
	const std::vector<float>& factors,
	const MatF& window,
	const Vector2i& modelSize,
	bool useNormal) const {
	if (factors.size() != window.Cols() || window.Rows() != 1) {
		Critical("InfoProvider::GetScaleFeatures: window and factor sizes are not match.");
		return;
	}
	if (useNormal && (normal.Size() == 0 || normal.Rows() != depth.Rows() || normal.Cols() != depth.Cols() * 2)) {
		Critical("InfoProvider::GetScaleFeatures: invalid normal patch data.");
		return;
	}

	if(!useNormal) {
		// Get first scale level features and do initialization work.
		Vector2i patchSize(std::floor(currentScale * factors[0] * orgSize.x),
                           std::floor(currentScale * factors[0] * orgSize.y));
		MatF depthPatch(&GInfoArenas[ThreadIndex]);
		GFilter.GetSubWindow(depth, depthPatch, pos, patchSize.x, patchSize.y);
		MatF resizePatch(&GInfoArenas[ThreadIndex]);
		depthPatch.Resize(resizePatch, modelSize.y, modelSize.x);
		std::vector<MatF> hogs;
		GFeatsExtractor.GetDepthFeaturesHOG(resizePatch, hogs, 4, &GInfoArenas[ThreadIndex]);
		int flen = (int)hogs[0].Size();
		int fcount = (int)hogs.size();
		MatF featT((int)factors.size(), flen * fcount, &GInfoArenas[ThreadIndex]);
		ParallelFor([&](int64_t c) {
			hogs[c].ReValue(window.Data()[0], 0.0f);
			memcpy(featT.Data() + c * flen, hogs[c].Data(), flen * sizeof(float));
		}, fcount, 4);

		// Get other scale level features in parallel.
		ParallelFor([&](int64_t y) {
			++y;
			Vector2i scalePatchSize(std::floor(currentScale * factors[y] * orgSize.x),
                                    std::floor(currentScale * factors[y] * orgSize.y));
			MatF scaleDepthPatch(&GInfoArenas[ThreadIndex]);
			GFilter.GetSubWindow(depth, scaleDepthPatch, pos, scalePatchSize.x, scalePatchSize.y);
			MatF scaleResizePatch(&GInfoArenas[ThreadIndex]);
			scaleDepthPatch.Resize(scaleResizePatch, modelSize.y, modelSize.x);
			std::vector<MatF> scaleHogs;
			GFeatsExtractor.GetDepthFeaturesHOG(scaleResizePatch, scaleHogs, 4, &GInfoArenas[ThreadIndex]);

			float *pd = featT.Data() + y * featT.Cols();
			for (int i = 0; i < fcount; ++i) {
				scaleHogs[i].ReValue(window.Data()[y], 0.0f);
				memcpy(pd + i * flen, scaleHogs[i].Data(), flen * sizeof(float));
			}
		}, factors.size() - 1, 2);

		// Transpose the feat matrix.
		GFFT.Transpose(featT, feat);
		ResetArenas();
	} else {
		// Get first scale level features and do initialization work.
		Vector2i newPos(pos.x * 2, pos.y);
		Vector2i patchSize(std::floor(currentScale * factors[0] * orgSize.x),
                           std::floor(currentScale * factors[0] * orgSize.y));
		MatF normalPatch(&GInfoArenas[ThreadIndex]);
		GFilter.GetSubWindow(normal, normalPatch, newPos, patchSize.x / 2 * 4, patchSize.y);
		MatF resizePatch(&GInfoArenas[ThreadIndex]);
		normalPatch.Resize(resizePatch, modelSize.y, modelSize.x, 2);
		std::vector<MatF> hogs;
		GFeatsExtractor.GetDepthFeaturesNormalHOG(resizePatch, hogs, 4, &GInfoArenas[ThreadIndex]);
		int flen = (int)hogs[0].Size();
		int fcount = (int)hogs.size();
		MatF featT((int)factors.size(), flen * fcount, &GInfoArenas[ThreadIndex]);
		ParallelFor([&](int64_t c) {
			hogs[c].ReValue(window.Data()[0], 0.0f);
			memcpy(featT.Data() + c * flen, hogs[c].Data(), flen * sizeof(float));
		}, fcount, 4);

		// Get other scale level features in parallel.
		ParallelFor([&](int64_t y) {
			++y;
			Vector2i scalePatchSize(std::floor(currentScale * factors[y] * orgSize.x),
                                    std::floor(currentScale * factors[y] * orgSize.y));
			MatF scaleNormalPatch(&GInfoArenas[ThreadIndex]);
			GFilter.GetSubWindow(normal, scaleNormalPatch, newPos, scalePatchSize.x / 2 * 4, scalePatchSize.y);
			MatF scaleResizePatch(&GInfoArenas[ThreadIndex]);
			scaleNormalPatch.Resize(scaleResizePatch, modelSize.y, modelSize.x, 2);
			std::vector<MatF> scaleHogs;
			GFeatsExtractor.GetDepthFeaturesNormalHOG(scaleResizePatch, scaleHogs, 4, &GInfoArenas[ThreadIndex]);

			float *pd = featT.Data() + y * featT.Cols();
			for (int i = 0; i < fcount; ++i) {
				scaleHogs[i].ReValue(window.Data()[y], 0.0f);
				memcpy(pd + i * flen, scaleHogs[i].Data(), flen * sizeof(float));
			}
		}, factors.size() - 1, 2);

		// Transpose the feat matrix.
		GFFT.Transpose(featT, feat);
		ResetArenas();
	}
}

//
// Features used for tracker
//

void InfoProvider::GetTrackFeatures(
	const Mat& patch,
	const Vector2i& featureSize,
	std::vector<MatF>& feats,
	const MatF &window,
	uint8_t mask,
	int cellSize,
	int hogNum,
	MemoryArena* targetArena) const {
	if (hogNum > HOG_FEATURE_COUNT) {
		Critical("InfoProvider::GetTrackFeatures: invalid hog count.");
		return;
	}
	if (featureSize.x != window.Cols() || featureSize.y != window.Rows()) {
		Critical("InfoProvider::GetTrackFeatures: feature and window sizes are not match.");
		return;
	}
	if (patch.Rows() / cellSize != window.Rows() || patch.Cols() / cellSize != window.Cols()) {
		Critical("InfoProvider::GetTrackFeatures: feature and cell sizes are not match.");
		return;
	}

	feats.clear();
	if (mask & 1) {
		std::vector<MatF> hog;
		GFeatsExtractor.GetFeaturesHOG(patch, hog, cellSize, targetArena);
		if (hog[0].Cols() != featureSize.x || hog[0].Rows() != featureSize.y) {
			Critical("InfoProvider::GetTrackFeatures: hog feature size not match, please check the cell size.");
			return;
		}
		for (int i = 0; i < hogNum; ++i)
			feats.push_back(std::move(hog[i]));
	}
	if (mask & 2) {
		std::vector<MatF> cn;
		GFeatsExtractor.GetFeaturesCN(patch, cn, featureSize, targetArena);
		for (int i = 0; i < (int)cn.size(); ++i)
			feats.push_back(std::move(cn[i]));
	}
	if (mask & 4) {
		Mat gray(targetArena), grayResized(targetArena);
		patch.RGBToGray(gray);
		// Issue: maybe use bicubic interpolation
		gray.Resize(grayResized, featureSize.y, featureSize.x);
		MatF grayFeat(targetArena);
		grayResized.ToMatF(grayFeat, 1.0f / 255.0f, -0.5f);
		feats.push_back(std::move(grayFeat));
	}
	if (mask & 8) {
		std::vector<MatF> rgb;
		GFeatsExtractor.GetFeaturesRGB(patch, rgb, featureSize, targetArena);
		for (int i = 0; i < (int)rgb.size(); ++i)
			feats.push_back(std::move(rgb[i]));
	}

	ParallelFor([&](int64_t index) {
		feats[index].CMul(window);
	}, (int)feats.size(), 4);
}

void InfoProvider::GetTrackFeatures(
	const MatF& depthPatch,
	const MatF &normalPatch,
	const Vector2i& featureSize,
	std::vector<MatF>& feats,
	const MatF& window,
	uint8_t mask,
	int cellSize,
	int hogNum,
	MemoryArena* targetArena) const {
	if (hogNum > HOG_FEATURE_COUNT) {
		Critical("InfoProvider::GetTrackFeatures: invalid hog count.");
		return;
	}
	if (featureSize.x != window.Cols() || featureSize.y != window.Rows()) {
		Critical("InfoProvider::GetTrackFeatures: feature and window sizes are not match.");
		return;
	}
	if (depthPatch.Rows() / cellSize != window.Rows() || depthPatch.Cols() / cellSize != window.Cols()) {
		Critical("InfoProvider::GetTrackFeatures: feature and cell sizes are not match.");
		return;
	}
	if(normalPatch.Size() != 0 && (normalPatch.Rows() != depthPatch.Rows() || normalPatch.Cols() != depthPatch.Cols() * 2)) {
		Critical("InfoProvider::GetTrackFeatures: depth patch and normal patch not match.");
		return;
	}

	feats.clear();
	if (mask & 1) {
		std::vector<MatF> hog;
		GFeatsExtractor.GetDepthFeaturesHOG(depthPatch, hog, cellSize, targetArena);
		if (hog[0].Cols() != featureSize.x || hog[0].Rows() != featureSize.y) {
			Critical("InfoProvider::GetTrackFeatures: hog feature size not match, please check the cell size.");
			return;
		}
		for (int i = 0; i < hogNum; ++i)
			feats.push_back(std::move(hog[i]));
	}
	if (mask & 2) {
		MatF grayFeat(targetArena);
		depthPatch.Resize(grayFeat, featureSize.y, featureSize.x);
		feats.push_back(std::move(grayFeat));
	}
	if(normalPatch.Size() != 0 && (mask & 4)) {
		std::vector<MatF> polarNormals;
		GFeatsExtractor.GetDepthFeaturesPolarNormal(normalPatch, polarNormals, featureSize, targetArena);
		for (int i = 0; i < (int)polarNormals.size(); ++i)
			feats.push_back(std::move(polarNormals[i]));
	}
	if(normalPatch.Size() != 0 && (mask & 8)) {
		std::vector<MatF> hon;
		int phiBinNum = 4;
		int thetaBinNum = 5;
		GFeatsExtractor.GetDepthFeaturesHON(normalPatch, hon, cellSize, phiBinNum, thetaBinNum, targetArena);
		if (hon[0].Cols() != featureSize.x || hon[0].Rows() != featureSize.y) {
			Critical("InfoProvider::GetTrackFeatures: hon feature size not match, please check the cell size.");
			return;
		}
		for (int i = 0; i < hogNum; ++i)
			feats.push_back(std::move(hon[i]));
	}
	if(normalPatch.Size() != 0 && (mask & 16)) {
		std::vector<MatF> hog;
		GFeatsExtractor.GetDepthFeaturesNormalHOG(normalPatch, hog, cellSize, targetArena);
		if (hog[0].Cols() != featureSize.x || hog[0].Rows() != featureSize.y) {
			Critical("InfoProvider::GetTrackFeatures: hog feature size not match, please check the cell size.");
			return;
		}
		for (int i = 0; i < hogNum; ++i)
			feats.push_back(std::move(hog[i]));
	}

	ParallelFor([&](int64_t index) {
		feats[index].CMul(window);
	}, (int)feats.size(), 4);
}

void InfoProvider::ConfigureTargets(
	int count,
	const Vector2i &moveRegion,
	const std::vector<Bounds2i> &bbs) {
	if (count <= 0 || moveRegion.LengthSquared() <= 0.0f || bbs.size() < count) {
		Critical("InfoProvider::ConfigureTargets: invalid targets configuration.");
		return;
	}

	targetCount = count;
	moveSize = moveRegion;

    orgTargetSizes.resize(targetCount);
    currentBounds.resize(targetCount);
    currentPositions.resize(targetCount);
    currentScales.resize(targetCount);
    memcpy(&currentBounds[0], &bbs[0], targetCount * sizeof(Bounds2i));
    for (int i = 0; i < targetCount; ++i) {
        orgTargetSizes[i] = currentBounds[i].Diagonal();
		currentPositions[i] = currentBounds[i].pMin + orgTargetSizes[i] / 2;
        currentScales[i] = 1.0f;
    }

    cellSize = 0;
    for(int i = 0; i < targetCount; ++i)
        cellSize += Clamp((int)std::ceil(orgTargetSizes[i].x * orgTargetSizes[i].y / 400.0f), 1, 4);
    cellSize = (int)std::round((float)cellSize / targetCount);
	//cellSize = 4;
}

void InfoProvider::ConfigureTracker(
	float localPadding,
	int referTemplateSize,
	float gaussianSigma,
	bool channelWeights,
	int pca,
	int iteration,
	float learnRateOfChannel,
	float learnRateOfFilter,
	WindowType windowFunc,
	float chebAttenuation,
	float kaiserAlpha) {
	if (localPadding <= 0.0f || referTemplateSize <= 0 || gaussianSigma <= 0.0f || iteration <= 0) {
		Critical("InfoProvider::ConfigureTracker: invalid track configuration.");
		return;
	}

	// Calculate template size
    templateSize.x = 0;
	templateSize.y = 0;
	for(int i = 0; i < targetCount; ++i) {
        float pad = localPadding * sqrt(orgTargetSizes[i].x * orgTargetSizes[i].y);
        templateSize.x += (int)(orgTargetSizes[i].x + pad);
        templateSize.y += (int)(orgTargetSizes[i].y + pad);
	}
    templateSize.x = (int)std::round((float)templateSize.x / targetCount);
    templateSize.y = (int)std::round((float)templateSize.y / targetCount);
    int squareSize = (int)((templateSize.x + templateSize.y) / 2.0f);
    templateSize.x = templateSize.y = squareSize;

    // Calculate final rescaled template size
    rescaleRatio = sqrt(pow(referTemplateSize, 2.0f) / (templateSize.x * templateSize.y));
    if (rescaleRatio > 1.0f) rescaleRatio = 1.0f;
    rescaledTemplateSize.x = (int)(templateSize.x * rescaleRatio);
    rescaledTemplateSize.y = (int)(templateSize.y * rescaleRatio);

//    rescaledTemplateSize.x = referTemplateSize;
//    rescaledTemplateSize.y = referTemplateSize;
//    Vector2i tempSize;
//    for (int i = 0; i < targetCount; ++i)
//        tempSize += orgTargetSizes[i];
//    tempSize /= targetCount;
//    templateSize.x = templateSize.y = (int)((tempSize.x + tempSize.y) / 2 + localPadding * sqrt(tempSize.x * tempSize.y));
//    rescaleRatio = (float)referTemplateSize / templateSize.x;

	// Get standard gaussian response.
	GFilter.GaussianShapedLabels(yf, gaussianSigma, rescaledTemplateSize.x / cellSize, rescaledTemplateSize.y / cellSize);

	// Get filter window function.
	const Vector2i windowSize(yf.Cols(), yf.Rows());
	switch (windowFunc) {
	case WindowType::Hann:
		GFilter.HannWin(window, windowSize);
		break;
	case WindowType::Cheb:
		GFilter.ChebyshevWin(window, windowSize, chebAttenuation);
		break;
	case WindowType::Kaiser:
		GFilter.KaiserWin(window, windowSize, kaiserAlpha);
		break;
	default:
		Critical("Tracker::Init: unknown window function type.");
		return;
	}

	// Create trackers.
	trackers.resize(targetCount, Tracker(channelWeights, pca, iteration, learnRateOfChannel, learnRateOfFilter));
}

void InfoProvider::ConfigureDSST(
	int numberOfScales,
	float stepOfScale,
	float sigmaFactor,
	float learnRateOfScale,
	float maxArea) {
	if(numberOfScales <= 0 || stepOfScale <= 0.0f) {
		Critical("InfoProvider::ConfigureDSST: invalid dsst configuration.");
		return;
	}

	scaleCount = numberOfScales;
	if (scaleCount % 2 == 0)	// Scale count must be odd number.
		++scaleCount;

	// Calculate minimum and maximum scale factors.
	scaleMinFactors.resize(targetCount);
	scaleMaxFactors.resize(targetCount);
	float minFactor = pow(stepOfScale, std::ceil(
		log(std::max(5.0f / templateSize.x, 5.0f / templateSize.y)) / log(stepOfScale)));
	for (int i = 0; i < targetCount; ++i) {
		scaleMinFactors[i] = minFactor;
		scaleMaxFactors[i] = pow(stepOfScale, std::floor(
			log(std::min((float)moveSize.y / orgTargetSizes[i].y, (float)moveSize.x / orgTargetSizes[i].x)) / log(stepOfScale)));
    }

	// Calculate standard response, scale factors and scale window.
	float scaleSigma = sqrt((float)scaleCount) * sigmaFactor;
	MatF gs(1, scaleCount, GInfoArenas);
	scaleFactors.resize(scaleCount);
	for (int i = 0; i < scaleCount; ++i) {
		float ss = i - std::floor(scaleCount / 2.0f);
		gs.Data()[i] = exp(-0.5f * pow(ss, 2) / pow(scaleSigma, 2));
		scaleFactors[i] = pow(stepOfScale, -ss);
	}
	GFilter.HannWin(scaleWindow, Vector2i(scaleCount, 1));
	GFFT.FFT2Row(gs, scaleGSF);

	// Calculate scale model size for this DSST.
	float scaleModelFactor = 1.0f;
	if (templateSize.x * templateSize.y > maxArea)
		scaleModelFactor = sqrt(maxArea / (templateSize.x * templateSize.y));
	scaleModelSize.x = (int)(templateSize.x * scaleModelFactor);
	scaleModelSize.y = (int)(templateSize.y * scaleModelFactor);

	// Create dssts.
	dssts.resize(targetCount, DSST(learnRateOfScale));
}

void InfoProvider::ConfigureSegment(
	int histDim,
	int histBin,
	float bgRatio,
	float learnRateOfHist,
	int regularCount,
	float maxSegArea) {
	if(histDim <= 0 || histBin <= 0 || bgRatio <= 0.0f || regularCount <= 0) {
		Critical("InfoProvider::ConfigureSegment: invalid segment configuration.");
		return;
	}

	const int rows = yf.Rows();
	const int cols = yf.Cols();

	// Set dummy mask and area;
	defaultMasks.resize(targetCount);
	defaultMaskAreas.resize(targetCount);
	ParallelFor([&](int64_t i) {
		Vector2i scaledObjSize = orgTargetSizes[i] * (rescaleRatio / cellSize);
		int x0 = std::max((cols - scaledObjSize.x) / 2, 0);
		int y0 = std::max((rows - scaledObjSize.y) / 2, 0);
		int x1 = std::min(x0 + scaledObjSize.x, cols);
		int y1 = std::min(y0 + scaledObjSize.y, rows);
		defaultMasks[i].Reshape(rows, cols);
		for (int y = y0; y < y1; ++y) {
			float *pd = defaultMasks[i].Data() + y * cols + x0;
			for (int x = x0; x < x1; ++x)
				(*pd++) = 1.0f;
		}
		defaultMaskAreas[i] = (int)(defaultMasks[i].Mean() * defaultMasks[i].Size());
	}, targetCount, 2);

	// Manual specify the ellipse shaped erode kernel.
	erodeKernel.Reshape(3, 3);
	erodeKernel.Data()[1] = 1.0f;
	erodeKernel.Data()[3] = 1.0f;
	erodeKernel.Data()[4] = 1.0f;
	erodeKernel.Data()[5] = 1.0f;
	erodeKernel.Data()[7] = 1.0f;

	// Initialize histograms, filterMasks and pFores
	histFores.resize(targetCount);
	histBacks.resize(targetCount);
	pFores.resize(targetCount);
	filterMasks.resize(targetCount);
	for (int i = 0; i < targetCount; ++i) {
		histFores[i].Reshape(histDim, histBin);
		histBacks[i].Reshape(histDim, histBin);
	}

	backgroundRatio = bgRatio;
	histLearnRate = learnRateOfHist;
	postRegularCount = regularCount;
	maxSegmentArea = maxSegArea;
}

void InfoProvider::ConfigureSmoother(float stepOfScale) {
	if(stepOfScale <= 0.0f) {
		Critical("InfoProvider::ConfigureSmoother: invalid smoother configuration.");
		return;
	}

	smoother1Ds.resize(targetCount);
	smoother2Ds.resize(targetCount);
	int referScale = 0;
	TransformSmoothParamter params2D(0.15f, 0.15f, 0.15f, referScale * 0.05f, referScale * 0.5f);
	TransformSmoothParamter params1D(0.15f, 0.15f, 0.15f, std::abs(stepOfScale - 1.0f), std::abs(std::pow(stepOfScale, scaleCount / 4.0f) - 1.0f));
	for (int i = 0; i < targetCount; ++i) {
		referScale = std::min(orgTargetSizes[i].x, orgTargetSizes[i].y);
		params2D.fJitterRadius = referScale * 0.05f;
		params2D.fMaxDeviationRadius = referScale * 0.5f;
		smoother1Ds[i].Init(params1D);
		smoother2Ds[i].Init(params2D);
	}
}

void InfoProvider::GetLocationPrior(
	const Vector2i& targetSize,
	const Vector2i& imgSize,
	MatF& prior) const {
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

bool InfoProvider::CheckMaskArea(
	const MatF& mat,
	int area) const {
	static const float threshold = 0.05f;
	float maskArea = mat.Mean() * mat.Size();
	return maskArea >= threshold * area;
}

void InfoProvider::ExtractHistograms(
    const Mat &image,
    const std::vector<Bounds2i> &bbs,
    std::vector<Histogram> &hfs,
    std::vector<Histogram> &hbs) {
    if (bbs.size() != hfs.size() || bbs.size() != hbs.size()) {
        Critical("InfoProvider::ExtractHistograms: bounding box count and histogram count not match.");
        return;
    }

    ParallelFor([&](int64_t i) {
        Vector2i vd = bbs[i].Diagonal();
        int offsetX = (int)(vd.x / backgroundRatio);
        int offsetY = (int)(vd.y / backgroundRatio);
        Bounds2i outbb;
        outbb.pMin.x = std::max(0, bbs[i].pMin.x - offsetX);
        outbb.pMin.y = std::max(0, bbs[i].pMin.y - offsetY);
        outbb.pMax.x = std::min(image.Cols(), bbs[i].pMax.x + offsetX);
        outbb.pMax.y = std::min(image.Rows(), bbs[i].pMax.y + offsetY);

        // Calculate probability for the foreground.
        pFores[i] = (float)bbs[i].Area() / outbb.Area();
        hfs[i].ExtractForeHist(image, bbs[i]);
        hbs[i].ExtractBackHist(image, bbs[i], outbb);
    }, targetCount, 2);
}

void InfoProvider::UpdateHistograms(
	const Mat& image,
	const std::vector<Bounds2i> &bbs) {
	// Create temporary histograms.
	int dim = histFores[0].Dim();
	int bin = histFores[0].Bins();
	std::vector<Histogram> hfs(targetCount, Histogram(dim, bin, &GInfoArenas[ThreadIndex]));
	std::vector<Histogram> hbs(targetCount, Histogram(dim, bin, &GInfoArenas[ThreadIndex]));
	ExtractHistograms(image, bbs, hfs, hbs);

	int len = histFores[0].Size();
	ParallelFor([&](int64_t index) {
		const float *pnewhf = hfs[index].Data();
		const float *pnewhb = hbs[index].Data();
		float *phf = histFores[index].Data();
		float *phb = histBacks[index].Data();
		// Update using histogram learning rate.
		for (int i = 0; i < len; ++i) {
			*phf = Lerp(histLearnRate, *phf, *pnewhf);
			*phb = Lerp(histLearnRate, *phb, *pnewhb);
			++phf; ++pnewhf;
			++phb; ++pnewhb;
		}
	}, targetCount, 2);

	ResetArenas();
}

void InfoProvider::SegmentRegion(
	const Mat &image,
	const std::vector<Vector2i> &objPositions,
	const std::vector<float> &objScales) {
	if (objPositions.size() != targetCount || objScales.size() != targetCount) {
		Critical("InfoProvider::SegmentRegion: invalid input parameters.");
		return;
	}

	for (int index = 0; index < targetCount; ++index) {
		// Calculate posterior of foreground and background.
		Vector2i scaledTemp = templateSize * objScales[index];
		Vector2i scaledTarget = orgTargetSizes[index] * objScales[index];
		Bounds2i validPixels;
		Mat patch(&GInfoArenas[ThreadIndex]);
		GFilter.GetSubWindow(image, patch, objPositions[index], scaledTemp.x, scaledTemp.y, &validPixels);
		MatF fgPrior(&GInfoArenas[ThreadIndex]);
		GetLocationPrior(scaledTarget, scaledTemp, fgPrior);
		MatF fgPost(&GInfoArenas[ThreadIndex]), bgPost(&GInfoArenas[ThreadIndex]);
		GSegment.ComputePosteriors(patch, fgPost, bgPost, pFores[index], fgPrior,
			histFores[index], histBacks[index], postRegularCount, maxSegmentArea);

		// Copy valid region.
		int rows = fgPost.Rows();
		int cols = fgPost.Cols();
		filterMasks[index].Reshape(rows, cols);
		const float *psrc = fgPost.Data();
		float *pdest = filterMasks[index].Data();
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
		filterMasks[index].MaxLoc(maxValue, maxLoc);
		maxValue /= 2.0f;
		ParallelFor([&](int64_t i) {
			if (pdest[i] > maxValue) pdest[i] = 1.0f;
			else pdest[i] = 0.0f;
		}, filterMasks[index].Size(), 2048);
	}

	int rows = yf.Rows();
	int cols = yf.Cols();
	// Issue: condider nearest resize.
	std::vector<MatF> resizedFilterMasks(targetCount, &GInfoArenas[ThreadIndex]);
	ParallelFor([&](int64_t index) {
		filterMasks[index].Resize(resizedFilterMasks[index], rows, cols);
		if (CheckMaskArea(resizedFilterMasks[index], defaultMaskAreas[index]))
			// Issue: consider use -inf border value for dilation
		    GFilter.Dilate2D(resizedFilterMasks[index], erodeKernel, filterMasks[index], BorderType::Zero);
		else
			filterMasks[index] = defaultMasks[index];
	}, targetCount, 2);

	ResetArenas();
}


}	// namespace CSRT