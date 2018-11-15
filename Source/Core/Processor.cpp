//
// Main interface for the tracking system
//

#include "Processor.h"
#include "../Utility/Parallel.h"
#include "Filter.h"

namespace CSRT {

// Default parameters.
TrackerParams::TrackerParams() {
	UseHOG = true;
	UseCN = true;
	UseGRAY = true;
	UseRGB = false;
	UseDepthHOG = false;
	UseDepthGray = true;
	UseDepthNormal = true;
	UseDepthHON = true;
	UseDepthNormalHOG = false;
	NumHOGChannelsUsed = 18;
	PCACount = 0;

	UseNormalForSegment = true;
	UseNormalForDSST = false;
	UseChannelWeights = true;
	UseSegmentation = true;

	CellSize = 4;
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
	UseScale = true;
	UseSmoother = true;
}

Processor::Processor(const TrackerParams& trackParams, int count, const Vector2i &moveSize, TrackMode mode) :
	params(trackParams), targetCount(count), moveRegion(moveSize), trackMode(mode), featMask(0), initialized(false), 
	arenas(nullptr), backgroundUpdateFlag(false), updateCounter(0) {
	if (params.NumHOGChannelsUsed > HOG_FEATURE_COUNT)
		Critical("Processor::Processor: used hog feature number is greater than hog feature total count.");
	if (trackMode == TrackMode::RGB) {
		if (params.UseHOG) featMask |= 1;
		if (params.UseCN) featMask |= 2;
		if (params.UseGRAY) featMask |= 4;
		if (params.UseRGB) featMask |= 8;
	} else if (trackMode == TrackMode::Depth) {
		if (params.UseDepthHOG) featMask |= 1;
		if (params.UseDepthGray) featMask |= 2;
		if (params.UseDepthNormal) featMask |= 4;
		if (params.UseDepthHON) featMask |= 8;
		if (params.UseDepthNormalHOG) featMask |= 16;
	}
}

Processor::~Processor() {
	if(params.UpdateInterval > 0) {
		updateCounter = 0;
		backgroundUpdateFlag = false;
		BackgroundWait();
	}
	if (arenas) {
		delete[] arenas;
		arenas = nullptr;
	}
}

void Processor::ResetArenas(bool background) const {
	if (!arenas) return;
	int len = MaxThreadIndex() - 1;
	if (background) {
		arenas[len].Reset();
	} else {
		for (int i = 0; i < len; ++i)
			arenas[i].Reset();
	}
}

void Processor::SetReinitialize() {
	if(params.UpdateInterval > 0) {
		updateCounter = 0;
		backgroundUpdateFlag = false;
		BackgroundWait();
	}
	initialized = false;
}

void Processor::Initialize(const Mat& rgbImage, const std::vector<Bounds2i>& bbs) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];
	for (auto &item : GInfoProvider.trackers) item.SetReinitialize();
	if (params.UseScale)
		for (auto &item : GInfoProvider.dssts) item.SetReinitialize();

	// Input check.
	if (trackMode == TrackMode::Depth) {
		Critical("Processor::Initialize: cannot initialize with rgb image in the depth track mode.");
		return;
	}
	if (bbs.size() != targetCount) {
		Critical("Processor::Initialize: invalid number of initial bounding boxes.");
		return;
	}
	if (rgbImage.Size() <= 0 || rgbImage.Channels() != 3 || rgbImage.Cols() != moveRegion.x || rgbImage.Rows() != moveRegion.y) {
		Critical("Processor::Initialize: must input 3-channel nonempty rgb image, size must match moveRegion.");
		return;
	}

	// Configure the InfoProvider.
	GInfoProvider.ConfigureTargets(targetCount, params.CellSize, moveRegion, bbs);
	GInfoProvider.ConfigureTracker(params.Padding, params.TemplateSize, params.GaussianSigma,
		params.UseChannelWeights, params.PCACount, params.AdmmIterations, params.WeightsLearnRate,
		params.FilterLearnRate, params.WindowFunc, params.ChebAttenuation, params.KaiserAlpha);
	if (params.UseScale)
		GInfoProvider.ConfigureDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	// Note: segmentation process extract histogram from 3 channel image.
	GInfoProvider.ConfigureSegment(3, params.HistogramBins, params.BackgroundRatio, params.HistLearnRate, params.PostRegularCount, params.MaxSegmentArea);
	if (params.UseSmoother)
		GInfoProvider.ConfigureSmoother(params.ScaleStep);

	// Do segmentation.
	if (params.UseSegmentation) {
		Mat hsvImg(&arenas[ThreadIndex]);
		rgbImage.RGBToHSV(hsvImg);
		GInfoProvider.ExtractHistograms(hsvImg, GInfoProvider.currentBounds, GInfoProvider.histFores, GInfoProvider.histBacks);
		GInfoProvider.SegmentRegion(hsvImg, GInfoProvider.currentPositions, GInfoProvider.currentScales);
	}

	// Initialize trackers.
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	std::vector<MatF> features;
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(rgbImage, orgPatch, GInfoProvider.currentPositions[i],
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		GInfoProvider.GetTrackFeatures(patch, featureSize, features, GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

		if (params.UseSegmentation)
			GInfoProvider.trackers[i].Initialize(features, GInfoProvider.yf, GInfoProvider.filterMasks[i]);
		else
			GInfoProvider.trackers[i].Initialize(features, GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
	}

	// Initialize dssts.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
				GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
			GInfoProvider.dssts[i].Initialize(scaleFeature, GInfoProvider.scaleGSF);
		}
	}

	ResetArenas(false);
	ResetArenas(true);
}

void Processor::Initialize(const MatF& depthImage, const MatF& normalImage, const std::vector<Bounds2i>& bbs) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];

	// Input check.
	if (trackMode == TrackMode::RGB) {
		Critical("Processor::Initialize: cannot initialize with depth image in the rgb track mode.");
		return;
	}
	if (bbs.size() != targetCount) {
		Critical("Processor::Initialize: invalid number of initial bounding boxes.");
		return;
	}
	if (depthImage.Size() <= 0 || depthImage.Cols() != moveRegion.x || depthImage.Rows() != moveRegion.y) {
		Critical("Processor::Initialize: must input depth image, size must match moveRegion.");
		return;
	}
	if((params.UseNormalForSegment || params.UseNormalForDSST) && normalImage.Size() == 0) {
		Critical("Processor::Initialize: must input normal image for normal mode.");
		return;
	}
	if (normalImage.Size() != 0 && (normalImage.Cols() != moveRegion.x * 2 || normalImage.Rows() != moveRegion.y)) {
		Critical("Processor::Initialize: must input normal image, size must match moveRegion.");
		return;
	}

	// Configure the InfoProvider.
	GInfoProvider.ConfigureTargets(targetCount, params.CellSize, moveRegion, bbs);
	GInfoProvider.ConfigureTracker(params.Padding, params.TemplateSize, params.GaussianSigma,
		params.UseChannelWeights, params.PCACount, params.AdmmIterations, params.WeightsLearnRate,
		params.FilterLearnRate, params.WindowFunc, params.ChebAttenuation, params.KaiserAlpha);
	if (params.UseScale)
		GInfoProvider.ConfigureDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	// Note: segmentation process extract histogram from 1 channel image for depth and 2 channel image for normal.
	GInfoProvider.ConfigureSegment(params.UseNormalForSegment ? 2 : 1, params.HistogramBins, params.BackgroundRatio, params.HistLearnRate, params.PostRegularCount, params.MaxSegmentArea);
	if (params.UseSmoother)
		GInfoProvider.ConfigureSmoother(params.ScaleStep);

	// Do segmentation.
	if (params.UseSegmentation) {
		Mat img(&arenas[ThreadIndex]);
		if (params.UseNormalForSegment)
			normalImage.ToMat(img, 2, 255.0f);
		else
			depthImage.ToMat(img, 1, 255.0f);
		GInfoProvider.ExtractHistograms(img, GInfoProvider.currentBounds, GInfoProvider.histFores, GInfoProvider.histBacks);
		GInfoProvider.SegmentRegion(img, GInfoProvider.currentPositions, GInfoProvider.currentScales);
	}

	// Extract features for tracker.
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	std::vector<std::vector<MatF>> features(targetCount, std::vector<MatF>());
	MatF orgDepthPatch(&arenas[ThreadIndex]), depthPatch(&arenas[ThreadIndex]);
	MatF orgNormalPatch(&arenas[ThreadIndex]), normalPatch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		GFilter.GetSubWindow(depthImage, orgDepthPatch, GInfoProvider.currentPositions[i],
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		if(normalImage.Size() != 0) {
			GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i(GInfoProvider.currentPositions[i].x * 2, GInfoProvider.currentPositions[i].y),
				((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2);
		}
		GInfoProvider.GetTrackFeatures(depthPatch, normalPatch, featureSize, features[i], GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);
	}

	// Initialize trackers.
	if (params.UseSegmentation) {
		for (int i = 0; i < targetCount; ++i)
			GInfoProvider.trackers[i].Initialize(features[i], GInfoProvider.yf, GInfoProvider.filterMasks[i]);
	} else {
		for (int i = 0; i < targetCount; ++i)
			GInfoProvider.trackers[i].Initialize(features[i], GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
	}

	if (params.UseScale) {
		// Extract features for dsst.
		std::vector<MatF> scaleFeatures(targetCount, MatF(&arenas[ThreadIndex]));
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeatures[i], GInfoProvider.currentPositions[i],
				GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
			GInfoProvider.dssts[i].Initialize(scaleFeatures[i], GInfoProvider.scaleGSF);
		}
	}

	ResetArenas(false);
	ResetArenas(true);
}

void Processor::Update(const Mat& rgbImage, std::vector<Bounds2i>& bbs) {
	if (!initialized) {
		Error("Processor::Update: the tracking system is not initialized.");
		return;
	}

	// Input check.
	if (trackMode == TrackMode::Depth) {
		Critical("Processor::Update: cannot update with rgb image in the depth track mode.");
		return;
	}
	if (rgbImage.Size() <= 0 || rgbImage.Channels() != 3 || rgbImage.Cols() != moveRegion.x || rgbImage.Rows() != moveRegion.y) {
		Critical("Processor::Update: must input 3-channel nonempty rgb image, size must match moveRegion.");
		return;
	}
	bbs.resize(targetCount);

	// Get new positions.
	Vector2i newPosition;
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(rgbImage, orgPatch, GInfoProvider.currentPositions[i],
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		GInfoProvider.GetTrackFeatures(patch, featureSize, features, GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

		GInfoProvider.trackers[i].GetPosition(
			features, GInfoProvider.currentScales[i], GInfoProvider.cellSize, GInfoProvider.rescaleRatio,
			GInfoProvider.currentPositions[i], GInfoProvider.moveSize, newPosition);
		GInfoProvider.currentPositions[i] = newPosition;
	}
	if (params.UseSmoother) {
		Vector2f filteredPosition;
		for (int i = 0; i < targetCount; ++i) {
			filteredPosition = GInfoProvider.smoother2Ds[i].Update(
				Vector2f(GInfoProvider.currentPositions[i].x, GInfoProvider.currentPositions[i].y));
			GInfoProvider.currentPositions[i].x = Clamp((int)filteredPosition.x, 1, rgbImage.Cols() - 2);
			GInfoProvider.currentPositions[i].y = Clamp((int)filteredPosition.y, 1, rgbImage.Rows() - 2);
		}
	}

	// Get new scales.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		float newScale = 1.0f;
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
				GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
			GInfoProvider.dssts[i].GetScale(scaleFeature, GInfoProvider.scaleFactors, GInfoProvider.currentScales[i],
				GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
			GInfoProvider.currentScales[i] = newScale;
		}
		if (params.UseSmoother) {
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.currentScales[i] = Clamp(GInfoProvider.smoother1Ds[i].Update(GInfoProvider.currentScales[i]),
					GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i]);
			}
		}
	}

	// Update bounding boxes.
	Vector2i newSize;
	auto &bounds = GInfoProvider.currentBounds;
	auto &positions = GInfoProvider.currentPositions;
	for (int i = 0; i < targetCount; ++i) {
		newSize = GInfoProvider.currentScales[i] * GInfoProvider.orgTargetSizes[i];
		bounds[i].pMin.x = std::max(positions[i].x - newSize.x / 2, 0);
		bounds[i].pMin.y = std::max(positions[i].y - newSize.y / 2, 0);
		bounds[i].pMax.x = std::min(bounds[i].pMin.x + newSize.x, rgbImage.Cols());
		bounds[i].pMax.y = std::min(bounds[i].pMin.y + newSize.y, rgbImage.Rows());
	}
	bbs = bounds;

	// Update the tracking system.
	if (params.UpdateInterval <= 0) {	// Do update work in the main thread.
		// Update filter masks.
		if (params.UseSegmentation) {
			Mat hsvImg(&arenas[ThreadIndex]);
			rgbImage.RGBToHSV(hsvImg);
			GInfoProvider.UpdateHistograms(hsvImg, GInfoProvider.currentBounds);
			GInfoProvider.SegmentRegion(hsvImg, GInfoProvider.currentPositions, GInfoProvider.currentScales);
		}
		// Update trackers.
		for (int i = 0; i < targetCount; ++i) {
			// Extract features for tracker.
			GFilter.GetSubWindow(rgbImage, orgPatch, GInfoProvider.currentPositions[i],
				(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
				(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
			orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
			GInfoProvider.GetTrackFeatures(patch, featureSize, features, GInfoProvider.window, featMask,
				GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

			if (params.UseSegmentation)
				GInfoProvider.trackers[i].Update(features, GInfoProvider.yf, GInfoProvider.filterMasks[i]);
			else
				GInfoProvider.trackers[i].Update(features, GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
		}
		// Update dssts.
		if (params.UseScale) {
			// Extract features for dsst.
			MatF scaleFeature(&arenas[ThreadIndex]);
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
					GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
					GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
				GInfoProvider.dssts[i].Update(scaleFeature);
			}
		}
	} else {	// Do update work in the background thread.
		++updateCounter;
		if (updateCounter >= params.UpdateInterval) {
			updateCounter = 0;
			BackgroundWait();
			FetchUpdateResult();
			StartBackgroundUpdate(rgbImage);
			BackgroundProcess([&]() {
				BackgroundUpdateRGB();
			});
		}
	}

	ResetArenas(false);
}

void Processor::Update(const MatF& depthImage, const MatF& normalImage, std::vector<Bounds2i>& bbs) {
	if (!initialized) {
		Error("Processor::Update: the tracking system is not initialized.");
		return;
	}

	// Input check.
	if (trackMode == TrackMode::RGB) {
		Critical("Processor::Update: cannot update with depth image in the rgb track mode.");
		return;
	}
	if (depthImage.Size() <= 0 || depthImage.Cols() != moveRegion.x || depthImage.Rows() != moveRegion.y) {
		Critical("Processor::Update: must input 1-channel nonempty depth image, size must match moveRegion.");
		return;
	}
	if ((params.UseNormalForSegment || params.UseNormalForDSST) && normalImage.Size() == 0) {
		Critical("Processor::Update: must input normal image for normal mode.");
		return;
	}
	if (normalImage.Size() != 0 && (normalImage.Cols() != moveRegion.x * 2 || normalImage.Rows() != moveRegion.y)) {
		Critical("Processor::Update: must input normal image, size must match moveRegion.");
		return;
	}
	bbs.resize(targetCount);

	// Get new positions.
	Vector2i newPosition;
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	MatF orgDepthPatch(&arenas[ThreadIndex]), depthPatch(&arenas[ThreadIndex]);
	MatF orgNormalPatch(&arenas[ThreadIndex]), normalPatch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(depthImage, orgDepthPatch, GInfoProvider.currentPositions[i],
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		if(normalImage.Size() != 0) {
			GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i(GInfoProvider.currentPositions[i].x * 2, GInfoProvider.currentPositions[i].y),
				((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2);
		}
		GInfoProvider.GetTrackFeatures(depthPatch, normalPatch, featureSize, features, GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

		GInfoProvider.trackers[i].GetPosition(
			features, GInfoProvider.currentScales[i], GInfoProvider.cellSize, GInfoProvider.rescaleRatio,
			GInfoProvider.currentPositions[i], GInfoProvider.moveSize, newPosition);
		GInfoProvider.currentPositions[i] = newPosition;
	}
	if (params.UseSmoother) {
		Vector2f filteredPosition;
		for (int i = 0; i < targetCount; ++i) {
			filteredPosition = GInfoProvider.smoother2Ds[i].Update(
				Vector2f(GInfoProvider.currentPositions[i].x, GInfoProvider.currentPositions[i].y));
			GInfoProvider.currentPositions[i].x = Clamp((int)filteredPosition.x, 1, depthImage.Cols() - 2);
			GInfoProvider.currentPositions[i].y = Clamp((int)filteredPosition.y, 1, depthImage.Rows() - 2);
		}
	}

	// Get new scales.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		float newScale = 1.0f;
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
				GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
			GInfoProvider.dssts[i].GetScale(scaleFeature, GInfoProvider.scaleFactors, GInfoProvider.currentScales[i],
				GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
			GInfoProvider.currentScales[i] = newScale;
		}
		if (params.UseSmoother) {
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.currentScales[i] = Clamp(GInfoProvider.smoother1Ds[i].Update(GInfoProvider.currentScales[i]),
					GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i]);
			}
		}
	}

	// Update bounding boxes.
	Vector2i newSize;
	auto &bounds = GInfoProvider.currentBounds;
	auto &positions = GInfoProvider.currentPositions;
	for (int i = 0; i < targetCount; ++i) {
		newSize = GInfoProvider.currentScales[i] * GInfoProvider.orgTargetSizes[i];
		bounds[i].pMin.x = std::max(positions[i].x - newSize.x / 2, 0);
		bounds[i].pMin.y = std::max(positions[i].y - newSize.y / 2, 0);
		bounds[i].pMax.x = std::min(bounds[i].pMin.x + newSize.x, depthImage.Cols());
		bounds[i].pMax.y = std::min(bounds[i].pMin.y + newSize.y, depthImage.Rows());
	}
	bbs = bounds;

	// Update the tracking system.
	if (params.UpdateInterval <= 0) {	// Do update work in the main thread.
		// Update filter masks.
		if (params.UseSegmentation) {
			Mat img(&arenas[ThreadIndex]);
			if (params.UseNormalForSegment)
				normalImage.ToMat(img, 2, 255.0f);
			else
				depthImage.ToMat(img, 1, 255.0f);
			GInfoProvider.UpdateHistograms(img, GInfoProvider.currentBounds);
			GInfoProvider.SegmentRegion(img, GInfoProvider.currentPositions, GInfoProvider.currentScales);
		}
		// Update trackers.
		for (int i = 0; i < targetCount; ++i) {
			// Extract features for tracker.
			GFilter.GetSubWindow(depthImage, orgDepthPatch, GInfoProvider.currentPositions[i],
				(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
				(int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
			orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
			if(normalImage.Size() != 0) {
				GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i(GInfoProvider.currentPositions[i].x * 2, GInfoProvider.currentPositions[i].y),
					((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
					((int)(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
				orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2);
			}
			GInfoProvider.GetTrackFeatures(depthPatch, normalPatch, featureSize, features, GInfoProvider.window, featMask,
				GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

			if (params.UseSegmentation)
				GInfoProvider.trackers[i].Update(features, GInfoProvider.yf, GInfoProvider.filterMasks[i]);
			else
				GInfoProvider.trackers[i].Update(features, GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
		}
		// Update dssts.
		if (params.UseScale) {
			// Extract features for dsst.
			MatF scaleFeature(&arenas[ThreadIndex]);
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
					GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
					GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
				GInfoProvider.dssts[i].Update(scaleFeature);
			}
		}
	} else {	// Do update work in the background thread.
		++updateCounter;
		if (updateCounter >= params.UpdateInterval) {
			updateCounter = 0;
			BackgroundWait();
			FetchUpdateResult();
			StartBackgroundUpdate(depthImage, normalImage);
			BackgroundProcess([&]() {
				BackgroundUpdateDepth();
			});
		}
	}

	ResetArenas(false);
}

void Processor::StartBackgroundUpdate(const Mat& rgbImage) {
	if (backgroundUpdateFlag) return;
	bk_rgbImage = rgbImage;
	bk_positions = GInfoProvider.currentPositions;
	bk_scales = GInfoProvider.currentScales;
	bk_bounds = GInfoProvider.currentBounds;
	for (auto &item : GInfoProvider.trackers) item.StartBackgroundUpdate();
	if (params.UseScale)
		for (auto &item : GInfoProvider.dssts) item.StartBackgroundUpdate();
	backgroundUpdateFlag = true;
}

void Processor::StartBackgroundUpdate(const MatF& depthImage, const MatF &normalImage) {
	if (backgroundUpdateFlag) return;
	bk_depthImage = depthImage;
	bk_normalImage = normalImage;
	bk_positions = GInfoProvider.currentPositions;
	bk_scales = GInfoProvider.currentScales;
	bk_bounds = GInfoProvider.currentBounds;
	for (auto &item : GInfoProvider.trackers) item.StartBackgroundUpdate();
	if (params.UseScale)
		for (auto &item : GInfoProvider.dssts) item.StartBackgroundUpdate();
	backgroundUpdateFlag = true;
}

void Processor::FetchUpdateResult() {
	if (!backgroundUpdateFlag) return;
	for (auto &item : GInfoProvider.trackers) item.FetchUpdateResult();
	if (params.UseScale)
		for (auto &item : GInfoProvider.dssts) item.FetchUpdateResult();
	backgroundUpdateFlag = false;
}

void Processor::BackgroundUpdateRGB() {
	// Update filter masks.
	if (params.UseSegmentation) {
		Mat hsvImg(&arenas[ThreadIndex]);
		bk_rgbImage.RGBToHSV(hsvImg);
		GInfoProvider.UpdateHistograms(hsvImg, bk_bounds);
		GInfoProvider.SegmentRegion(hsvImg, bk_positions, bk_scales);
	}
	// Update trackers.
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(bk_rgbImage, orgPatch, bk_positions[i],
			(int)(bk_scales[i] * GInfoProvider.templateSize.x),
			(int)(bk_scales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		GInfoProvider.GetTrackFeatures(patch, featureSize, features, GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

		if (params.UseSegmentation)
			GInfoProvider.trackers[i].BackgroundUpdate(features, GInfoProvider.yf, GInfoProvider.filterMasks[i]);
		else
			GInfoProvider.trackers[i].BackgroundUpdate(features, GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
	}
	// Update dssts.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(bk_rgbImage, scaleFeature, bk_positions[i],
				GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
			GInfoProvider.dssts[i].BackgroundUpdate(scaleFeature);
		}
	}

	ResetArenas(true);
}

void Processor::BackgroundUpdateDepth() {
	// Update filter masks.
	if (params.UseSegmentation) {
		Mat img(&arenas[ThreadIndex]);
		if (params.UseNormalForSegment)
			bk_normalImage.ToMat(img, 2, 255.0f);
		else
			bk_depthImage.ToMat(img, 1, 255.0f);
		GInfoProvider.UpdateHistograms(img, bk_bounds);
		GInfoProvider.SegmentRegion(img, bk_positions, bk_scales);
	}
	// Update trackers.
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	MatF orgDepthPatch(&arenas[ThreadIndex]), depthPatch(&arenas[ThreadIndex]);
	MatF orgNormalPatch(&arenas[ThreadIndex]), normalPatch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(bk_depthImage, orgDepthPatch, bk_positions[i],
			(int)(bk_scales[i] * GInfoProvider.templateSize.x),
			(int)(bk_scales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x);
		if(bk_normalImage.Size() != 0) {
			GFilter.GetSubWindow(bk_normalImage, orgNormalPatch, Vector2i(bk_positions[i].x * 2, bk_positions[i].y),
				((int)(bk_scales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)(bk_scales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2);
		}
		GInfoProvider.GetTrackFeatures(depthPatch, normalPatch, featureSize, features, GInfoProvider.window, featMask,
			GInfoProvider.cellSize, params.NumHOGChannelsUsed, &arenas[ThreadIndex]);

		if (params.UseSegmentation)
			GInfoProvider.trackers[i].BackgroundUpdate(features, GInfoProvider.yf, GInfoProvider.filterMasks[i]);
		else
			GInfoProvider.trackers[i].BackgroundUpdate(features, GInfoProvider.yf, GInfoProvider.defaultMasks[i]);
	}
	// Update dssts.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		for (int i = 0; i < targetCount; ++i) {
			GInfoProvider.GetScaleFeatures(bk_depthImage, bk_normalImage, scaleFeature, bk_positions[i],
				GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
				GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
			GInfoProvider.dssts[i].BackgroundUpdate(scaleFeature);
		}
	}

	ResetArenas(true);
}

}	// namespace CSRT