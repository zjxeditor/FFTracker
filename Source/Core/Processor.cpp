//
// Main interface for the tracking system
//

#include <Utility/FFT.h>
#include "Processor.h"
#include "Filter.h"
#include "FeaturesExtractor.h"
#include "../Utility/Parallel.h"

namespace CSRT {

void StartSystem(int count, const char *logPath) {
    if (logPath == nullptr)
        CSRT::CreateLogger();
    else
        CSRT::CreateLogger(std::string(logPath));

    if(count <= 0)
        CSRT::SetThreadCount(NumSystemCores() + 1);
    else {
        if(count == 1) ++count;
        CSRT::SetThreadCount(count);
    }

    CSRT::ParallelInit();
    Eigen::setNbThreads(MaxThreadIndex());
    Eigen::initParallel();
    GFFT.Initialize();
    GImageMemoryArena.Initialize();
    GFeatsExtractor.Initialize();
    GFilter.Initialize();
    GSegment.Initialize();
    GInfoProvider.Initialize();
}

void CloseSystem() {
    CSRT::ParallelCleanup();
    CSRT::Info("All work is done!");
    CSRT::ClearLogger();
}

void Rgba2Rgb(unsigned char *srcData, unsigned char *dstData, int rows, int cols) {
    ParallelFor([&](int64_t y) {
        const unsigned char *ps = srcData + y * cols * 4;
        unsigned char *pd = dstData + y * cols * 3;
        for (int i = 0; i < cols; ++i) {
            *(pd++) = *(ps++);
            *(pd++) = *(ps++);
            *(pd++) = *(ps++);
            ++ps;
        }
    }, rows, 32);
}

//
// Compute normal from point cloud.
//

void ComputeNormalFromPointCloud(const Vector3f *pointCloud, Vector3f *normal, int width, int height) {
    // Calculate normals
    ParallelFor([&](int64_t y) {
        ++y;
        const Vector3f *src = pointCloud + y * width + 1;
        Vector3f *dest = normal + y * width + 1;
        Vector3f diff;
        float dy, dx, v;

        for (int x = 1; x < width - 1; ++x) {
            diff = *(src - width) - *(src + width);	// Note the y-axis direction in different coordinate system.
            dy = diff.z / diff.y;
            diff = *(src + 1) - *(src - 1);
            dx = diff.z / diff.x;
            v = dx * dx + dy * dy + 1.0f;
            v = sqrt(v);

            dest->x = -dx / v;
            dest->y = -dy / v;
            dest->z = 1.0f / v;

            ++src;
            ++dest;
        }
    }, height - 2, 32);

    // Pad borders
    Vector3f *src = normal + width + 1;
    Vector3f *dest = normal + 1;
    for (int i = 1; i < width - 1; ++i)
        *(dest++) = *(src++);
    src = normal + (height - 2) * width + 1;
    dest = normal + (height - 1) * width + 1;
    for (int i = 1; i < width - 1; ++i)
        *(dest++) = *(src++);
    src = normal + 1;
    dest = normal;
    for (int i = 0; i < height; ++i) {
        *dest = *src;
        src += width;
        dest += width;
    }
    src = normal + width - 2;
    dest = src + 1;
    for (int i = 0; i < height; ++i) {
        *dest = *src;
        src += width;
        dest += width;
    }
}

void ComputePolarNormalFromPointCloud(const Vector3f *pointCloud, Vector2f *polarNormal, int width, int height) {
    // Calculate normals
    ParallelFor([&](int64_t y) {
        ++y;
        const Vector3f *src = pointCloud + y * width + 1;
        Vector2f *dest = polarNormal + y * width + 1;
        Vector3f diff;
        float dy, dx, v;

        for (int x = 1; x < width - 1; ++x) {
            diff = *(src - width) - *(src + width);	// Note the y-axis direction in different coordinate system.
            dy = diff.z / diff.y;
            diff = *(src + 1) - *(src - 1);
            dx = diff.z / diff.x;
            v = dx * dx + dy * dy + 1.0f;
            v = sqrt(v);

            if(std::abs(v - 1.0f) < 1e-3f) {
                dest->x = 0.0f;
                dest->y = 0.0f;
            } else {
                dest->x = acos(Clamp(1.0f / v, -1.0f, 1.0f));
                dest->y = acos(Clamp((-dx) / sqrt(v * v - 1.0f), -1.0f, 1.0f));
                if (dy > 0) dest->y = 2 * Pi - dest->y;
                dest->x /= PiOver2;
                dest->y *= Inv2Pi;
            }

            ++src;
            ++dest;
        }
    }, height - 2, 32);

    // Pad borders
    Vector2f *src = polarNormal + width + 1;
    Vector2f *dest = polarNormal + 1;
    for (int i = 1; i < width - 1; ++i)
        *(dest++) = *(src++);
    src = polarNormal + (height - 2) * width + 1;
    dest = polarNormal + (height - 1) * width + 1;
    for (int i = 1; i < width - 1; ++i)
        *(dest++) = *(src++);
    src = polarNormal + 1;
    dest = polarNormal;
    for (int i = 0; i < height; ++i) {
        *dest = *src;
        src += width;
        dest += width;
    }
    src = polarNormal + width - 2;
    dest = src + 1;
    for (int i = 0; i < height; ++i) {
        *dest = *src;
        src += width;
        dest += width;
    }
}

void ConvertNormalToPolarNormal(const Vector3f* normal, Vector2f* polarNormal, int width, int height) {
    ParallelFor([&](int64_t y) {
        const Vector3f *src = normal + y * width;
        Vector2f *dest = polarNormal + y * width;
        float v = 0.0f;
        for (int x = 0; x < width; ++x) {
            v = src->Length();
            if (std::abs(v - src->z) < 1e-3f) {
                dest->x = 0.0f;
                dest->y = 0.0f;
            } else {
                dest->x = acos(Clamp(src->z / v, -1.0f, 1.0f));
                dest->y = acos(Clamp((src->x) / sqrt(v * v - src->z * src->z), -1.0f, 1.0f));
                if (src->y < 0.0f) dest->y = 2 * Pi - dest->y;
                dest->x /= PiOver2;
                dest->y *= Inv2Pi;
            }

            ++src;
            ++dest;
        }
    }, height, 32);
}

void ConvertPolarNormalToNormal(const Vector2f* polarNormal, Vector3f* normal, int width, int height) {
    ParallelFor([&](int64_t y) {
        const Vector2f *src = polarNormal + y * width;
        Vector3f *dest = normal + y * width;
        float v = 0.0f, theta = 0.0f;
        for (int x = 0; x < width; ++x) {
            dest->z = cos(src->x * PiOver2);
            v = sqrt(std::max(1.0f - dest->z * dest->z, 0.0f));
            theta = src->y * 2 * Pi;
            dest->x = v * cos(theta);
            dest->y = v * sin(theta);

            ++src;
            ++dest;
        }
    }, height, 32);
}

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

	AdmmIterations = 4;
	Padding = 3.0f;
	TemplateSize = 200.0f;
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
	UseSmoother = false;
	UseFastScale = false;
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
        FetchUpdateResult();
	}
	initialized = false;
}

void Processor::Initialize(const Mat& rgbImage, const std::vector<Bounds2f>& bbs) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];
	for (auto &item : GInfoProvider.trackers) item.SetReinitialize();
	if (params.UseScale) {
        for (auto &item : GInfoProvider.dssts) item.SetReinitialize();
        for (auto &item : GInfoProvider.fdssts) item.SetReinitialize();
	}

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
	GInfoProvider.ConfigureTargets(targetCount, moveRegion, bbs);
	GInfoProvider.ConfigureTracker(params.Padding, params.TemplateSize, params.GaussianSigma,
		params.UseChannelWeights, params.PCACount, params.AdmmIterations, params.WeightsLearnRate,
		params.FilterLearnRate, params.WindowFunc, params.ChebAttenuation, params.KaiserAlpha);
	if (params.UseScale) {
	    if(params.UseFastScale)
            GInfoProvider.ConfigureFDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	    else
            GInfoProvider.ConfigureDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	}
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
		GFilter.GetSubWindow(rgbImage, orgPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
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
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.fdssts[i].Initialize(scaleFeature, GInfoProvider.scaleGSF);
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.dssts[i].Initialize(scaleFeature, GInfoProvider.scaleGSF);
            }
		}
	}

	ResetArenas(false);
	ResetArenas(true);
}

void Processor::Initialize(const MatF& depthImage, const MatF& normalImage, const std::vector<Bounds2f>& bbs) {
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
	GInfoProvider.ConfigureTargets(targetCount, moveRegion, bbs);
	GInfoProvider.ConfigureTracker(params.Padding, params.TemplateSize, params.GaussianSigma,
		params.UseChannelWeights, params.PCACount, params.AdmmIterations, params.WeightsLearnRate,
		params.FilterLearnRate, params.WindowFunc, params.ChebAttenuation, params.KaiserAlpha);
	if (params.UseScale) {
	    if(params.UseFastScale)
            GInfoProvider.ConfigureFDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	    else
            GInfoProvider.ConfigureDSST(params.ScaleCount, params.ScaleStep, params.ScaleSigma, params.ScaleLearnRate, params.ScaleMaxArea);
	}
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
		GFilter.GetSubWindow(depthImage, orgDepthPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
		if(normalImage.Size() != 0) {
			GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x * 2.0f), (int)std::floor(GInfoProvider.currentPositions[i].y)),
				((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2, ResizeMode::Bicubic);
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
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeatures[i], GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.fdssts[i].Initialize(scaleFeatures[i], GInfoProvider.scaleGSF);
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeatures[i], GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.dssts[i].Initialize(scaleFeatures[i], GInfoProvider.scaleGSF);
            }
		}
	}

	ResetArenas(false);
	ResetArenas(true);
}

void Processor::Update(const Mat& rgbImage, std::vector<Bounds2f>& bbs) {
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
	Vector2f newPosition;
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	Mat orgPatch(&arenas[ThreadIndex]), patch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(rgbImage, orgPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
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
			filteredPosition = GInfoProvider.smoother2Ds[i].Update(GInfoProvider.currentPositions[i]);
			GInfoProvider.currentPositions[i].x = Clamp(filteredPosition.x, 0.0f, rgbImage.Cols() - 1.0f);
			GInfoProvider.currentPositions[i].y = Clamp(filteredPosition.y, 0.0f, rgbImage.Rows() - 1.0f);
		}
	}

	// Get new scales.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		float newScale = 1.0f;
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.fdssts[i].GetScale(scaleFeature, GInfoProvider.scaleInterpFactors, GInfoProvider.currentScales[i],
                                                GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
                GInfoProvider.currentScales[i] = newScale;
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.dssts[i].GetScale(scaleFeature, GInfoProvider.scaleFactors, GInfoProvider.currentScales[i],
                                                GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
                GInfoProvider.currentScales[i] = newScale;
            }
		}

		if (params.UseSmoother) {
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.currentScales[i] = Clamp(GInfoProvider.smoother1Ds[i].Update(GInfoProvider.currentScales[i]),
					GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i]);
			}
		}
	}

	// Update bounding boxes.
	Vector2f newSize;
	auto &bounds = GInfoProvider.currentBounds;
	auto &positions = GInfoProvider.currentPositions;
	for (int i = 0; i < targetCount; ++i) {
		newSize = GInfoProvider.currentScales[i] * GInfoProvider.orgTargetSizes[i];
		bounds[i].pMin.x = std::max(positions[i].x - newSize.x / 2.0f, 0.0f);
		bounds[i].pMin.y = std::max(positions[i].y - newSize.y / 2.0f, 0.0f);
		bounds[i].pMax.x = std::min(bounds[i].pMin.x + newSize.x, (float)rgbImage.Cols());
		bounds[i].pMax.y = std::min(bounds[i].pMin.y + newSize.y, (float)rgbImage.Rows());
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
			GFilter.GetSubWindow(rgbImage, orgPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
				(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
				(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
			orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
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
			if(params.UseFastScale) {
			    for (int i = 0; i < targetCount; ++i) {
                    GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                                   GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                                   GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                    GInfoProvider.fdssts[i].Update(scaleFeature);
			    }
			} else {
                for (int i = 0; i < targetCount; ++i) {
                    GInfoProvider.GetScaleFeatures(rgbImage, scaleFeature, GInfoProvider.currentPositions[i],
                                                   GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                                   GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                    GInfoProvider.dssts[i].Update(scaleFeature);
                }
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

void Processor::Update(const MatF& depthImage, const MatF& normalImage, std::vector<Bounds2f>& bbs) {
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
	Vector2f newPosition;
	std::vector<MatF> features;
	const Vector2i featureSize(GInfoProvider.yf.Cols(), GInfoProvider.yf.Rows());
	MatF orgDepthPatch(&arenas[ThreadIndex]), depthPatch(&arenas[ThreadIndex]);
	MatF orgNormalPatch(&arenas[ThreadIndex]), normalPatch(&arenas[ThreadIndex]);
	for (int i = 0; i < targetCount; ++i) {
		// Extract features for tracker.
		GFilter.GetSubWindow(depthImage, orgDepthPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
		if(normalImage.Size() != 0) {
			GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x * 2.0f), (int)std::floor(GInfoProvider.currentPositions[i].y)),
				((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2, ResizeMode::Bicubic);
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
			filteredPosition = GInfoProvider.smoother2Ds[i].Update(GInfoProvider.currentPositions[i]);
			GInfoProvider.currentPositions[i].x = Clamp(filteredPosition.x, 0.0f, depthImage.Cols() - 1.0f);
			GInfoProvider.currentPositions[i].y = Clamp(filteredPosition.y, 0.0f, depthImage.Rows() - 1.0f);
		}
	}

	// Get new scales.
	if (params.UseScale) {
		// Extract features for dsst.
		MatF scaleFeature(&arenas[ThreadIndex]);
		float newScale = 1.0f;
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.fdssts[i].GetScale(scaleFeature, GInfoProvider.scaleInterpFactors, GInfoProvider.currentScales[i],
                                                GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
                GInfoProvider.currentScales[i] = newScale;
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
                                               GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.dssts[i].GetScale(scaleFeature, GInfoProvider.scaleFactors, GInfoProvider.currentScales[i],
                                                GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i], newScale);
                GInfoProvider.currentScales[i] = newScale;
            }
		}

		if (params.UseSmoother) {
			for (int i = 0; i < targetCount; ++i) {
				GInfoProvider.currentScales[i] = Clamp(GInfoProvider.smoother1Ds[i].Update(GInfoProvider.currentScales[i]),
					GInfoProvider.scaleMinFactors[i], GInfoProvider.scaleMaxFactors[i]);
			}
		}
	}

	// Update bounding boxes.
	Vector2f newSize;
	auto &bounds = GInfoProvider.currentBounds;
	auto &positions = GInfoProvider.currentPositions;
	for (int i = 0; i < targetCount; ++i) {
		newSize = GInfoProvider.currentScales[i] * GInfoProvider.orgTargetSizes[i];
		bounds[i].pMin.x = std::max(positions[i].x - newSize.x / 2, 0.0f);
		bounds[i].pMin.y = std::max(positions[i].y - newSize.y / 2, 0.0f);
		bounds[i].pMax.x = std::min(bounds[i].pMin.x + newSize.x, (float)depthImage.Cols());
		bounds[i].pMax.y = std::min(bounds[i].pMin.y + newSize.y, (float)depthImage.Rows());
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
			GFilter.GetSubWindow(depthImage, orgDepthPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x), (int)std::floor(GInfoProvider.currentPositions[i].y)),
				(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x),
				(int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y));
			orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
			if(normalImage.Size() != 0) {
				GFilter.GetSubWindow(normalImage, orgNormalPatch, Vector2i((int)std::floor(GInfoProvider.currentPositions[i].x * 2.0f), (int)std::floor(GInfoProvider.currentPositions[i].y)),
					((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.x) / 2) * 4,
					((int)std::floor(GInfoProvider.currentScales[i] * GInfoProvider.templateSize.y)));
				orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2, ResizeMode::Bicubic);
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
			if(params.UseFastScale) {
                for (int i = 0; i < targetCount; ++i) {
                    GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
                                                   GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                                   GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                    GInfoProvider.fdssts[i].Update(scaleFeature);
                }
			} else {
                for (int i = 0; i < targetCount; ++i) {
                    GInfoProvider.GetScaleFeatures(depthImage, normalImage, scaleFeature, GInfoProvider.currentPositions[i],
                                                   GInfoProvider.orgTargetSizes[i], GInfoProvider.currentScales[i], GInfoProvider.scaleFactors,
                                                   GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                    GInfoProvider.dssts[i].Update(scaleFeature);
                }
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
	if (params.UseScale) {
	    if(params.UseFastScale)
            for (auto &item : GInfoProvider.fdssts) item.StartBackgroundUpdate();
        else
            for (auto &item : GInfoProvider.dssts) item.StartBackgroundUpdate();
	}
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
	if (params.UseScale) {
	    if(params.UseFastScale)
            for (auto &item : GInfoProvider.fdssts) item.StartBackgroundUpdate();
        else
            for (auto &item : GInfoProvider.dssts) item.StartBackgroundUpdate();
	}
	backgroundUpdateFlag = true;
}

void Processor::FetchUpdateResult() {
	if (!backgroundUpdateFlag) return;
	for (auto &item : GInfoProvider.trackers) item.FetchUpdateResult();
	if (params.UseScale) {
	    if(params.UseFastScale)
            for (auto &item : GInfoProvider.fdssts) item.FetchUpdateResult();
        else
            for (auto &item : GInfoProvider.dssts) item.FetchUpdateResult();
	}
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
		GFilter.GetSubWindow(bk_rgbImage, orgPatch, Vector2i((int)std::floor(bk_positions[i].x), (int)std::floor(bk_positions[i].y)),
			(int)std::floor(bk_scales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(bk_scales[i] * GInfoProvider.templateSize.y));
		orgPatch.Resize(patch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
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
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(bk_rgbImage, scaleFeature, bk_positions[i],
                                               GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.fdssts[i].BackgroundUpdate(scaleFeature);
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(bk_rgbImage, scaleFeature, bk_positions[i],
                                               GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize);
                GInfoProvider.dssts[i].BackgroundUpdate(scaleFeature);
            }
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
		GFilter.GetSubWindow(bk_depthImage, orgDepthPatch, Vector2i((int)std::floor(bk_positions[i].x), (int)std::floor(bk_positions[i].y)),
			(int)std::floor(bk_scales[i] * GInfoProvider.templateSize.x),
			(int)std::floor(bk_scales[i] * GInfoProvider.templateSize.y));
		orgDepthPatch.Resize(depthPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, ResizeMode::Bicubic);
		if(bk_normalImage.Size() != 0) {
			GFilter.GetSubWindow(bk_normalImage, orgNormalPatch, Vector2i((int)std::floor(bk_positions[i].x * 2.0f), (int)std::floor(bk_positions[i].y)),
				((int)std::floor(bk_scales[i] * GInfoProvider.templateSize.x) / 2) * 4,
				((int)std::floor(bk_scales[i] * GInfoProvider.templateSize.y)));
			orgNormalPatch.Resize(normalPatch, GInfoProvider.rescaledTemplateSize.y, GInfoProvider.rescaledTemplateSize.x, 2, ResizeMode::Bicubic);
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
		if(params.UseFastScale) {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(bk_depthImage, bk_normalImage, scaleFeature, bk_positions[i],
                                               GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.fdssts[i].BackgroundUpdate(scaleFeature);
            }
		} else {
            for (int i = 0; i < targetCount; ++i) {
                GInfoProvider.GetScaleFeatures(bk_depthImage, bk_normalImage, scaleFeature, bk_positions[i],
                                               GInfoProvider.orgTargetSizes[i], bk_scales[i], GInfoProvider.scaleFactors,
                                               GInfoProvider.scaleWindow, GInfoProvider.scaleModelSize, params.UseNormalForDSST);
                GInfoProvider.dssts[i].BackgroundUpdate(scaleFeature);
            }
		}
	}

	ResetArenas(true);
}

}	// namespace CSRT