#include "../Source/Core/Processor.h"
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

using namespace CSRT;

void ReadGT(const std::string &fileName, std::vector<Bounds2f> & gt) {
	std::ifstream fin(fileName);
	if (!fin.is_open()) {
		Error("ReadGT: cannot open ground truth file.");
		return;
	}
	gt.clear();
	float x, y, w, h;
	while (fin) {
		fin >> x >> y >> w >> h;
		if (fin.good())
			gt.push_back(Bounds2f(x, y, w, h));
	}
}

inline bool IsGoodBox(const Bounds2f &bb) { return bb.Area() > 0; }

void GetImageFile(const std::string& foldername, std::vector<std::string> &files) {
    files.clear();
    DIR *dirp = opendir(foldername.c_str());
    if(dirp == nullptr) {
        Critical("Cannot find image folder.");
        return;
    }
    struct dirent *dp;
    while ((dp = readdir(dirp)) != nullptr) {
        std::string current(dp->d_name);
        if(current == "." || current == "..") continue;
        files.push_back(foldername + "/" + current);
    }
    closedir(dirp);
    std::sort(files.begin(), files.end());
}

void DrawDashRect(cv::Mat img, int linelength, int dashlength, cv::Rect blob, cv::Scalar color, int thickness) {
	int w = cvRound(blob.width);
	int h = cvRound(blob.height);
	int tl_x = cvRound(blob.x);
	int tl_y = cvRound(blob.y);

	int totallength = dashlength + linelength;
	int nCountX = w / totallength;
	int nCountY = h / totallength;

	cv::Point start, end;
	start.y = tl_y;
	start.x = tl_x;
	end.x = tl_x;
	end.y = tl_y;

	for (int i = 0; i < nCountX; i++) {
		end.x = tl_x + (i + 1) * totallength - dashlength;
		end.y = tl_y;
		start.x = tl_x + i * totallength;
		start.y = tl_y;
		cv::line(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountX; i++) {
		start.x = tl_x + i * totallength;
		start.y = tl_y + h;
		end.x = tl_x + (i + 1) * totallength - dashlength;
		end.y = tl_y + h;
		cv::line(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountY; i++) {
		start.x = tl_x;
		start.y = tl_y + i * totallength;
		end.y = tl_y + (i + 1) * totallength - dashlength;
		end.x = tl_x;
		cv::line(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountY; i++) {
		start.x = tl_x + w;
		start.y = tl_y + i * totallength;
		end.y = tl_y + (i + 1) * totallength - dashlength;
		end.x = tl_x + w;
		cv::line(img, start, end, color, thickness);
	}
}

class TrackerWrapper {
public:
	TrackerWrapper(float threshold = 0.2f, float scale = 1.0f, bool reinit = true, TrackerParams trackerParams = TrackerParams())
		: overlapThreshold(threshold), imageScale(scale), useReinit(reinit), params(trackerParams),
		trackedFrames(0), failedFrames(0), fps(0.0f), accuracy(0.0f), robustness(0.0f) {}

	void Run(const std::vector<std::string>& frames, const std::vector<Bounds2f>& gt) {
		if (frames.size() < 30 || frames.size() != gt.size()) {
			std::cout << "Invalid data input for tracking." << std::endl;
			return;
		}

		std::ofstream fout("result.txt");
		std::ofstream fscale("scale.txt");
		std::ofstream fscore("score.txt");

		int len = (int)frames.size();
		float totalOverlap = 0.0f;
		int64_t totalTime = 0;
		const int interval = 5;

		Mat currentFrame, postFrame;
		std::vector<Bounds2f> bbs(1, Bounds2f()), bbgts(1, Bounds2f());
		auto &bb = bbs[0];
		auto &bbgt = bbgts[0];
		currentFrame = Mat(frames[0]);
		currentFrame.Resize(postFrame, imageScale, imageScale);
		bbgt = gt[0] * imageScale;

		pTracker = std::unique_ptr<Processor>(new Processor(params, 1, Vector2i(postFrame.Cols(), postFrame.Rows()), TrackMode::RGB));
		pTracker->Initialize(postFrame, bbgts);
		GImageMemoryArena.ResetImgLoadArena();

		bbgt = bbgt / imageScale;
		fout << bbgt.pMin.x << "\t" << bbgt.pMin.y << "\t" << (bbgt.pMax.x - bbgt.pMin.x) << "\t" << (bbgt.pMax.y - bbgt.pMin.y) << std::endl;
        fscale << GInfoProvider.GetCurrentScale(0) << std::endl;
        fscore << 10000 << std::endl;

		int percent = 0, a = 0, b = 0;
		std::cout << "%" << a << b;
		float score = 0.0f;
		for (int i = 1; i < len; ++i) {
			percent = (int)((double)i / len * 100);
			a = percent / 10;
			b = percent % 10;
			std::cout << "\r%" << a << b;

			currentFrame = Mat(frames[i]);
			currentFrame.Resize(postFrame, imageScale, imageScale);
			bbgt = gt[i] * imageScale;
			TimePt startTime = TimeNow();
			pTracker->Update(postFrame, bbs);
			int64_t frameTime = Duration(startTime, TimeNow());
			GImageMemoryArena.ResetImgLoadArena();

			float intersectArea = (float)Intersect(bbgt, bb).Area();
			float unionArea = (float)Union(bbgt, bb).Area();
			float overlap = unionArea > 0 ? intersectArea / unionArea : 0;

			// Track failed.
			if (overlap < overlapThreshold && useReinit) {
                bb = bb / imageScale;
			    for (int k = 0; k < interval; ++k) {
                    fout << -bb.pMin.x << "\t" << -bb.pMin.y << "\t" << -(bb.pMax.x - bb.pMin.x) << "\t" << -(bb.pMax.y - bb.pMin.y) << std::endl;
                    fscale << -GInfoProvider.GetCurrentScale(0) << std::endl;
                    fscore << -GInfoProvider.GetScorePos(0) << std::endl;
				}

				i += interval;
				if (i < len) {
					++failedFrames;
					currentFrame = std::move(Mat(frames[i]));
					currentFrame.Resize(postFrame, imageScale, imageScale);
					bbgt = gt[i] * imageScale;

					pTracker->SetReinitialize();
					pTracker->Initialize(postFrame, bbgts);
					GImageMemoryArena.ResetImgLoadArena();

					bbgt = bbgt / imageScale;
					fout << bbgt.pMin.x << "\t" << bbgt.pMin.y << "\t" << (bbgt.pMax.x - bbgt.pMin.x) << "\t" << (bbgt.pMax.y - bbgt.pMin.y) << std::endl;
                    fscale << GInfoProvider.GetCurrentScale(0) << std::endl;
                    fscore << 10000 << std::endl;
				}
				continue;
			}

			bb = bb / imageScale;
			fout << bb.pMin.x << "\t" << bb.pMin.y << "\t" << (bb.pMax.x - bb.pMin.x) << "\t" << (bb.pMax.y - bb.pMin.y) << std::endl;
            fscale << GInfoProvider.GetCurrentScale(0) << std::endl;
            fscore << GInfoProvider.GetScorePos(0) << std::endl;

			++trackedFrames;
			totalOverlap += overlap;
			totalTime += frameTime;
		}

		fps = trackedFrames / (totalTime * 1e-6f);
		accuracy = totalOverlap / trackedFrames;
		robustness = 1.0f / (failedFrames + 1.0f);

		fout.close();
		fscale.close();
	}

	inline int TrackedFrames() const { return trackedFrames; }
	inline int FailedFrames() const { return failedFrames; }
	inline float Fps() const { return fps; }
	inline float Accuracy() const { return accuracy; }
	inline float Robustness() const { return robustness; }

private:
	float overlapThreshold;
	float imageScale;
	bool useReinit;
	TrackerParams params;
	std::unique_ptr<Processor> pTracker;

	int trackedFrames;
	int failedFrames;
	float fps;
	float accuracy;
	float robustness;
};

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

static const cv::Scalar GTColor = cv::Scalar(0, 255, 0);
static const cv::Scalar TrackColor = cv::Scalar(0, 0, 255);
static const cv::Scalar ErrorColor = cv::Scalar(255, 0, 0);

int main() {
	StartSystem();

	std::string basePath = "/Users/jxzhang/Learn/handtrack/HandData";
//	std::vector<std::string> sequenceNames = { "zjx0_a0b0c0", "zjx0_a0b0c1", "zjx0_a0b1c0", "zjx0_a0b1c1",
//		"zjx0_a1b0c0" , "zjx0_a1b0c1" , "zjx0_a1b1c0" , "zjx0_a1b1c1" };
	std::vector<std::string> sequenceNames = { "crab" };
	float scale = 1.0f;
	bool produceVideo = true;

	float learnRates[3] = { 0.02f, 0.08f, 0.16f };
	TrackerParams params;
	params.UseHOG = true;
	params.UseCN = true;
	params.UseGRAY = true;
	params.UseRGB = false;
	params.UseDepthHOG = true;
	params.UseDepthGray = true;
	params.UseDepthNormal = true;
	params.UseDepthHON = true;
	params.UseDepthNormalHOG = true;
	params.NumHOGChannelsUsed = 18;
	params.PCACount = 0;
	params.UseNormalForSegment = true;
	params.UseNormalForDSST = true;
	params.UseChannelWeights = true;
	params.UseSegmentation = true;
	params.AdmmIterations = 3;
	params.Padding = 3.0f;
	params.TemplateSize = 200;
	params.GaussianSigma = 1.5f;
	params.WindowFunc = WindowType::Hann;
	params.ChebAttenuation = 45.0f;
	params.KaiserAlpha = 3.75f;
	params.WeightsLearnRates = &learnRates[0];
	params.FilterLearnRates = &learnRates[0];
	params.LearnRateNum = sizeof(learnRates) / sizeof(learnRates[0]);
	params.HistLearnRate = 0.04f;
	params.ScaleLearnRate = 0.025f;
	params.BackgroundRatio = 2.0f;
	params.HistogramBins = 16;
	params.PostRegularCount = 16;
	params.MaxSegmentArea = 1024.0f;
	params.ScaleCount = 33;
	params.ScaleSigma = 0.25f;
	params.ScaleMaxArea = 512.0f;
	params.ScaleStep = 1.02f;
	params.UpdateInterval = 0;
	params.UseScale = true;
	params.UseSmoother = false;
	params.UseFastScale = false;
	params.FailThreshold = 0.08f;

	if (produceVideo) {
		std::string sequenceName = "crab";
		std::string imgPath = basePath + "/" + sequenceName + "/img";
		std::string gtPath = basePath + "/" + sequenceName + "/gt.txt";
		std::vector<std::string> files;
		std::vector<Bounds2f> gt;
		GetImageFile(imgPath, files);
		ReadGT(gtPath, gt);
		if (files.size() == 0 || gt.size() == 0 || files.size() != gt.size()) {
			std::cout << "Invalid data for sqeuence " << sequenceName << ": image number is " <<
				files.size() << ", ground truth number is " << gt.size() << "." << std::endl;
			return 1;
		}

		std::vector<Bounds2f> resgt;
		ReadGT("result.txt", resgt);
		if (resgt.size() != gt.size()) {
			std::cout << "bounding box count not match!" << std::endl;
			return 1;
		}
		int len = (int)gt.size();
		cv::Mat currentFrame = cv::imread(files[0]);
		int outputWidth = currentFrame.cols;
		int outputHeight = currentFrame.rows;

		cv::VideoWriter voutput;
		voutput.open("p_" + sequenceName + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, cv::Size(outputWidth, outputHeight));

		int percent = 0, a = 0, b = 0;
		std::cout << "%" << a << b;
		for (int i = 0; i < len; ++i) {
			percent = (int)((double)i / len * 100);
			a = percent / 10;
			b = percent % 10;
			std::cout << "\b\b" << a << b;

			currentFrame = cv::imread(files[i]);
			Bounds2f &bbgt = gt[i];
			Bounds2f &bb = resgt[i];
			cv::Rect gtrect(bbgt.pMin.x, bbgt.pMin.y, bbgt.pMax.x - bbgt.pMin.x, bbgt.pMax.y - bbgt.pMin.y);
			cv::Rect rect(bb.pMin.x, bb.pMin.y, bb.pMax.x - bb.pMin.x, bb.pMax.y - bb.pMin.y);

			DrawDashRect(currentFrame, 2, 4, gtrect, GTColor, 2);
			cv::rectangle(currentFrame, rect, TrackColor, 2);

			if (bb.pMin.x <= 0) {
				//cv::putText(currentFrame, "Tracking failed.", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
				rect.x = -rect.x;
				rect.y = -rect.y;
				rect.height = -rect.height;
				rect.width = -rect.width;
                cv::rectangle(currentFrame, rect, ErrorColor, 2);
				for (int fk = 0; fk < 60; ++fk)
					voutput.write(currentFrame);
				i += 5;
			} else {
				voutput.write(currentFrame);
			}
		}

		voutput.release();
	} else {
		for (auto &sequenceName : sequenceNames) {
			std::string imgPath = basePath + "/" + sequenceName + "/img";
			std::string gtPath = basePath + "/" + sequenceName + "/gt.txt";
			std::vector<std::string> files;
			std::vector<Bounds2f> gt;
			GetImageFile(imgPath, files);
			ReadGT(gtPath, gt);
			if (files.size() == 0 || gt.size() == 0 || files.size() != gt.size()) {
				std::cout << "Invalid data for sqeuence " << sequenceName << ": image number is " <<
					files.size() << ", ground truth number is " << gt.size() << "." << std::endl;
				return 1;
			}

			std::cout << "sequence: " << sequenceName << std::endl;
			TrackerWrapper tracker(0.1f, scale, true, params);
			tracker.Run(files, gt);

			std::cout << std::endl
				<< "accuracy: " << tracker.Accuracy() << std::endl
				//<< "robustness: " << tracker.Robustness() << std::endl
				//<< "tracked frames: " << tracker.TrackedFrames() << std::endl
				<< "failed frames: " << tracker.FailedFrames() << std::endl
				<< "fps: " << tracker.Fps() << std::endl;
		}
	}

	CloseSystem();
	return 0;
}
