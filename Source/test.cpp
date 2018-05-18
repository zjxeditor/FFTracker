#include "CSRT.h"
#include "Utility/Geometry.h"
#include "Utility/Mat.h"
#include "Export/CSRTracker.h"
#include <fstream>
#include <dirent.h>

using namespace CSRT;

void ReadGT(const std::string &fileName, std::vector<Bounds2i> & gt) {
    std::ifstream fin(fileName);
    if (!fin.is_open()) {
        Error("ReadGT: cannot open ground truth file.");
        return;
    }
    gt.clear();
    int x, y, w, h;
    while (fin) {
        fin >> x >> y >> w >> h;
        if (fin.good())
            gt.push_back(Bounds2i(x, y, w, h));
    }
}

inline bool IsGoodBox(const Bounds2i &bb) { return bb.Area() > 0; }

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

class TrackerWrapper {
public:
    TrackerWrapper(float threshold = 0.2f, float scale = 1.0f, bool reinit = true, CSRTrackerParams trackerParams = CSRTrackerParams())
            : overlapThreshold(threshold), imageScale(scale), useReinit(reinit), params(trackerParams),
              trackedFrames(0), failedFrames(0), fps(0.0f), accuracy(0.0f), robustness(0.0f) {}

    void Run(const std::vector<std::string>& frames, const std::vector<Bounds2i>& gt) {
        if (frames.size() < 30 || frames.size() != gt.size()) {
            std::cout << "Invalid data input for tracking." << std::endl;
            return;
        }

        std::ofstream fout("./result.txt");
        std::ofstream fscore("./score.txt");

        int len = (int)frames.size();
        float totalOverlap = 0.0f;
        int64_t totalTime = 0;
        const int interval = 5;

        Mat currentFrame;
        Bounds2i bb, bbgt;
        currentFrame = Mat(frames[0]);
        bbgt = gt[0];
        pTracker = std::unique_ptr<CSRTracker>(new CSRTracker(currentFrame.Rows(), currentFrame.Cols(), imageScale, params));
        pTracker->SetDrawMode(true, 255, 0, 0, 0.5f);
        Bounds initbbgt;
        memcpy(&initbbgt, &bbgt, sizeof(Bounds));
        pTracker->Initialize(currentFrame.Data(), 3, initbbgt);
        GImageMemoryArena.ResetImgLoadArena();

        fout << bbgt.pMin.x << "\t" << bbgt.pMin.y << "\t" << (bbgt.pMax.x - bbgt.pMin.x) << "\t" << (bbgt.pMax.y - bbgt.pMin.y) << std::endl;
        fscore << "10000" << std::endl;

        int percent = 0, a = 0, b = 0;
        std::cout << a << b << "%\r";
        float score = 0.0f;
        for (int i = 1; i < len; ++i) {
            percent = (int)((double)i / len * 100);
            a = percent / 10;
            b = percent % 10;
            std::cout << a << b << "%\r";
            std::cout.flush();

            currentFrame = Mat(frames[i]);
            bbgt = gt[i];

            TimePt startTime = TimeNow();
            Bounds outputbb;
            bool res = pTracker->Update(currentFrame.Data(), 3, outputbb, score);
            memcpy(&bb, &outputbb, sizeof(Bounds));
            int64_t frameTime = Duration(startTime, TimeNow());
            GImageMemoryArena.ResetImgLoadArena();

            float intersectArea = (float)Intersect(bbgt * imageScale, bb * imageScale).Area();
            float unionArea = (float)Union(bbgt * imageScale, bb * imageScale).Area();
            float overlap = unionArea > 0 ? intersectArea / unionArea : 0;

            // Track failed.
            if ((overlap < overlapThreshold) && useReinit) {
                for (int k = 0; k < interval; ++k) {
                    fout << "-1\t-1\t-1\t-1" << std::endl;
                    fscore << std::to_string(-score) << std::endl;
                }

                i += interval;
                if (i < len) {
                    ++failedFrames;
                    currentFrame = std::move(Mat(frames[i]));
                    bbgt = gt[i];

                    pTracker->SetReinitialize();
                    Bounds initbbgt;
                    memcpy(&initbbgt, &bbgt, sizeof(Bounds));
                    pTracker->Initialize(currentFrame.Data(), 3, initbbgt);
                    GImageMemoryArena.ResetImgLoadArena();

                    fout << bbgt.pMin.x << "\t" << bbgt.pMin.y << "\t" << (bbgt.pMax.x - bbgt.pMin.x) << "\t" << (bbgt.pMax.y - bbgt.pMin.y) << std::endl;
                    fscore << "10000" << std::endl;
                }
                continue;
            }

            fout << bb.pMin.x << "\t" << bb.pMin.y << "\t" << (bb.pMax.x - bb.pMin.x) << "\t" << (bb.pMax.y - bb.pMin.y) << std::endl;
            fscore << std::to_string(score) << std::endl;

            ++trackedFrames;
            totalOverlap += overlap;
            totalTime += frameTime;
        }
        std::cout << std::endl;

        fps = trackedFrames / (totalTime * 1e-6f);
        accuracy = totalOverlap / trackedFrames;
        robustness = 1.0f / (failedFrames + 1.0f);

        fout.close();
        fscore.close();
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
    CSRTrackerParams params;
    std::unique_ptr<CSRTracker> pTracker;

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

int main() {
    StartSystem();

    std::string basePath = "/Users/jxzhang/Desktop/HandData";
    std::vector<std::string> sequenceNames = { "zjx0_a0b0c0", "zjx0_a0b0c1", "zjx0_a0b1c0", "zjx0_a0b1c1",
                                               "zjx0_a1b0c0" , "zjx0_a1b0c1" , "zjx0_a1b1c0" , "zjx0_a1b1c1" };
    //std::vector<std::string> sequenceNames = { "zjx0_a1b0c1" };
    float scale = 0.4f;
    bool produceVideo = false;

    CSRTrackerParams params;
    params.UseHOG = true;
    params.UseCN = true;
    params.UseGRAY = true;
    params.UseRGB = false;
    params.NumHOGChannelsUsed = 18;
    params.UsePCA = false;
    params.PCACount = 20;
    params.UseChannelWeights = true;
    params.UseSegmentation = true;
    params.AdmmIterations = 4;
    params.Padding = 3.0f;
    params.TemplateSize = 200;
    params.GaussianSigma = 1.0f;
    params.WindowFunc = CSRTWindowType::Hann;
    params.ChebAttenuation = 45.0f;
    params.KaiserAlpha = 3.75f;
    params.WeightsLearnRate = 0.02f;
    params.FilterLearnRate = 0.02f;
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
    params.UpdateInterval = 1;
    params.PeakRatio = 0.1f;
    params.PSRThreshold = 10.0f;

    if(produceVideo) {
        std::string sequenceName = "zjx0_a1b0c1";
        std::string imgPath = basePath + "/" + sequenceName + "/img";
        std::string gtPath = basePath + "/" + sequenceName + "/gt.txt";
        std::vector<std::string> files;
        std::vector<Bounds2i> gt;
        GetImageFile(imgPath, files);
        ReadGT(gtPath, gt);
        if (files.size() == 0 || gt.size() == 0 || files.size() != gt.size()) {
            std::cout << "Invalid data for sqeuence " << sequenceName << ": image number is " <<
                      files.size() << ", ground truth number is " << gt.size() << "." << std::endl;
            return 1;
        }

        std::vector<Bounds2i> resgt;
        ReadGT("./result.txt", resgt);
        if(resgt.size() != gt.size()) {
            std::cout << "bounding box count not match!" << std::endl;
            return 1;
        }
        int len = (int)gt.size();
        cv::VideoWriter voutput;
        voutput.open("p_" + sequenceName + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, cv::Size(720, 1280));

        cv::Mat currentFrame;
        int percent = 0, a = 0, b = 0;
        std::cout << a << b << "%\r";
        for (int i = 0; i < len; ++i) {
            percent = (int)((double)i / len * 100);
            a = percent / 10;
            b = percent % 10;
            std::cout << a << b << "%\r";
            std::cout.flush();

            currentFrame = cv::imread(files[i]);
            Bounds2i &bbgt = gt[i];
            Bounds2i &bb = resgt[i];
            cv::Rect gtrect(bbgt.pMin.x, bbgt.pMin.y, bbgt.pMax.x - bbgt.pMin.x, bbgt.pMax.y - bbgt.pMin.y);
            cv::Rect rect(bb.pMin.x, bb.pMin.y, bb.pMax.x - bb.pMin.x, bb.pMax.y - bb.pMin.y);

            cv::rectangle(currentFrame, gtrect, GTColor, 2);
            cv::rectangle(currentFrame, rect, TrackColor, 2);

            if(bb.pMin.x <= 0) {
                cv::putText(currentFrame, "Tracking failed.", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
                for (int fk = 0; fk < 60; ++fk)
                    voutput.write(currentFrame);
                i += 5;
            } else {
                voutput.write(currentFrame);
            }
        }
        std::cout << std::endl;
        voutput.release();
    } else {
        for (auto &sequenceName : sequenceNames) {
            std::string imgPath = basePath + "/" + sequenceName + "/img";
            std::string gtPath = basePath + "/" + sequenceName + "/gt.txt";
            std::vector<std::string> files;
            std::vector<Bounds2i> gt;
            GetImageFile(imgPath, files);
            ReadGT(gtPath, gt);
            if (files.size() == 0 || gt.size() == 0 || files.size() != gt.size()) {
                std::cout << "Invalid data for sqeuence " << sequenceName << ": image number is " <<
                          files.size() << ", ground truth number is " << gt.size() << "." << std::endl;
                return 1;
            }

            std::cout << std::endl << "sequence: " << sequenceName << std::endl;
            TrackerWrapper tracker(0.1f, scale, true, params);
            tracker.Run(files, gt);

            std::cout << "accuracy: " << tracker.Accuracy() << std::endl
                      << "robustness: " << tracker.Robustness() << std::endl
                      << "fps: " << tracker.Fps() << std::endl
                      << "tracked frames: " << tracker.TrackedFrames() << std::endl
                      << "failed frames: " << tracker.FailedFrames() << std::endl;
        }
    }

    CloseSystem();
    return 0;
}
