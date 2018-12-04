#include <fstream>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include "../cvcsrt/tracker.hpp"
#include <iostream>

using namespace cv;

void ReadGT(const std::string &fileName, std::vector<Rect> & gt) {
    std::ifstream fin(fileName);
    if (!fin.is_open()) {
        std::cout << "ReadGT: cannot open ground truth file." << std::endl;
        return;
    }
    gt.clear();
    int x, y, w, h;
    while (fin) {
        fin >> x >> y >> w >> h;
        if (fin.good())
            gt.push_back(Rect(x, y, w, h));
    }
}

inline bool IsGoodBox(const Rect &bb) { return bb.area() > 0; }

void GetImageFile(const std::string& foldername, std::vector<std::string> &files) {
    files.clear();
    DIR *dirp = opendir(foldername.c_str());
    if(dirp == nullptr) {
        std::cout << "Cannot find image folder." <<std::endl;
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
    TrackerWrapper(float threshold = 0.2f, float scale = 1.0f, bool reinit = true)
            : overlapThreshold(threshold), imageScale(scale), useReinit(reinit),
              trackedFrames(0), failedFrames(0), accuracy(0.0f), robustness(0.0f) {}

    void Run(const std::vector<std::string>& frames, const std::vector<Rect>& gt) {
        if (frames.size() < 30 || frames.size() != gt.size()) {
            std::cout << "Invalid data input for tracking." << std::endl;
            return;
        }

        std::ofstream fout("result0.txt");
        //std::ofstream fscore("score.txt");

        int len = (int)frames.size();
        float totalOverlap = 0.0f;
        int64_t totalTime = 0;
        const int interval = 5;

        Mat currentFrame, postFrame;
        std::vector<Rect> bbs(1, Rect()), bbgts(1, Rect());
        auto &bb = bbs[0];
        auto &bbgt = bbgts[0];
        currentFrame = imread(frames[0]);
        resize(currentFrame, postFrame, Size(), imageScale, imageScale);
        bbgt = gt[0];
        bbgt.x *= imageScale;
        bbgt.y *= imageScale;
        bbgt.width *= imageScale;
        bbgt.height *= imageScale;

        pTracker = cvcsrt::TrackerCSRT::create();
        pTracker->init(postFrame, bbgts[0]);

        bbgt.x /= imageScale;
        bbgt.y /= imageScale;
        bbgt.width /= imageScale;
        bbgt.height /= imageScale;

        fout << bbgt.x << "\t" << bbgt.y << "\t" << bbgt.width << "\t" << bbgt.height << std::endl;
        //fscore << "10000" << std::endl;

        int percent = 0, a = 0, b = 0;
        std::cout << "%" << a << b;
        float score = 0.0f;
        for (int i = 1; i < len; ++i) {
            percent = (int)((double)i / len * 100);
            a = percent / 10;
            b = percent % 10;
            std::cout << "\r%" << a << b;

            currentFrame = imread(frames[i]);
            resize(currentFrame, postFrame, Size(), imageScale, imageScale);
            bbgt = gt[i];
            bbgt.x *= imageScale;
            bbgt.y *= imageScale;
            bbgt.width *= imageScale;
            bbgt.height *= imageScale;

            Rect2d bbout;
            pTracker->update(postFrame, bbout);
            bbs[0] = bbout;

            float intersectArea = (float)(bbgt & bb).area();
            float unionArea = (float)(bbgt | bb).area();
            float overlap = unionArea > 0 ? intersectArea / unionArea : 0;

            // Track failed.
            if (overlap < overlapThreshold && useReinit) {
                for (int k = 0; k < interval; ++k) {
                    fout << "-1\t-1\t-1\t-1" << std::endl;
                    //fscore << std::to_string(-score) << std::endl;
                }

                i += interval;
                if (i < len) {
                    ++failedFrames;
                    currentFrame = imread(frames[i]);
                    resize(currentFrame, postFrame, Size(), imageScale, imageScale);
                    bbgt = gt[i];
                    bbgt.x *= imageScale;
                    bbgt.y *= imageScale;
                    bbgt.width *= imageScale;
                    bbgt.height *= imageScale;

                    pTracker.release();
                    pTracker = cvcsrt::TrackerCSRT::create();
                    pTracker->init(postFrame, bbgts[0]);

                    bbgt.x /= imageScale;
                    bbgt.y /= imageScale;
                    bbgt.width /= imageScale;
                    bbgt.height /= imageScale;
                    fout << bbgt.x << "\t" << bbgt.y << "\t" << bbgt.width << "\t" << bbgt.height << std::endl;
                    //fscore << "10000" << std::endl;
                }
                continue;
            }

            bbgt.x /= imageScale;
            bbgt.y /= imageScale;
            bbgt.width /= imageScale;
            bbgt.height /= imageScale;
            fout << bbgt.x << "\t" << bbgt.y << "\t" << bbgt.width << "\t" << bbgt.height << std::endl;
            //fscore << std::to_string(score) << std::endl;

            ++trackedFrames;
            totalOverlap += overlap;
        }

        accuracy = totalOverlap / trackedFrames;
        robustness = 1.0f / (failedFrames + 1.0f);

        fout.close();
        //fscore.close();
    }

    inline int TrackedFrames() const { return trackedFrames; }
    inline int FailedFrames() const { return failedFrames; }
    inline float Accuracy() const { return accuracy; }
    inline float Robustness() const { return robustness; }

private:
    float overlapThreshold;
    float imageScale;
    bool useReinit;

    Ptr<cvcsrt::TrackerCSRT> pTracker;

    int trackedFrames;
    int failedFrames;
    float accuracy;
    float robustness;
};

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

static const cv::Scalar GTColor = cv::Scalar(0, 255, 0);
static const cv::Scalar TrackColor = cv::Scalar(0, 0, 255);

int main() {

    std::string basePath = "/Users/jxzhang/Learn/handtrack/HandData";
    std::vector<std::string> sequenceNames = { "zjx0_a0b0c0", "zjx0_a0b0c1", "zjx0_a0b1c0", "zjx0_a0b1c1",
                                               "zjx0_a1b0c0" , "zjx0_a1b0c1" , "zjx0_a1b1c0" , "zjx0_a1b1c1" };
    //std::vector<std::string> sequenceNames = { "zjx0_a1b1c0" };
    float scale = 0.4f;
    for (auto &sequenceName : sequenceNames) {
        std::string imgPath = basePath + "/" + sequenceName + "/img";
        std::string gtPath = basePath + "/" + sequenceName + "/gt.txt";
        std::vector<std::string> files;
        std::vector<Rect> gt;
        GetImageFile(imgPath, files);
        ReadGT(gtPath, gt);
        if (files.size() == 0 || gt.size() == 0 || files.size() != gt.size()) {
            std::cout << "Invalid data for sqeuence " << sequenceName << ": image number is " <<
                      files.size() << ", ground truth number is " << gt.size() << "." << std::endl;
            return 1;
        }

        std::cout << "sequence: " << sequenceName << std::endl;
        TrackerWrapper tracker(0.1f, scale, true);
        tracker.Run(files, gt);

        std::cout << std::endl
                  << "accuracy: " << tracker.Accuracy() << std::endl
                  //<< "robustness: " << tracker.Robustness() << std::endl
                  //<< "tracked frames: " << tracker.TrackedFrames() << std::endl
                  << "failed frames: " << tracker.FailedFrames() << std::endl;
    }

    return 0;
}
