//
// Object tracking demo.
// This demo relies OpenCV to fetch camera data and do image preprocessing. Once running this demo, place the
// target tracking object in the circle, press "=" button to enlarge the circle and press "-" button to shrink
// the circle. Then press the space button to start tracking. At any time, you can press "esc" button to quit
// the tracking process.
//

#include "Export/CSRTracker.h"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

typedef std::chrono::high_resolution_clock::time_point TimePt;
inline TimePt TimeNow() { return std::chrono::high_resolution_clock::now(); }
inline int64_t Duration(TimePt start, TimePt end) { return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); }

int main() {
    CSRT::StartTheSystem();

    float rawScale = 0.6f;
    float scale = 0.4f;
    int radius = 70;
    int targetCount = 1;
    int trackedFrames = 0;
    float fps = 0.0f;
    int64_t totalTime = 0;

    // Configure camera and OpenCV
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        CSRT::Warning("Cannot open the default camera.");
        CSRT::CloseTheSystem();
        return -1;
    }
    int frameWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frameHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int rawWidth = (int)(frameWidth * rawScale);
    int rawHeight = (int)(frameHeight * rawScale);
    int postWidth = (int)(rawWidth * scale);
    int postHeight = (int)(rawHeight * scale);
    std::stringstream ss;
    ss << "original frame size: " << frameWidth << " x " << frameHeight;
    CSRT::Info(ss.str().c_str());
    ss.clear();
    ss << "raw frame size: " << rawWidth << " x " << rawHeight;
    CSRT::Info(ss.str().c_str());
    ss.clear();
    ss << "post frame size: " << postWidth << " x " << postHeight;
    CSRT::Info(ss.str().c_str());
    cv::Mat frame(frameHeight, frameWidth, CV_8UC3);
    cv::Mat rawFrame(rawHeight, rawWidth, CV_8UC3);
    cv::Mat postFrame(postHeight, postWidth, CV_8UC3);
    cv::Scalar WaitColor = cv::Scalar(0, 0, 255);
    cv::Scalar TrackColor = cv::Scalar(0, 255, 0);
    std::string WindowName = "Tracker";
    cv::namedWindow(WindowName, 1);

    // Create tracker.
    CSRT::CSRTrackerParams params;
    params.UpdateInterval = 0;
    CSRT::CSRTracker tracker(postHeight, postWidth, targetCount, CSRT::CSRTrackMode::RGB, params);

    // Create initial bounding boxes and output bounding boxes.
    std::vector<CSRT::Bounds> rawInitbbs(targetCount, CSRT::Bounds());
    float deltaX = ((float)rawWidth) / (targetCount + 1);
    float centerY = rawHeight / 2.0f;
    float centerX = 0.0f;
    float maxRadius = std::min(deltaX, centerY);
    float minRadius = 20;
    if(radius > maxRadius) radius = maxRadius;
    if(radius < minRadius) radius = minRadius;
    for (int i = 0; i < targetCount; ++i) {
        centerX += deltaX;
        rawInitbbs[i].x0 = centerX - radius;
        rawInitbbs[i].y0 = centerY - radius;
        rawInitbbs[i].x1 = centerX + radius;
        rawInitbbs[i].y1 = centerY + radius;
    }
    std::vector<CSRT::Bounds> postInitbbs(targetCount, CSRT::Bounds());
    for (int i = 0; i < targetCount; ++i) {
        postInitbbs[i].x0 = rawInitbbs[i].x0 * scale;
        postInitbbs[i].x1 = rawInitbbs[i].x1 * scale;
        postInitbbs[i].y0 = rawInitbbs[i].y0 * scale;
        postInitbbs[i].y1 = rawInitbbs[i].y1 * scale;
    }
    std::vector<CSRT::Bounds> rawOutputbbs(targetCount, CSRT::Bounds());
    std::vector<CSRT::Bounds> postOutputbbs(targetCount, CSRT::Bounds());

    // Main loop.
    bool started = false;
    TimePt startTime = TimeNow();
    while (true) {
        // Fetch data
        cap >> frame;
        cv::resize(frame, rawFrame, cv::Size(rawWidth, rawHeight));
        cv::resize(rawFrame, postFrame, cv::Size(postWidth, postHeight));

        // Get operation
        int key = cv::waitKey(1);
        if (key == 27) break;
        if(!started) {
            if(key == 61) {
                radius += 4;
                if(radius >= maxRadius) radius = maxRadius;
                float centery = rawHeight / 2.0f;
                float centerx = 0.0f;
                for (int i = 0; i < targetCount; ++i) {
                    centerx += deltaX;
                    rawInitbbs[i].x0 = centerx - radius;
                    rawInitbbs[i].y0 = centery - radius;
                    rawInitbbs[i].x1 = centerx + radius;
                    rawInitbbs[i].y1 = centery + radius;
                    postInitbbs[i].x0 = rawInitbbs[i].x0 * scale;
                    postInitbbs[i].x1 = rawInitbbs[i].x1 * scale;
                    postInitbbs[i].y0 = rawInitbbs[i].y0 * scale;
                    postInitbbs[i].y1 = rawInitbbs[i].y1 * scale;
                }
            } else if (key == 45) {
                radius -= 4;
                if(radius <= minRadius) radius = minRadius;
                float centery = rawHeight / 2.0f;
                float centerx = 0.0f;
                for (int i = 0; i < targetCount; ++i) {
                    centerx += deltaX;
                    rawInitbbs[i].x0 = centerx - radius;
                    rawInitbbs[i].y0 = centery - radius;
                    rawInitbbs[i].x1 = centerx + radius;
                    rawInitbbs[i].y1 = centery + radius;
                    postInitbbs[i].x0 = rawInitbbs[i].x0 * scale;
                    postInitbbs[i].x1 = rawInitbbs[i].x1 * scale;
                    postInitbbs[i].y0 = rawInitbbs[i].y0 * scale;
                    postInitbbs[i].y1 = rawInitbbs[i].y1 * scale;
                }
            } else if(key == 32) {
                // Start tracking
                started = true;
                tracker.Initialize(postFrame.data, 3, &postInitbbs[0]);
                for (int i = 0; i < targetCount; ++i) {
                    int centerx = (int)std::floor((rawInitbbs[i].x0 + rawInitbbs[i].x1) / 2);
                    int centery = (int)std::floor((rawInitbbs[i].y0 + rawInitbbs[i].y1) / 2);
                    int newRadius = (int)std::floor(std::max(rawInitbbs[i].x1 - rawInitbbs[i].x0, rawInitbbs[i].y1 - rawInitbbs[i].y0) / 2);
                    cv::circle(rawFrame, cv::Point(centerx, centery), newRadius, TrackColor, 2);
                }
            }
        }

        if (started) {
            TimePt sTime = TimeNow();
            tracker.Update(postFrame.data, 3, &postOutputbbs[0]);
            int64_t frameTime = Duration(sTime, TimeNow());
            ++trackedFrames;
            totalTime += frameTime;
            for (int i = 0; i < targetCount; ++i) {
                rawOutputbbs[i].x0 = postOutputbbs[i].x0 / scale;
                rawOutputbbs[i].x1 = postOutputbbs[i].x1 / scale;
                rawOutputbbs[i].y0 = postOutputbbs[i].y0 / scale;
                rawOutputbbs[i].y1 = postOutputbbs[i].y1 / scale;
            }
            for (int i = 0; i < targetCount; ++i) {
                int centerx = (int)std::floor((rawOutputbbs[i].x0 + rawOutputbbs[i].x1) / 2);
                int centery = (int)std::floor((rawOutputbbs[i].y0 + rawOutputbbs[i].y1) / 2);
                int newRadius = (int)std::floor(std::max(rawOutputbbs[i].x1 - rawOutputbbs[i].x0, rawOutputbbs[i].y1 - rawOutputbbs[i].y0) / 2);
                cv::circle(rawFrame, cv::Point(centerx, centery), newRadius, TrackColor, 2);
            }
        } else {
            for (int i = 0; i < targetCount; ++i) {
                int centerx = (int)std::floor((rawInitbbs[i].x0 + rawInitbbs[i].x1) / 2);
                int centery = (int)std::floor((rawInitbbs[i].y0 + rawInitbbs[i].y1) / 2);
                int newRadius = (int)std::floor(std::max(rawInitbbs[i].x1 - rawInitbbs[i].x0, rawInitbbs[i].y1 - rawInitbbs[i].y0) / 2);
                cv::circle(rawFrame, cv::Point(centerx, centery), newRadius, WaitColor, 2);
            }
        }

        cv::flip(rawFrame, rawFrame, 1);
        cv::imshow(WindowName, rawFrame);
    }

    fps = trackedFrames / (totalTime * 1e-6f);
    CSRT::Info(("fps: " + std::to_string(fps)).c_str());

    cap.release();
    cv::destroyAllWindows();
    CSRT::CloseTheSystem();
    return 0;
}


