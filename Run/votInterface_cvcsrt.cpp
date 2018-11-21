#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include "vot.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    VOT vot;
    Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
    Mat frame;
    Rect2d rect;

    string path = vot.frame();
    frame = imread(path);
    VOTRegion region = vot.region();
    rect.x = (double)region.get_x();
    rect.y = (double)region.get_y();
    rect.width = (double)region.get_width();
    rect.height = (double)region.get_height();
    tracker->init(frame, rect);

    while (true) {
        path = vot.frame();
        if (path.empty()) break;
        frame = imread(path);
        tracker->update(frame, rect);
        float confidence = 1.0f;
        region.set_x((float)rect.x);
        region.set_y((float)rect.y);
        region.set_width((float)rect.width);
        region.set_height((float)rect.height);
        vot.report(region, confidence);
    }
}