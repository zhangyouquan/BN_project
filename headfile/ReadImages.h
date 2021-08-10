#ifndef READIMAGES_H
#define READIMAGES_H

#include "common.h"
using namespace cv;

namespace third_vision
{
    const int width = 640;
    const int height = 480;
    int fps = 30;
    enum CAMERA_TYPE
    {
        Mat_LEFT,
        Mat_RIGHT,
        DEPTH_IMAGE,
        COLOR_IMAGE
    };

    void ReadImages( enum CAMERA_TYPE mode, bool flag);
    float Getdistance(const Point2f &p, rs2::depth_frame im);
};
#endif
