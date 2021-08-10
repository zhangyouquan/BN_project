#include "ReadImages.h"

namespace third_vision
{
    void ReadImages(enum CAMERA_TYPE mod,  bool flag)
    {
        const char* image_win = "show frames";
        namedWindow(image_win, WINDOW_AUTOSIZE);

        rs2::colorizer color_map;
        ///开启管道
        rs2::pipeline pipe;
        rs2::config pipe_config;

        pipe_config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        pipe_config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
        pipe_config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
        pipe_config.enable_stream(RS2_STREAM_COLOR , width, height, RS2_FORMAT_BGR8, fps);

        rs2::pipeline_profile profile = pipe.start(pipe_config);

        auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();

        /* while (cvGetWindowHandle(diaplay))
         {
             /// TODO
         }*/
        while (flag)
        {
            ///等待图像
            rs2::frameset frameset = pipe.wait_for_frames();
            ///取深度图和彩色图
            rs2::frame depth_frame = frameset.get_depth_frame().apply_filter(color_map);
            rs2::video_frame ir_frame_left = frameset.get_infrared_frame(1);
            rs2::video_frame ir_frame_right = frameset.get_infrared_frame(2);
            rs2::frame color_frame = frameset.get_color_frame();

            ///其自带sdk里为方便用opencv显示，提供了一个将frame转换为Mat类型的API
            Mat dMat_left(Size(width, height), CV_8UC1, (void*)ir_frame_left.get_data());
            Mat dMat_right(Size(width, height), CV_8UC1, (void*)ir_frame_right.get_data());
            Mat depth_image(Size(width, height),CV_8UC3, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
            Mat color_image(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);

            ///选择返回哪种图片
            switch (mod)
            {
                case Mat_LEFT:
                    imshow(image_win,dMat_left);
                    break;
                case Mat_RIGHT:
                    imshow(image_win,dMat_right);
                    break;
                case DEPTH_IMAGE:
                    imshow(image_win,depth_image);
                    break;
                case COLOR_IMAGE:
                    imshow(image_win,color_image);
                    break;
            }
            ///按键退出
            char c = waitKey(1);
            if (c == 'q' || c == 'Q' || (int)c == 27)
                break;
        }
        return;
    }
    ///这个先放这吧
    float Getdistance(const Point2f &p, rs2::depth_frame im)
    {
        return im.get_distance(p.x,p.y);
    }
};
