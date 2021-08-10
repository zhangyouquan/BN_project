/**

读图像测试，选择输出类型某种图像类型，包括深度图、颜色图、左红外图、右红外图。按q, Q、Esc退出。

*/

#include "ReadImages.h"

///想要一个窗口把所有图像全部显示出来
//void cvShowMultiImages(char* title, int nArgs, ...)
//{
//    // img - Used for getting the arguments
//    IplImage* img;
//    // DispImage - the image in which all the input images are to be copied
//    IplImage* DispImage;
//    int size;    // size - the size of the images in the window
//    int ind;        // ind - the index of the image shown in the window
//    int x, y;    // x,y - the coordinate of top left coner of input images
//    int w, h;    // w,h - the width and height of the image
//
//    // r - Maximum number of images in a column
//    // c - Maximum number of images in a row
//    int r, c;
//
//    // scale - How much we have to resize the image
//    float scale;
//    // max - Max value of the width and height of the image
//    int max;
//    // space - the spacing between images
//    int space;
//
//    // If the number of arguments is lesser than 0 or greater than 12
//    // return without displaying
//    if(nArgs <= 0) {
//        printf("Number of arguments too small..../n");
//        return;
//    }
//    else if(nArgs > 12) {
//        printf("Number of arguments too large..../n");
//        return;
//    }
//        // Determine the size of the image,
//        // and the number of rows/cols
//        // from number of arguments
//    else if (nArgs == 1) {
//        r = c = 1;
//        size = 300;
//    }
//    else if (nArgs == 2) {
//        r = 2; c = 1;
//        size = 300;
//    }
//    else if (nArgs == 3 || nArgs == 4) {
//        r = 2; c = 2;
//        size = 300;
//    }
//    else if (nArgs == 5 || nArgs == 6) {
//        r = 3; c = 2;
//        size = 200;
//    }
//    else if (nArgs == 7 || nArgs == 8) {
//        r = 4; c = 2;
//        size = 200;
//    }
//    else {
//        r = 4; c = 3;
//        size = 150;
//    }
//    // Create a new 3 channel image to show all the input images
//    DispImage = cvCreateImage( cvSize(60 + size*r, 20 + size*c), IPL_DEPTH_8U, 3 );
//    // Used to get the arguments passed
//    va_list args;
//    va_start(args, nArgs);
//
//    // Loop for nArgs number of arguments
//    space = 20;
//    for (ind = 0, x = space, y = space; ind < nArgs; ind++, x += (space + size)) {
//
//
//        // Get the Pointer to the IplImage
//        img = va_arg(args, IplImage*);
//
//        // Check whether it is NULL or not
//        // If it is NULL, release the image, and return
//        if(img == 0) {
//            printf("Invalid arguments");
//            cvReleaseImage(&DispImage);
//            return;
//        }
//
//        // Find the width and height of the image
//        w = img->width;
//        h = img->height;
//
//        // Find whether height or width is greater in order to resize the image
//        max = (w > h)? w: h;
//
//        // Find the scaling factor to resize the image
//        scale = (float) ( (float) max / size );
//        // Used to Align the images
//        // i.e. Align the image to next row
//        if( ind % r == 0 && x!= space) {
//            x  = space;
//            y += space + size;
//        }
//
//        // Set the image ROI to display the current image
//        cvSetImageROI(DispImage, cvRect(x, y, (int)( w/scale ), (int)( h/scale )));
//        // Resize the input image and copy the it to the Single Big Image
//        cvResize(img, DispImage);
//        // Reset the ROI in order to display the next image
//        cvResetImageROI(DispImage);
//    }
//
//
//    // Create a new window, and show the Single Big Image
//    //cvNamedWindow( title, 1 );
//    cvShowImage( title, DispImage);
//    // End the number of arguments
//    va_end(args);
//
//
//    // Release the Image Memory
//    cvReleaseImage(&DispImage);
//}

#include <queue>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <cmath>
#include "yolo_v2_class.hpp"

#ifdef OPENCV
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
/*
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
*/
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH
using namespace std;


vector<string> split(const string&s,char sepeartor)
{
    vector<string> split_vector;
    int subinit=0;
    for (int id=0;id!=s.length();id++)
    {
        if (s[id]==sepeartor)
        {
            split_vector.push_back(s.substr(subinit,id-subinit));
            subinit=id+1;
        }
    }
    split_vector.push_back(s.substr(subinit,s.length()-subinit));
    return split_vector;
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
                int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                          cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                          color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}
#endif    // OPENCV

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
                  << ", w = " << i.w << ", h = " << i.h
                  << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

int main(int argc,char** argv)
{
    std::string  names_file = "../coco.names";
    std::string  cfg_file = "../cfg/yolov4.cfg";
    std::string  weights_file = "../yolov4.weights";
    std::string filename;

    if (argc > 4) {    //voc.names yolo-voc.cfg yolo-voc.weights
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
        filename = argv[4];
    }
    float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.2;

    /*int c;
    third_vision::CAMERA_TYPE mod;
    std::cout << "0:左图； 1:右图 2:RGB 3: 深度图。" << std::endl
                << "请输入你想要看的图片类型(int类整数)： " << std::endl;
    std::cin >> c;
    switch (c)
    {
        case 0:
            mod = third_vision::Mat_LEFT;
            break;
        case 1:
            mod = third_vision::Mat_RIGHT;
            break;
        case 2:
            mod = third_vision::COLOR_IMAGE;
            break;
        case 3:
            mod = third_vision::DEPTH_IMAGE;
            break;

    }*/
    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);
    bool const send_network = false;        // true - for remote detection
    bool const use_kalman_filter = false;   // true - for stationary camera

    bool detection_sync = false;             // true - for video-file
#ifdef TRACK_OPTFLOW    // for slow GPU
    detection_sync = false;
    Tracker_optflow tracker_flow;
    //detector.wait_stream = true;
#endif  // TRACK_OPTFLOW
    ///third_vision::ReadImages(mod,true);
    while(true)
    {
        std::cout << "请输入相机指令（camera）: ";
        if(filename.size() == 0) std::cin >> filename;
        if (filename.size() == 0) break;

        try
        {
#ifdef OPENCV
            preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
            bool show_small_boxes = false;

            std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
            std::string const protocol = filename.substr(0, 5);
            if (filename == "camera" )   //image or camera
            {
                cv::Mat cur_frame;
                std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
                std::atomic<int> current_fps_cap(0), current_fps_det(0);
                std::atomic<bool> exit_flag(false);
                std::chrono::steady_clock::time_point steady_start, steady_end;
                int video_fps = 30;
                bool camera = false;

                track_kalman_t track_kalman;

                /// 判断是否有设备接入。
                rs2::context ctx;
                auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
                if (list.size() == 0)
                    throw std::runtime_error("相机未连接？或者连接失败了？");
                rs2::device dev = list.front();

                const char *image_win = "show frames";
                namedWindow(image_win, WINDOW_AUTOSIZE);

                ///这个是实现让深度图看起来有色彩的。
                rs2::colorizer color_map;

                ///开启管道
                rs2::pipeline pipe;
                rs2::config pipe_config;

                const int width = 640;
                const int height = 480;
                int fps = 30;

                pipe_config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
                pipe_config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
                pipe_config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
                pipe_config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
                pipe_config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
                pipe_config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

                rs2::pipeline_profile profile = pipe.start(pipe_config);
                ///定义一个变量去转换深度到距离
                ///float depth_clipping_distance = 1.f;
                ///声明数据流
                auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
                auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

                while (cvGetWindowHandle(image_win))
                {
                    ///等待图像
                    rs2::frameset frameset = pipe.wait_for_frames();

                    ///对齐操作
                    rs2::align align_to_depth(RS2_STREAM_DEPTH);
                    frameset = align_to_depth.process(frameset);

                    /// 获取IMU数据
                    if (rs2::motion_frame accel_frame = frameset.first_or_default(RS2_STREAM_ACCEL)) {
                        rs2_vector accel_sample = accel_frame.get_motion_data();
                        ///std::cout << "Accel:" << accel_sample.x << ", " << accel_sample.y << ", " << accel_sample.z << std::endl;
                    }
                    if (rs2::motion_frame gyro_frame = frameset.first_or_default(RS2_STREAM_GYRO)) {
                        rs2_vector gyro_sample = gyro_frame.get_motion_data();
                        ///std::cout << "Gyro:" << gyro_sample.x << ", " << gyro_sample.y << ", " << gyro_sample.z << std::endl;
                    }

                    ///取深度图和彩色图
                    ///rs2::frame depth_frame = frameset.get_depth_frame().apply_filter(color_map);
                    rs2::frame depth_frame = frameset.get_depth_frame();
                    rs2::video_frame ir_frame_left = frameset.get_infrared_frame(1);
                    rs2::video_frame ir_frame_right = frameset.get_infrared_frame(2);
                    rs2::frame color_frame = frameset.get_color_frame();


                    ///其自带sdk里为方便用opencv显示，提供了一个将frame转换为Mat类型的API
                    Mat dMat_left(Size(width, height), CV_8UC1, (void *) ir_frame_left.get_data());
                    Mat dMat_right(Size(width, height), CV_8UC1, (void *) ir_frame_right.get_data());
                    Mat depth_image(Size(width, height), CV_16U, (void *) depth_frame.get_data(), Mat::AUTO_STEP);
                    Mat color_image(Size(width, height), CV_8UC3, (void *) color_frame.get_data(), Mat::AUTO_STEP);

                    float x = width / 2.0;
                    float y = height / 2.0;

                    float distance_centor = depth_image.at<uint16_t>(Point(x, y));
                    std::cout << "相机中心距离检测物体的距离为： " << distance_centor / 1000 << " m." << std::endl;

                    color_image.copyTo(cur_frame);
                    camera = true;

                    cv::Size const frame_size = cur_frame.size();
                    std::cout << "\n Video size: " << frame_size << std::endl;

                    struct detection_data_t {
                        cv::Mat cap_frame;
                        std::shared_ptr<image_t> det_image;
                        std::vector<bbox_t> result_vec;
                        cv::Mat draw_frame;
                        bool new_detection;
                        uint64_t frame_id;
                        bool exit_flag;
                        cv::Mat zed_cloud;
                        std::queue<cv::Mat> track_optflow_queue;
                        detection_data_t() : exit_flag(false), new_detection(false) {}
                    };
                    const bool sync = detection_sync; // sync data exchange
                    send_one_replaceable_object_t<detection_data_t> cap2prepare(sync), cap2draw(sync),
                            prepare2detect(sync), detect2draw(sync), draw2show(sync), draw2write(sync), draw2net(sync);

                    std::thread t_cap, t_prepare, t_detect, t_post, t_draw, t_write, t_network;

                    // capture new video-frame
                    if (t_cap.joinable()) t_cap.join();
                    t_cap = std::thread([&]()
                                        {
                                            uint64_t frame_id = 0;
                                            detection_data_t detection_data;
                                            do {
                                                detection_data = detection_data_t();
                                                if (camera)
                                                {
                                                    detection_data.cap_frame = color_image;
                                                    detection_data.zed_cloud = depth_image;
                                                }
                                                fps_cap_counter++;
                                                detection_data.frame_id = frame_id++;
                                                if (detection_data.cap_frame.empty() || exit_flag) {
                                                    std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size() << std::endl;
                                                    detection_data.exit_flag = true;
                                                    detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
                                                }

                                                if (!detection_sync) {
                                                    cap2draw.send(detection_data);       // skip detection
                                                }
                                                cap2prepare.send(detection_data);
                                            } while (!detection_data.exit_flag);
                                            std::cout << " t_cap exit \n";
                                        });
                    // pre-processing video frame (resize, convertion)
                    t_prepare = std::thread([&]()
                                            {
                                                std::shared_ptr<image_t> det_image;
                                                detection_data_t detection_data;
                                                do {
                                                    detection_data = cap2prepare.receive();

                                                    det_image = detector.mat_to_image_resize(detection_data.cap_frame);
                                                    detection_data.det_image = det_image;
                                                    prepare2detect.send(detection_data);    // detection

                                                } while (!detection_data.exit_flag);
                                                std::cout << " t_prepare exit \n";
                                            });
                    // detection by Yolo
                    if (t_detect.joinable()) t_detect.join();
                    t_detect = std::thread([&]()
                                           {
                                               std::shared_ptr<image_t> det_image;
                                               detection_data_t detection_data;
                                               do {
                                                   detection_data = prepare2detect.receive();
                                                   det_image = detection_data.det_image;
                                                   std::vector<bbox_t> result_vec;

                                                   if(det_image)
                                                       result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height, thresh, true);  // true
                                                   fps_det_counter++;
                                                   //std::this_thread::sleep_for(std::chrono::milliseconds(150));

                                                   detection_data.new_detection = true;
                                                   detection_data.result_vec = result_vec;
                                                   detect2draw.send(detection_data);
                                               } while (!detection_data.exit_flag);
                                               std::cout << " t_detect exit \n";
                                           });

                    // draw rectangles (and track objects)
                    t_draw = std::thread([&]()
                                         {
                                             std::queue<cv::Mat> track_optflow_queue;
                                             detection_data_t detection_data;
                                             do {
                                                 // for Video-file
                                                 if (detection_sync) {
                                                     detection_data = detect2draw.receive();
                                                 }
                                                     // for Video-camera
                                                 else
                                                 {
                                                     // get new Detection result if present
                                                     if (detect2draw.is_object_present()) {
                                                         cv::Mat old_cap_frame = detection_data.cap_frame;   // use old captured frame
                                                         detection_data = detect2draw.receive();
                                                         if (!old_cap_frame.empty()) detection_data.cap_frame = old_cap_frame;
                                                     }
                                                         // get new Captured frame
                                                     else {
                                                         std::vector<bbox_t> old_result_vec = detection_data.result_vec; // use old detections
                                                         detection_data = cap2draw.receive();
                                                         detection_data.result_vec = old_result_vec;
                                                     }
                                                 }

                                                 cv::Mat cap_frame = detection_data.cap_frame;
                                                 cv::Mat draw_frame = detection_data.cap_frame.clone();
                                                 std::vector<bbox_t> result_vec = detection_data.result_vec;
#ifdef TRACK_OPTFLOW
                                                 if (detection_data.new_detection) {
                        tracker_flow.update_tracking_flow(detection_data.cap_frame, detection_data.result_vec);
                        while (track_optflow_queue.size() > 0) {
                            draw_frame = track_optflow_queue.back();
                            result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), false);
                            track_optflow_queue.pop();
                        }
                    }
                    else {
                        track_optflow_queue.push(cap_frame);
                        result_vec = tracker_flow.tracking_flow(cap_frame, false);
                    }
                    detection_data.new_detection = true;    // to correct kalman filter
#endif //TRACK_OPTFLOW
// track ID by using kalman filter
                                                 if (use_kalman_filter) {
                                                     if (detection_data.new_detection) {
                                                         result_vec = track_kalman.correct(result_vec);
                                                     }
                                                     else {
                                                         result_vec = track_kalman.predict();
                                                     }
                                                 }
                                                     // track ID by using custom function
                                                 else {
                                                     int frame_story = std::max(5, current_fps_cap.load());
                                                     result_vec = detector.tracking_id(result_vec, true, frame_story, 40);
                                                 }

                                                 draw_boxes(draw_frame, result_vec, obj_names, current_fps_det, current_fps_cap);

                                                 detection_data.result_vec = result_vec;
                                                 detection_data.draw_frame = draw_frame;
                                                 draw2show.send(detection_data);
                                                 if (send_network) draw2net.send(detection_data);
                                             } while (!detection_data.exit_flag);
                                             std::cout << " t_draw exit \n";
                                         });
                    // send detection to the network
                    t_network = std::thread([&]()
                                            {
                                                if (send_network) {
                                                    detection_data_t detection_data;
                                                    do {
                                                        detection_data = draw2net.receive();

                                                        detector.send_json_http(detection_data.result_vec, obj_names, detection_data.frame_id, filename);

                                                    } while (!detection_data.exit_flag);
                                                }
                                                std::cout << " t_network exit \n";
                                            });


                    // show detection
                    detection_data_t detection_data;
                    do {
                        steady_end = std::chrono::steady_clock::now();
                        float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                        if (time_sec >= 1) {
                            current_fps_det = fps_det_counter.load() / time_sec;
                            current_fps_cap = fps_cap_counter.load() / time_sec;
                            steady_start = steady_end;
                            fps_det_counter = 0;
                            fps_cap_counter = 0;
                        }

                        detection_data = draw2show.receive();
                        cv::Mat draw_frame = detection_data.draw_frame;

                        //if (extrapolate_flag) {
                        //    cv::putText(draw_frame, "extrapolate", cv::Point2f(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
                        //}

                        cv::imshow(image_win, draw_frame);
                        filename.replace(filename.end()-4, filename.end(), "_yolov4_out.jpg");

                        int key = cv::waitKey(3);    // 3 or 16ms
                        if (key == 'f') show_small_boxes = !show_small_boxes;
                        if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
                        //if (key == 'e') extrapolate_flag = !extrapolate_flag;
                        if (key == 27) { exit_flag = true;}

                        //std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap << std::endl;
                    } while (!detection_data.exit_flag);
                    std::cout << " show detection exit \n";
                    // wait for all threads
                    if (t_cap.joinable()) t_cap.join();
                    if (t_prepare.joinable()) t_prepare.join();
                    if (t_detect.joinable()) t_detect.join();
                    if (t_post.joinable()) t_post.join();
                    if (t_draw.joinable()) t_draw.join();
                    if (t_write.joinable()) t_write.join();
                    if (t_network.joinable()) t_network.join();

                    break;
                }

            }


#else   // OPENCV
            //std::vector<bbox_t> result_vec = detector.detect(filename);

            auto img = detector.load_image(filename);
            std::vector<bbox_t> result_vec = detector.detect(img);
            detector.free_image(img);
            show_console_result(result_vec, obj_names);
#endif  // OPENCV
        }
        catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { std::cerr << "unknown exception \n"; getchar(); }
        filename.clear();
    }
    return 0;
}

