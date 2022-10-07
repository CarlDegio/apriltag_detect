#include <iostream>
#include "opencv2/opencv.hpp"
#include <apriltag/tagStandard41h12.h>
#include <apriltag/apriltag_pose.h>
using namespace std;

apriltag_detector_t* create_detector()
{
    apriltag_family_t* tag_family = tagStandard41h12_create();
    apriltag_detector_t *tag_detector = apriltag_detector_create();
    apriltag_detector_add_family(tag_detector, tag_family);
    return tag_detector;
}

int main(int argc, char** argv)
{
    cv::Mat frame,frame_gray;

    apriltag_detector_t *tag_detector;
    tag_detector=create_detector();

    frame=cv::imread("1_1.jpg");
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    image_u8_t im = {frame_gray.cols,
                     frame_gray.rows,
                     frame_gray.cols,
                     frame_gray.data
    };

    zarray_t *detections = apriltag_detector_detect(tag_detector, &im);

    apriltag_detection_info_t info;
    info.fx=299.5237348696959;
    info.fy=299.3479963553295;
    info.cx=325.1746597360504;
    info.cy=246.0877039461241;
    info.tagsize=5.75*0.01;


    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *detect_result;
        zarray_get(detections, i, &detect_result);
        line(frame, cv::Point(detect_result->p[0][0], detect_result->p[0][1]),
             cv::Point(detect_result->p[1][0], detect_result->p[1][1]),
             cv::Scalar(0, 0xff, 0), 2);
        line(frame, cv::Point(detect_result->p[0][0], detect_result->p[0][1]),
             cv::Point(detect_result->p[3][0], detect_result->p[3][1]),
             cv::Scalar(0, 0, 0xff), 2);
        line(frame, cv::Point(detect_result->p[1][0], detect_result->p[1][1]),
             cv::Point(detect_result->p[2][0], detect_result->p[2][1]),
             cv::Scalar(0xff, 0, 0), 2);
        line(frame, cv::Point(detect_result->p[2][0], detect_result->p[2][1]),
             cv::Point(detect_result->p[3][0], detect_result->p[3][1]),
             cv::Scalar(0xff, 0, 0xff), 2);

        info.det=detect_result;
        apriltag_pose_t pose;
        double err = estimate_tag_pose(&info, &pose);
        std::cout<<"nrows:"<<pose.t->nrows<<"ncols:"<<pose.t->ncols<<std::endl;
        std::cout<<"data:"<<pose.t->data[0]<<" "<<pose.t->data[1]<<" "<<pose.t->data[2]<<std::endl;
    }



    cv::imshow("apri", frame);
    cv::waitKey(0);
    cout << "hello world!" << endl;
    return 0;
}
