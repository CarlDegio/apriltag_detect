
#include <iostream>
#include <winsock.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <apriltag/tagStandard41h12.h>
#include <apriltag/apriltag_pose.h>
#include <random>

using namespace std;
using namespace sl;

sl::InitParameters camera_setting()
{
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution= sl::RESOLUTION::HD720;
    init_parameters.camera_fps=60;
    init_parameters.depth_mode = sl::DEPTH_MODE::NONE; // no depth computation required here
    return init_parameters;
}

bool open_camera(sl::Camera &zed)
{
    sl::InitParameters init_parameters = camera_setting();
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout<<"Camera Open Fail"<<toString(returned_state);
        return false;
    }else return true;
}

void print_camera_parameter(sl::Camera &zed)
{
    auto camera_info = zed.getCameraInformation();
    cout << endl;
    cout <<"ZED Model                 : "<<camera_info.camera_model << endl;
    cout <<"ZED Serial Number         : "<< camera_info.serial_number << endl;
    cout <<"ZED Camera Firmware       : "<< camera_info.camera_configuration.firmware_version <<"/"<<camera_info.sensors_configuration.firmware_version<< endl;
    cout <<"ZED Camera Resolution     : "<< camera_info.camera_configuration.resolution.width << "x" << camera_info.camera_configuration.resolution.height << endl;
    cout <<"ZED Camera FPS            : "<< zed.getInitParameters().camera_fps << endl;
}

void create_cv_windows(const cv::String& windows_name)
{
    cv::namedWindow(windows_name, cv::WINDOW_NORMAL);
}

int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

bool get_sl_image(sl::Camera &zed,Mat &zed_image)
{
    auto returned_state = zed.grab();
    if (returned_state == ERROR_CODE::SUCCESS)
    {
        zed.retrieveImage(zed_image, VIEW::LEFT);
        return true;
    }
    else
    {
        cout<<"Error during capture : "<<toString(returned_state);
        return false;
    }
}

void sl_image_to_cv_image(Mat &zed_image,cv::Mat &cv_image)
{
    cv_image = cv::Mat((int) zed_image.getHeight(), (int) zed_image.getWidth(), getOCVtype(zed_image.getDataType())  , zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
}

class april_util{
    apriltag_detector_t* tag_detector;
    apriltag_family_t* tag_family;
    apriltag_detection_t *detect_result;
    apriltag_detection_info_t info;
    cv::Mat *image_rgb,image_gray;
    cv::Mat camera_matrix=cv::Mat::zeros(3,3,CV_64F);
    cv::Mat dist_coeff=cv::Mat::zeros(1,4,CV_64F);
    apriltag_pose_t pose_result;

    cv::Mat rotate_matrix,translate_matrix,rotate_vector;

    void pose_estimate_process()
    {
        double err = estimate_tag_pose(&info, &pose_result);//t是三行一列，R是3*3
        //cout<<pose_result.t->data[0]<<','<<pose_result.t->data[1]<<','<<pose_result.t->data[2]<<endl;
        rotate_matrix=cv::Mat(3,3,CV_64F,pose_result.R->data);
        translate_matrix=cv::Mat(3,1,CV_64F,pose_result.t->data);
        translate_matrix=rotate_matrix*cv::Vec3d(0,0,0.2)+translate_matrix;
        cv::Rodrigues(rotate_matrix,rotate_vector);
    }

    void draw_debug()
    {
        line(*image_rgb, cv::Point(detect_result->p[0][0], detect_result->p[0][1]),
             cv::Point(detect_result->p[1][0], detect_result->p[1][1]),
             cv::Scalar(0, 0xff, 0), 2);
        line(*image_rgb, cv::Point(detect_result->p[0][0], detect_result->p[0][1]),
             cv::Point(detect_result->p[3][0], detect_result->p[3][1]),
             cv::Scalar(0, 0, 0xff), 2);
        line(*image_rgb, cv::Point(detect_result->p[1][0], detect_result->p[1][1]),
             cv::Point(detect_result->p[2][0], detect_result->p[2][1]),
             cv::Scalar(0xff, 0, 0), 2);
        line(*image_rgb, cv::Point(detect_result->p[2][0], detect_result->p[2][1]),
             cv::Point(detect_result->p[3][0], detect_result->p[3][1]),
             cv::Scalar(0xff, 0, 0xff), 2);

        std::vector<cv::Point3d> axis;
        axis.emplace_back(cv::Point3d(0,0,0));
        axis.emplace_back(cv::Point3d(0.1,0,0));
        axis.emplace_back(cv::Point3d(0,0.1,0));
        axis.emplace_back(cv::Point3d(0,0,0.1));

        cv::Mat axis_end_point,jacob;
        cv::projectPoints(axis,rotate_vector,translate_matrix,camera_matrix,dist_coeff,axis_end_point,jacob);

        cv::arrowedLine(*image_rgb, axis_end_point.at<cv::Point2d>(0),
             axis_end_point.at<cv::Point2d>(1),
             cv::Scalar(0, 0, 0xff), 2);

        cv::arrowedLine(*image_rgb, axis_end_point.at<cv::Point2d>(0),
             axis_end_point.at<cv::Point2d>(2),
             cv::Scalar(0, 0xff, 0), 2);

        cv::arrowedLine(*image_rgb, axis_end_point.at<cv::Point2d>(0),
             axis_end_point.at<cv::Point2d>(3),
             cv::Scalar(0xff, 0, 0), 2);
    }

    void camera_parameter()
    {
        info.fx=528.885;
        info.fy=528.5100;
        info.cx=636.0100;
        info.cy=351.1240;
        info.tagsize=5.75*0.01;

        camera_matrix.at<double>(0,0)=info.fx;
        camera_matrix.at<double>(0,2)=info.cx;
        camera_matrix.at<double>(1,1)=info.fy;
        camera_matrix.at<double>(1,2)=info.cy;
        camera_matrix.at<double>(2,2)=1.0;
    }

    void create_41h12_detector()
    {
        tag_family = tagStandard41h12_create();
        tag_detector = apriltag_detector_create();
        apriltag_detector_add_family(tag_detector, tag_family);
    }

public:
    april_util(cv::Mat *image)
    {
        create_41h12_detector();
        camera_parameter();
        image_rgb=image;
    }

    void detect_process(bool debug=false)
    {
        cvtColor(*image_rgb, image_gray, cv::COLOR_BGR2GRAY);
        image_u8_t im = {image_gray.cols,image_gray.rows,image_gray.cols,image_gray.data};
        zarray_t *detections_raw = apriltag_detector_detect(tag_detector, &im);
        for (int i = 0; i < zarray_size(detections_raw); i++) {
            zarray_get(detections_raw, i, &detect_result);//detect result一直在改变
            info.det=detect_result;
            pose_estimate_process();
            if(debug)draw_debug();
        }
    }
};


int main(int argc, char **argv)
{
    sl::Camera zed;
    string windows_name="zed april detect";

    if(!open_camera(zed))return EXIT_FAILURE;
    print_camera_parameter(zed);
    create_cv_windows(windows_name);

    Mat zed_image;
    cv::Mat cv_image;
    april_util april_manager=april_util(&cv_image);

    default_random_engine random_engine;
    uniform_int_distribution<int> uniform_distribution(0, 100);
    char key = ' ';
    while(key!='q')
    {
        if(!get_sl_image(zed,zed_image))break;
        sl_image_to_cv_image(zed_image,cv_image);
        april_manager.detect_process(true);

        //if(uniform_distribution(random_engine)==50)cout<<err<<endl;

        cv::imshow(windows_name, cv_image);
        key = cv::waitKey(10);
    }
    zed.close();
    return EXIT_SUCCESS;
}