//
// Created by Eternity on 2022/10/6.
//

#include <iostream>
#include "opencv2/opencv.hpp"

#define debug_cv false

cv::Mat chassis,chassis_gray,chassis_undistort;
std::vector<std::vector<cv::Point3f> > objpoints;
std::vector<std::vector<cv::Point2f> > imgpoints;
cv::Size patternsize(6,9);

void get_world_axis(std::vector<cv::Point3f>& world_point)
{
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 6; ++j) {
            world_point.emplace_back(j*1.7*0.01,i*1.7*0.01,0);
        }
    }
}

void get_pic_axis(std::vector<cv::Point2f>& corners)
{
    cv::cvtColor(chassis,chassis_gray,cv::COLOR_RGB2GRAY);
    bool patternfound=cv::findChessboardCorners(chassis_gray,patternsize,corners,
                                                cv::CALIB_CB_ADAPTIVE_THRESH+cv::CALIB_CB_NORMALIZE_IMAGE);

    cv::TermCriteria criteria( cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
    cv::cornerSubPix(chassis_gray,corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
    if(debug_cv)
    {
        cv::drawChessboardCorners(chassis,patternsize,cv::Mat(corners),patternfound);
        cv::imshow("chassis", chassis);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void pic_world_every(const std::string& name)
{
    chassis=cv::imread(name);
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point3f> world_points;

    get_world_axis(world_points);
    get_pic_axis(corners);

    objpoints.emplace_back(world_points);
    imgpoints.emplace_back(corners);
}

int main(int argc, char** argv)
{

    cv::Mat cameraMatrix,dist,R,T;
    std::stringstream fmt;

    for (int i = 1; i <= 7; ++i) {
        fmt<<i<<".jpg";
        pic_world_every(fmt.str());
        fmt.str("");
    }

    cv::calibrateCamera(objpoints, imgpoints,cv::Size(chassis_gray.rows,chassis_gray.cols), cameraMatrix, dist, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << dist << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;

    cv::undistort(chassis,chassis_undistort,cameraMatrix,dist);
    cv::imshow("undistort", chassis_undistort);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;

}