#ifndef __FDA_IMAGE_OPERATION_HPP__
#define __FDA_IMAGE_OPERATION_HPP__
#include <string>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "inf_face_detection_ssd_api.hh"

std::vector<std::string> decode_image_opencvMap(const std::string dataset, std::vector<cv::Mat> &imgsMatList, bool Isrand, unsigned int cnt = 0);
int get_images_path(std::string dataset, std::vector<std::string> &imgsPathList);
int video_capture(const std::string &videoName);
void plot_face_rect(const std::string savePath, std::string imgName, const cv::Mat & img, const facesPerImg_t &faces);

#endif