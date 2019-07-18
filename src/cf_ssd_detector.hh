#ifndef __CF_SSD_DETECTOR_HH__
#define __CF_SSD_DETECTOR_HH__

#include <string>
#include <vector>

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "inf_face_detection_ssd_api.hh"
using namespace caffe;

class Detector {
public:
    Detector(const std::string& prototxtFile,
            const std::string& caffemodelFile);
    ~Detector(){printf("~Detector\n");};
    std::vector<facesPerImg_t> run(const std::vector<cv::Mat>& imgs);

private:
    std::vector<vector<float>> Detect(const cv::Mat & img);
    void wrap_input_layer(std::vector<cv::Mat> *inputChannels);
    void preprocess(const cv::Mat & imgMat, std::vector<cv::Mat> *inputChannels);

private:
    shared_ptr<Net<float>> net_;
    cv::Size inputGeometry_;
    int numChannles_;
};

#endif