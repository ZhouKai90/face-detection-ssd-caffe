#ifndef __CF_SSD_DEFINE_HH__
#define __CF_SSD_DEFINE_HH__

#include <string>

#define FACE_SCORE 0.8

#define MAX_IMG 800

const std::string ROOT_PATH = "/kyle/workspace/project/face-detection-ssd-caffe";
const std::string MODEL_PATH = ROOT_PATH + "/models/deploy";
const std::string OUTPUT_PATH = ROOT_PATH + "/output";
const std::string PROTOTXT = "/ssd_vgg16_512.prototxt";
const std::string CAFFEMODEL = "/ssd_vgg16_512.caffemodel";

#endif