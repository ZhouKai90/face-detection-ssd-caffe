#ifndef __CF_SSD_DEFINE_HH__
#define __CF_SSD_DEFINE_HH__

#include <string>

#define FACE_SCORE 0.5

#define MAX_IMG 512

const std::string ROOT_PATH = "/kyle/workspace/project/face-detection-ssd-caffe";
const std::string MODEL_PATH = ROOT_PATH + "/models/deploy";
const std::string OUTPUT_PATH = ROOT_PATH + "/output";
const std::string PROTOTXT = "/VGG16_SSD_512.prototxt";
const std::string CAFFEMODEL = "/VGG16_SSD_512.caffemodel";

#endif