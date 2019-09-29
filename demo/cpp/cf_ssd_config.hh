#ifndef __CF_SSD_DEFINE_HH__
#define __CF_SSD_DEFINE_HH__

#include <string>

#define FACE_SCORE 0.9

#define MIN_IMG 512

#define GPU_ID 0
const std::string ROOT_PATH = "/kyle/workspace/project/face-detection-ssd-caffe";
const std::string MODEL_PATH = ROOT_PATH + "/models/deploy";
const std::string OUTPUT_PATH = ROOT_PATH + "/output";
const std::string PROTOTXT = "/half_VGG16_SSD_512.prototxt";
const std::string CAFFEMODEL = "/half_VGG16_SSD_512.caffemodel";

#endif