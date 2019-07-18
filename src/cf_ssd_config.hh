#ifndef __CF_SSD_DEFINE_HH__
#define __CF_SSD_DEFINE_HH__

#include <string>

#define IMG_WIDTH 1920 * 0.8
#define IMG_HEIGHT 1080 * 0.8

const std::string ROOT_PATH = "/kyle/workspace/project/caffe_ssd";
const std::string PROTOTXT =  ROOT_PATH + "/models/ssd_vgg16_512.prototxt";
const std::string CAFFEMODEL = ROOT_PATH + "/models/ssd_vgg16_512.caffemodel";

#endif