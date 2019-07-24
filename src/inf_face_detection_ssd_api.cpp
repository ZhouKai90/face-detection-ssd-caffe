#include "inf_face_detection_ssd_api.hh"
#include "cf_ssd_detector.hh"
#include "cf_ssd_config.hh"

FDHandle inf_face_detection_ssd_init(const std::string &modelPath)
{
    const std::string prototxt = modelPath + PROTOTXT;
    const std::string caffemodel = modelPath + CAFFEMODEL;

    SingleDetector *handle = new SingleDetector(prototxt, caffemodel);
    std::cout << "facial expression recongnize handle init succeed." << std::endl;
    return (FDHandle) handle;
}

infResult inf_face_detection_ssd_uint(FDHandle handle)
{
    if (handle != NULL) {
        SingleDetector* handle = (SingleDetector*) handle;
        delete handle;
    }

    return INF_SUCCESS;
}


infResult inf_face_detecte(FDHandle handle, const std::vector<cv::Mat>& imgsMat, \
                                         std::vector<facesPerImg_t> &list)
{
    if (handle == NULL) {
        std::cout << "handle is NULL" << std::endl;
        return INF_FAIL;
    }
    SingleDetector* DetectorHandle = (SingleDetector*) handle;
    list = DetectorHandle->run(imgsMat);
    return INF_SUCCESS;
}
