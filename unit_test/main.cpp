#include "cf_ssd_util.hh"
#include "inf_face_detection_ssd_api.hh"

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Input test images path" << std::endl;
        return -1;
    }

    const std::string imgPath = argv[1];

    std::vector<std::string>imgsNameList;
    get_images_path(imgPath, imgsNameList);

    std::string modelPath = "/kyle/workspace/project/caffe-ssd/models";
    FDHandle handle = inf_face_detection_ssd_init(modelPath);
    std::vector<cv::Mat> matList;
    for (auto img : imgsNameList) {
        cv::Mat tmp = cv::imread(img);
        matList.push_back(tmp);
    }

    std::vector<facesPerImg_t> resaults;
    inf_face_detecte(handle, matList, resaults);

    const std::string savePath = "/kyle/workspace/project/caffe-ssd/output";
    assert(imgsNameList.size() == resaults.size());
    for (int i = 0; i < resaults.size(); i++) {
        std::cout << ">>>>>>>>>>>>>>" << imgsNameList[i] << "<<<<<<<<<<<<<<<" << std::endl;
        printf("hasFace: %d, faceCnt: %d\n", resaults[i].hasFace, resaults[i].faceCnt);
        for (auto k : resaults[i].faces) {
            printf("score: %f, bbox[%f, %f, %f, %f]\n", k.score, k.x1, k.y1, k.x2, k.y2);
        }
        plot_face_rect(savePath, imgsNameList[i], matList[i], resaults[i]);
    }
    printf("test finish\n");
    inf_face_detection_ssd_uint(handle);
}