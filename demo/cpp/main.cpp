#include "cf_ssd_util.hh"
#include "inf_face_detection_ssd_api.hh"
#include "cf_ssd_config.hh"

#include <chrono>
using milli = std::chrono::milliseconds;

#define TEST_CNT 1

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Input test images path" << std::endl;
        return -1;
    }

    const std::string imgPath = argv[1];

    std::vector<std::string>imgsNameList;
    get_images_path(imgPath, imgsNameList);

    FDHandle handle = inf_face_detection_ssd_init(MODEL_PATH);
    std::vector<cv::Mat> matList;
    for (auto img : imgsNameList) {
        cv::Mat tmp = cv::imread(img);
        matList.push_back(tmp);
    }

	auto start = std::chrono::high_resolution_clock::now();
    std::vector<facesPerImg_t> resaults;
    for (int i = 0; i < TEST_CNT; i++) {
        inf_face_detecte(handle, matList, resaults);
    }
	auto end = std::chrono::high_resolution_clock::now();

    std::cout << "face detect took " << std::chrono::duration_cast<milli>(end - start).count() << " ms\n";
    std::cout << "speed " << (double(imgsNameList.size()) / std::chrono::duration_cast<milli>(end - start).count()) * 1000 * TEST_CNT<< " fps\n";

    assert(imgsNameList.size() == resaults.size());
    for (int i = 0; i < resaults.size(); i++) {
        std::cout << ">>>>>>>>>>>>>>" << imgsNameList[i] << "<<<<<<<<<<<<<<<" << std::endl;
        printf("hasFace: %d, faceCnt: %d\n", resaults[i].hasFace, resaults[i].faceCnt);
        for (auto k : resaults[i].faces) {
            printf("score: %f, bbox[%f, %f, %f, %f]\n", k.score, k.x1, k.y1, k.x2, k.y2);
        }
        plot_face_rect(OUTPUT_PATH, imgsNameList[i], matList[i], resaults[i]);
    }
    printf("test finish\n");
    inf_face_detection_ssd_uint(handle);
}