#ifndef __INF_FACIAL_EXPRESSION_RECOGNITION_API_HH__
#define __INF_FACIAL_EXPRESSION_RECOGNITION_API_HH__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

typedef void *FDHandle;
typedef int infResult;

#define INF_SUCCESS     0
#define INF_FAIL        -1

#define MAX_FACE_CNT_PER_IMG 16

typedef struct bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} bbox_t;

typedef struct facesPerImg
{
    bool hasFace;
    int faceCnt;
    std::vector<bbox_t> faces;
    // std::vector<bbox_t> faces[MAX_FACE_CNT_PER_IMG];
} facesPerImg_t;

typedef std::vector<facesPerImg> facesList;

FDHandle inf_face_detection_ssd_init(const std::string &modelPath);

infResult inf_face_detection_ssd_uint(FDHandle handle);

infResult inf_face_detecte(FDHandle handle, const std::vector<cv::Mat>& imgsMat, facesList &list);


#endif