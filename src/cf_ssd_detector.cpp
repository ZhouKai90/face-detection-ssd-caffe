#include "cf_ssd_config.hh"
#include "cf_ssd_detector.hh"

SingleDetector::SingleDetector(const std::string & prototxtFile,
                    const std::string & caffemodelFile)
{
    Caffe::set_mode(Caffe::GPU);
    
    net_.reset(new Net<float>(prototxtFile, TEST));
    net_->CopyTrainedLayersFrom(caffemodelFile);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* inputLayer = net_->input_blobs()[0];
    numChannles_ = inputLayer->channels();
    CHECK(numChannles_ == 3 || numChannles_ == 1) << "Input layer should have 1 or 3 channels.";
    inputGeometry_ = cv::Size(inputLayer->width(), inputLayer->height());
    printf("channles:%d, width:%d, height:%d\n", numChannles_, inputLayer->width(), inputLayer->height());
    Blob<float>* outputLayer = net_->output_blobs()[0];
    printf("Out put layers channels:%d\n", outputLayer->channels());

}

std::vector<facesPerImg_t> SingleDetector::run(const std::vector<cv::Mat>& imgs)
{
    std::vector<facesPerImg_t> result;
    for (auto img : imgs) {
        facesPerImg_t tmp;
        memset(&tmp, 0 ,sizeof(facesPerImg_t));
        std::vector<vector<float>> output = Detect(img);
        if (output.size() == 0) {
            printf("No face detected.\n");
            tmp.hasFace = false;
            result.push_back(tmp);
            continue;
        } 
        tmp.hasFace = true;
        for (auto bbox: output) {
            bbox_t box;
            memset(&box, 0 ,sizeof(bbox_t));
            tmp.faceCnt += 1;
            assert(bbox.size() == 7);
            box.score = bbox[2];
            box.x1 = bbox[3];
            box.y1 = bbox[4];
            box.x2 = bbox[5];
            box.y2 = bbox[6];
            tmp.faces.push_back(box);
        }
        assert(tmp.faceCnt == tmp.faces.size());
        result.push_back(tmp);
    }
    return result;
}

std::vector<vector<float>> SingleDetector::Detect(const cv::Mat& img) 
{
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    if (width > height) {       // 以最长边为基准，等比例缩放图像
        resizeFactor_ = static_cast<double>(width) / static_cast<double>(MAX_IMG);
        inputGeometry_.height = static_cast<int>(MAX_IMG / static_cast<float>(width) * height);
        inputGeometry_.width = MAX_IMG;
    } else {
        resizeFactor_ = static_cast<double>(height) / static_cast<double>(MAX_IMG);
        inputGeometry_.width = static_cast<int>(MAX_IMG / static_cast<float>(height) * width);
        inputGeometry_.height = MAX_IMG;
    }

    Blob<float>* inputLayer = net_->input_blobs()[0];
    inputLayer->Reshape(1, numChannles_, inputGeometry_.height, inputGeometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> inputChannels;
    wrap_input_layer(&inputChannels);
    preprocess(img, &inputChannels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* resultBlob = net_->output_blobs()[0];
    const float* result = resultBlob->cpu_data();
    const int num_det = resultBlob->height();
    std::vector<std::vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1 || result[2] < FACE_SCORE) {
        // Skip invalid detection.
        result += 7;
        continue;
        }
        std::vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

void SingleDetector::wrap_input_layer(std::vector<cv::Mat>* inputChannels)
{
    Blob<float>* inputLayer = net_->input_blobs()[0];

    int width = inputLayer->width();
    int height = inputLayer->height();
    float* inputData = inputLayer->mutable_cpu_data();
    for (int i = 0; i < inputLayer->channels(); i++) {
        cv::Mat channel(height, width, CV_32FC1, inputData);
        inputChannels->push_back(channel);
        inputData += width * height;
    }
}

void SingleDetector::preprocess(const cv::Mat& img, std::vector<cv::Mat>* inputChannels)
{
    cv::Mat sample;
    if (img.channels() == 3 || img.channels() == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && numChannles_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && numChannles_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && numChannles_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sampleResized;
    if (sample.size() != inputGeometry_)
        cv::resize(sample, sampleResized, inputGeometry_);
    else
        sampleResized = sample;
    
    cv::Mat sampleFloat;
    if (numChannles_ == 3)
        sampleResized.convertTo(sampleFloat, CV_32FC3);
    else
        sampleResized.convertTo(sampleFloat, CV_32FC1);
    
    cv::split(sampleFloat, *inputChannels);
    CHECK(reinterpret_cast<float*>(inputChannels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}