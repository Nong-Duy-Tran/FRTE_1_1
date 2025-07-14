#ifndef RETINAFACE_ONNX_H
#define RETINAFACE_ONNX_H

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

struct anchor_win {
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts {
    float x[5];
    float y[5];
};

struct FaceDetectInfo {
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg {
    int STRIDE = 0;
    vector<int> SCALES;
    int BASE_SIZE = 0;
    vector<float> RATIOS;
    int ALLOWED_BORDER = 0;
};

class RetinaFaceONNX {
public:
    RetinaFaceONNX(const std::string& model_path, float nms_thresh = 0.4);
    ~RetinaFaceONNX();

    std::vector<FaceDetectInfo> detect(const cv::Mat& img, float threshold = 0.5, float scale = 1.0);

private:
    void preprocess(const cv::Mat& img, std::vector<float>& input_tensor_values);
    std::vector<FaceDetectInfo> postprocess(const std::vector<float>& scores,
                                            const std::vector<float>& boxes,
                                            const std::vector<float>& landmarks,
                                            float threshold,
                                            int img_width,
                                            int img_height,
                                            float scale);

    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo>& detections, float threshold);
    anchor_box bbox_pred(const anchor_box& anchor, const cv::Vec4f& regress);
    FacePts landmark_pred(const anchor_box& anchor, const FacePts& facePt);
    static bool CompareBBox(const FaceDetectInfo& a, const FaceDetectInfo& b);

    bool FileExists(const std::string& name) {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    float pixel_means[3] = {0.0f, 0.0f, 0.0f};
    float pixel_stds[3] = {1.0f, 1.0f, 1.0f};
    float pixel_scale = 1.0f;

    float nms_threshold;
    std::string model_path_;

    // anchor-related structures
    std::vector<float> _ratio;
    std::vector<anchor_cfg> cfg;
    std::vector<int> _feat_stride_fpn;
    std::map<std::string, std::vector<anchor_box>> _anchors_fpn;
    std::map<std::string, std::vector<anchor_box>> _anchors;
    std::map<std::string, int> _num_anchors;
};

#endif // RETINAFACE_ONNX_H
