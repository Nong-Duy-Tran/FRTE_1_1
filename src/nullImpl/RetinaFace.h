#ifndef RETINAFACE_ONNX_H
#define RETINAFACE_ONNX_H

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};


class RetinaFace
{
public:
    // Constructor now takes an NMS threshold and a flag for GPU usage.
    RetinaFace(float nms = 0.4f, bool use_gpu = false);
    ~RetinaFace();

    // Initialization now loads a single .onnx model file.
    void initialize(const std::string& model_path);
    
    // The main detection function signature remains the same for API compatibility.
    std::vector<FaceDetectInfo> detect(cv::Mat img, float threshold = 0.5f, float scale = 1.0f);

private:
    // --- Pre-processing and Post-processing (Largely unchanged logic) ---
    // These methods perform calculations on the raw model output and are not tied to the inference engine.
    anchor_box bbox_pred(const anchor_box& prior, const cv::Vec4f& regress, const float variance[]);
    FacePts landmark_pred(const anchor_box& prior, const float* landm_regress, const float variance[]);
    static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);

    // Helper to check if a file exists.
    inline bool FileExists(const std::string &name)
    {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }
    
    // New pre-processing helper for ONNX Runtime
    void preprocess(cv::Mat& img, std::vector<float>& input_tensor_values);


private:
    // --- ONNX Runtime Members (Replaces MXNet members) ---
    Ort::Env env_;
    Ort::Session* session_ = nullptr; // Using a pointer to manage lifetime
    Ort::AllocatorWithDefaultOptions allocator_;

    // Model input and output details
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_dims_; // e.g., {1, 3, 480, 640}

    float nms_threshold_;
    bool use_gpu_;
    int input_width_;
    int input_height_;
    
    // Image normalization parameters
    float pixel_means[3] = {104.0f, 117.0f, 123.0f}; // Typical BGR means for RetinaFace
    float pixel_stds[3] = {1.0f, 1.0f, 1.0f};
    float pixel_scale_ = 1.0f;

    float variance[2] = {0.1f, 0.2f};

    // Anchor generation configuration (we will simplify this)
    std::vector<std::vector<int>> min_sizes_;
    std::vector<int> steps_;
    
    // This will hold the generated priors in [cx, cy, w, h] format
    std::vector<anchor_box> priors_; 
};

#endif // RETINAFACE_ONNX_H