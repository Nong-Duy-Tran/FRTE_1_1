#ifndef ARCFACE_ONNX_H
#define ARCFACE_ONNX_H

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "RetinaFace.h"

class ArcFace
{
    public:
        ArcFace();
        ~ArcFace();

        // Initialization now loads a single .onnx model file.
        void initialize(const std::string& model_path);
        
        // The main detection function signature remains the same for API compatibility.
        std::vector<float> GetEmbedding(cv::Mat img, FacePts landmarks);

    private:
        inline bool FileExists(const std::string &name)
        {
            std::ifstream fhandle(name.c_str());
            return fhandle.good();
        }

        std::vector<float> preprocess(cv::Mat img, FacePts landmark);

    private:
        // --- ONNX Runtime Members (Replaces MXNet members) ---
        Ort::Env env_;
        Ort::Session* session_ = nullptr; // Using a pointer to manage lifetime
        Ort::AllocatorWithDefaultOptions allocator_;

        // Model input and output details
        std::vector<std::string> input_node_names_;
        std::vector<std::string> output_node_names_;
        std::vector<int64_t> input_dims_; // e.g., {1, 3, 480, 640}
        int input_width_;
        int input_height_;
        float reference_landmark_[5][2] = {
            {30.2946f + 8.0, 51.6963f},
            {65.5318f + 8.0, 51.5014f},
            {48.0252f + 8.0, 71.7366f},
            {33.5493f + 8.0, 92.3655f},
            {62.7299f + 8.0, 92.2041f}
        };

        
};


#endif