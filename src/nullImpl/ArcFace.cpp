#include "ArcFace.h"
#include <numeric>

ArcFace::ArcFace(){}

ArcFace::~ArcFace(){
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

void ArcFace::initialize(const std::string& model_path) {
    if (!FileExists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = new Ort::Session(env_, model_path.c_str(), session_options);

    // --- Get Input and Output Node Details (CORRECTED) ---
    Ort::AllocatorWithDefaultOptions allocator;

    // --- Get Input Node Details ---
    input_node_names_.clear();
    // Use GetInputNameAllocated and copy the result into a std::string
    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_node_names_.push_back(input_name_ptr.get());

    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_dims_ = input_tensor_info.GetShape();

    input_height_ = static_cast<int>(input_dims_[2]);
    input_width_  = static_cast<int>(input_dims_[3]);

    // --- Get Output Node Details ---
    output_node_names_.clear();
    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        // Use GetOutputNameAllocated and copy the result into a std::string
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
        output_node_names_.push_back(output_name_ptr.get());
    }
}

cv::Mat ArcFace::preprocess(cv::Mat img, FacePts landmark) {
    
    // normalization process
    cv::Mat src(5, 2, CV_32FC1, reference_landmark_);

    float dest_landmark[5][2];

    for (int i = 0; i < 5; i++)
    {
        dest_landmark[i][0] = landmark.x[i];
        dest_landmark[i][1] = landmark.y[i];
    }

    cv::Mat dst(5, 2, CV_32FC1, dest_landmark);

    cv::Mat M = cv::estimateAffine2D(dst, src);

    // Handle the case where the estimation fails
    if (M.empty()) {
        // You might want to return an empty Mat or handle the error appropriately
        // For example, resize and return the original unaligned image crop
        cv::resize(img, img, cv::Size(input_width_, input_height_));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        return img;
    }

    cv::Mat img_aligned;
    cv::warpAffine(img, img_aligned, M, cv::Size(input_width_, input_height_));

    // resize into (112, 112)
    cv::cvtColor(img_aligned, img_aligned, cv::COLOR_BGR2RGB);
    cv::resize(img_aligned, img_aligned, cv::Size(input_width_, input_height_));

    return img_aligned;
}


std::vector<float> ArcFace::GetEmbedding(cv::Mat img, FacePts landmarks) {
    std::vector<float> embedding;
    if (img.empty()) {
        return embedding;
    }
    
    cv::Mat processed_img = preprocess(img, landmarks);
    
    size_t proImgSize = processed_img.total() * processed_img.elemSize();
    float* proImgData = (float*)processed_img.data;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> fixed_input_dims = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        proImgData, proImgSize,
        fixed_input_dims.data(), fixed_input_dims.size()
    );

    std::vector<const char*> input_names_char;
    input_names_char.reserve(input_node_names_.size());
    for (const auto& name: input_node_names_) {
        input_names_char.push_back(name.c_str());
    }

    std::vector<const char*> output_names_char;
    output_names_char.reserve(output_node_names_.size());
    for (const auto& name: output_node_names_) {
        output_names_char.push_back(name.c_str());
    }

    Ort::RunOptions runOption = Ort::RunOptions{nullptr};
    std::vector<Ort::Value> output_tensors = session_->Run(
        runOption,
        input_names_char.data(), &input_tensor, input_names_char.size(),
        output_names_char.data(), output_node_names_.size()
    );

    const float* a = output_tensors[0].GetTensorData<float>();

    std::cout<<"embedding: " << a[0] << std::endl;
    return embedding;
}