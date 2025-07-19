#include "RetinaFace.h"
#include <numeric> // For std::accumulate

void clip_boxes(anchor_box &box, int width, int height) {
    box.x1 = std::max(0.0f, std::min(box.x1, (float)width - 1));
    box.y1 = std::max(0.0f, std::min(box.y1, (float)height - 1));
    box.x2 = std::max(0.0f, std::min(box.x2, (float)width - 1));
    box.y2 = std::max(0.0f, std::min(box.y2, (float)height - 1));
}

// --- End of Framework-Independent Code ---


// ######################################################################
// RetinaFace Class Implementation (ONNX Runtime Version)
// ######################################################################

RetinaFace::RetinaFace(float nms, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RetinaFace-ONNX"),
      nms_threshold_(nms), use_gpu_(use_gpu) {

    // --- Anchor Configuration (matches cfg_mnet from Python) ---
    steps_ = {8, 16, 32};
    min_sizes_ = {{16, 32}, {64, 128}, {256, 512}};
    // Note: The priors will be generated in initialize() once we know the input size.
}

RetinaFace::~RetinaFace() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

void RetinaFace::initialize(const std::string& model_path) {
    if (!FileExists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_gpu_) {
        // OrtCUDAProviderOptions cuda_options{};
        // session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

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

    priors_.clear();
    std::vector<std::pair<int, int>> feature_maps;
    for (const auto& step : steps_) {
        feature_maps.push_back({
            (int)ceilf((float)input_height_ / step),
            (int)ceilf((float)input_width_ / step)
        });
    }

    for (size_t k = 0; k < feature_maps.size(); ++k) {
        auto f_map = feature_maps[k];
        auto m_sizes = min_sizes_[k];
        for (int i = 0; i < f_map.first; ++i) {
            for (int j = 0; j < f_map.second; ++j) {
                for (const auto& min_size : m_sizes) {
                    float s_kx = (float)min_size / input_width_;
                    float s_ky = (float)min_size / input_height_;
                    float cx = ((float)j + 0.5f) * steps_[k] / input_width_;
                    float cy = ((float)i + 0.5f) * steps_[k] / input_height_;
                    priors_.push_back({cx, cy, s_kx, s_ky});
                }
            }
        }
    }

}

std::vector<FaceDetectInfo> RetinaFace::detect(cv::Mat img, float threshold, float scale) {
    std::vector<FaceDetectInfo> proposals;
    if (img.empty()) {
        return proposals;
    }

    float im_h = img.rows;
    float im_w = img.cols;
    
    // --- 1. Pre-processing ---
    std::vector<float> input_tensor_values;
    preprocess(img, input_tensor_values);

    std::vector<int64_t> actual_input_dim = {1, 3, (int64_t)this->input_height_, (int64_t)this->input_width_};

    // --- 2. Create Input Tensor ---
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        actual_input_dim.data(), actual_input_dim.size()
    );

    // --- 3. Run Inference ---
    std::vector<const char*> input_names_char;
    input_names_char.reserve(input_node_names_.size());
    for (const auto& name : input_node_names_) {
        input_names_char.push_back(name.c_str());
    }

    std::vector<const char*> output_names_char;
    output_names_char.reserve(output_node_names_.size());
    for (const auto& name : output_node_names_) {
        output_names_char.push_back(name.c_str());
    }

    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_names_char.data(), &input_tensor, input_names_char.size(), 
        output_names_char.data(), output_names_char.size()
    );

    // --- 4. Post-processing ---

    const float* loc_ptr = output_tensors[0].GetTensorData<float>();
    const float* conf_ptr = output_tensors[1].GetTensorData<float>();
    const float* landmarks_ptr = output_tensors[2].GetTensorData<float>();

    int loc_ptr_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "number boxes element:" << loc_ptr_size << std::endl;
    for (int i = 0; i < loc_ptr_size; i++) {
        if (loc_ptr[i] < 0) std::cout << "Nope 1" << std::endl; break;
    }

    int conf_ptr_size = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "number scores element:" << conf_ptr_size << std::endl;
    for (int i = 0; i < conf_ptr_size; i++) {
        if (conf_ptr[i] < 0) std::cout << "Nope 2" << std::endl; break;
    }

    int landmarks_ptr_size = output_tensors[2].GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "number landmarks element:" << landmarks_ptr_size << std::endl;
    for (int i = 0; i < landmarks_ptr_size; i++) {
        if (landmarks_ptr[i] < 0) std::cout << "Nope 3" << std::endl; break;
    }

    // ====================================================================
    // CLARIFICATION POINT 2: Correct Scaling Factors
    // ====================================================================
    // The Python script defines `scale` and `scale1` arrays to convert the
    // normalized coordinates back to the pixel coordinates of the *resized*
    // input image (e.g., 640x640). We will do the same.
    // Note: The Python script seems to do two scalings. One to the resized
    // image size and then another to the original. We can do this in one step.
    float scale_boxes[4] = {im_w, im_h, im_w, im_h};
    float scale_landms[10];
    for (int i = 0; i < 5; ++i) {
        scale_landms[i*2 + 0] = im_w;
        scale_landms[i*2 + 1] = im_h;
    }
    
    // Iterate through all the priors (anchors).
    for (size_t i = 0; i < priors_.size(); ++i) {
        // Get the confidence score for the "face" class (index 1).
        float score = conf_ptr[i * 2 + 1];

        if (score < threshold) {
            continue;
        }

        // Decode the bounding box and landmarks. These are normalized (0-1 range).
        cv::Vec4f regress(loc_ptr[i*4 + 0], loc_ptr[i*4 + 1], loc_ptr[i*4 + 2], loc_ptr[i*4 + 3]);
        anchor_box rect_normalized = bbox_pred(priors_[i], regress, variance);
        
        const float* landm_regress = &landmarks_ptr[i * 10];
        FacePts landmarks_normalized = landmark_pred(priors_[i], landm_regress, variance);

        // ====================================================================
        // CLARIFICATION POINT 3: Scaling to Original Image Size
        // ====================================================================
        // Now we convert the normalized coordinates directly to the original
        // image's pixel coordinates by multiplying with our scale factors.
        // This avoids intermediate scaling and is more efficient.
        FaceDetectInfo info;
        info.score = score;
        info.rect.x1 = rect_normalized.x1 * scale_boxes[0];
        info.rect.y1 = rect_normalized.y1 * scale_boxes[1];
        info.rect.x2 = rect_normalized.x2 * scale_boxes[2];
        info.rect.y2 = rect_normalized.y2 * scale_boxes[3];

        for (int j = 0; j < 5; ++j) {
            info.pts.x[j] = landmarks_normalized.x[j] * scale_landms[j*2];
            info.pts.y[j] = landmarks_normalized.y[j] * scale_landms[j*2+1];
        }
        
        // Before NMS, all coordinates are now in the original image's space.
        proposals.push_back(info);
    }
    

    // --- 5. NMS ---
    std::vector<FaceDetectInfo> final_faces = nms(proposals, nms_threshold_);

    return final_faces;
}

void RetinaFace::preprocess(cv::Mat& img, std::vector<float>& input_tensor_values) {
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb_img.convertTo(float_img, CV_32FC3); 
    
    cv::Mat resized_img;
    cv::resize(float_img, resized_img, cv::Size(this->input_width_, this->input_height_));

    resized_img -= cv::Scalar(104.0f, 117.0f, 123.0f);
    input_tensor_values.resize(1 * 3 * this->input_height_ * this->input_width_);
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < this->input_height_; ++h) {
            for (int w = 0; w < this->input_width_; ++w) {
                int out_idx = c * (this->input_height_ * this->input_width_) + h * this->input_width_ + w;
                input_tensor_values[out_idx] = resized_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

// --- Framework-Independent Helper Methods (No changes needed) ---

anchor_box RetinaFace::bbox_pred(const anchor_box& prior, const cv::Vec4f& regress, const float variance[]) {
    // prior is [cx, cy, w, h]
    // regress is [dx, dy, dw, dh]
    float cx = prior.x1 + regress[0] * variance[0] * prior.x2;
    float cy = prior.y1 + regress[1] * variance[0] * prior.y2;
    float w = prior.x2 * exp(regress[2] * variance[1]);
    float h = prior.y2 * exp(regress[3] * variance[1]);
    
    // convert [cx, cy, w, h] to [x1, y1, x2, y2]
    anchor_box box;
    box.x1 = (cx - w / 2.0f);
    box.y1 = (cy - h / 2.0f);
    box.x2 = (cx + w / 2.0f);
    box.y2 = (cy + h / 2.0f);
    return box;
}

FacePts RetinaFace::landmark_pred(const anchor_box& prior, const float* landm_regress, const float variance[]) {
    FacePts pts;
    for (int i = 0; i < 5; ++i) {
        pts.x[i] = prior.x1 + landm_regress[i*2 + 0] * variance[0] * prior.x2;
        pts.y[i] = prior.y1 + landm_regress[i*2 + 1] * variance[0] * prior.y2;
    }
    return pts;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b) {
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo> &bboxes, float threshold) {
    if (bboxes.empty()) {
        return {};
    }
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    std::vector<bool> is_suppressed(bboxes.size(), false);
    for (size_t i = 0; i < bboxes.size(); ++i) {
        if (is_suppressed[i]) {
            continue;
        }
        bboxes_nms.push_back(bboxes[i]);
        anchor_box select_bbox = bboxes[i].rect;
        float area1 = (select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1);

        for (size_t j = i + 1; j < bboxes.size(); ++j) {
            if (is_suppressed[j]) {
                continue;
            }
            anchor_box& bbox_j = bboxes[j].rect;
            float x = std::max(select_bbox.x1, bbox_j.x1);
            float y = std::max(select_bbox.y1, bbox_j.y1);
            float w = std::min(select_bbox.x2, bbox_j.x2) - x + 1;
            float h = std::min(select_bbox.y2, bbox_j.y2) - y + 1;
            if (w <= 0 || h <= 0) {
                continue;
            }

            float area2 = (bbox_j.x2 - bbox_j.x1 + 1) * (bbox_j.y2 - bbox_j.y1 + 1);
            float area_intersect = w * h;
            float iou = area_intersect / (area1 + area2 - area_intersect);
            
            if (iou > threshold) {
                is_suppressed[j] = true;
            }
        }
    }
    return bboxes_nms;
}