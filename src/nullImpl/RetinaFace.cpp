#include "RetinaFace.h"
#include <numeric> // For std::accumulate

// ######################################################################
// This entire section of code is FRAMEWORK-INDEPENDENT.
// It deals with the core logic of the RetinaFace algorithm, such as
// anchor generation, bounding box transformations, and NMS.
// It does not need to be changed when migrating from MXNet to ONNX Runtime.
// ######################################################################

// --- Start of Framework-Independent Code ---

anchor_win _whctrs(anchor_box anchor) {
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5f * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5f * (win.h - 1);
    return win;
}

anchor_box _mkanchors(anchor_win win) {
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5f * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5f * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5f * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5f * (win.h - 1);
    return anchor;
}

std::vector<anchor_box> _ratio_enum(anchor_box anchor, std::vector<float> ratios) {
    std::vector<anchor_box> anchors;
    for (float ratio : ratios) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratio;
        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratio);
        anchors.push_back(_mkanchors(win));
    }
    return anchors;
}

std::vector<anchor_box> _scale_enum(anchor_box anchor, std::vector<int> scales) {
    std::vector<anchor_box> anchors;
    for (int scale : scales) {
        anchor_win win = _whctrs(anchor);
        win.w = win.w * scale;
        win.h = win.h * scale;
        anchors.push_back(_mkanchors(win));
    }
    return anchors;
}

std::vector<anchor_box> generate_anchors(int base_size, const std::vector<float>& ratios, const std::vector<int>& scales) {
    anchor_box base_anchor = {0.0f, 0.0f, (float)base_size - 1, (float)base_size - 1};
    std::vector<anchor_box> ratio_anchors = _ratio_enum(base_anchor, ratios);
    std::vector<anchor_box> anchors;
    for (const auto& r_anchor : ratio_anchors) {
        std::vector<anchor_box> s_anchors = _scale_enum(r_anchor, scales);
        anchors.insert(anchors.end(), s_anchors.begin(), s_anchors.end());
    }
    return anchors;
}

std::vector<std::vector<anchor_box>> generate_anchors_fpn(const std::vector<anchor_cfg>& cfg) {
    std::vector<std::vector<anchor_box>> anchors;
    for (const auto& c : cfg) {
        anchors.push_back(generate_anchors(c.BASE_SIZE, c.RATIOS, c.SCALES));
    }
    return anchors;
}

std::vector<anchor_box> anchors_plane(int height, int width, int stride, const std::vector<anchor_box>& base_anchors) {
    std::vector<anchor_box> all_anchors;
    all_anchors.reserve(base_anchors.size() * height * width);
    for (const auto& base_anchor : base_anchors) {
        for (int ih = 0; ih < height; ++ih) {
            int sh = ih * stride;
            for (int iw = 0; iw < width; ++iw) {
                int sw = iw * stride;
                anchor_box tmp = {
                    base_anchor.x1 + sw,
                    base_anchor.y1 + sh,
                    base_anchor.x2 + sw,
                    base_anchor.y2 + sh
                };
                all_anchors.push_back(tmp);
            }
        }
    }
    return all_anchors;
}

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
    
    // --- Anchor Configuration ---
    // This logic is moved from the old `initialize` method to the constructor.
    _feat_stride_fpn = {32, 16, 8};
    _ratio = {1.0f, 1.5f}; // Example ratio, adjust if needed for your model

    anchor_cfg cfg32;
    cfg32.SCALES = {32, 16};
    cfg32.BASE_SIZE = 16;
    cfg32.RATIOS = _ratio;
    cfg32.ALLOWED_BORDER = 9999;
    cfg32.STRIDE = 32;
    cfg.push_back(cfg32);

    anchor_cfg cfg16;
    cfg16.SCALES = {8, 4};
    cfg16.BASE_SIZE = 16;
    cfg16.RATIOS = _ratio;
    cfg16.ALLOWED_BORDER = 9999;
    cfg16.STRIDE = 16;
    cfg.push_back(cfg16);
    
    anchor_cfg cfg8;
    cfg8.SCALES = {2, 1};
    cfg8.BASE_SIZE = 16;
    cfg8.RATIOS = _ratio;
    cfg8.ALLOWED_BORDER = 9999;
    cfg8.STRIDE = 8;
    cfg.push_back(cfg8);

    // --- Pre-calculate Anchors ---
    std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(cfg);
    for (size_t i = 0; i < anchors_fpn.size(); ++i) {
        std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = static_cast<int>(anchors_fpn[i].size());
    }
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
    
    // --- 2. Create Input Tensor ---
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_dims_.data(), input_dims_.size()
    );

    // --- 3. Run Inference ---
    // std::vector<Ort::Value> output_tensors = session_->Run(
    //     Ort::RunOptions{nullptr}, 
    //     input_node_names_.data(), &input_tensor, 1, 
    //     output_node_names_.data(), output_node_names_.size()
    // );

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
        input_names_char.data(), &input_tensor, 1, 
        output_names_char.data(), output_names_char.size()
    );

    

    // --- 4. Post-processing ---
    // Assuming 3 outputs: [boxes, scores, landmarks] OR [scores, boxes, landmarks]
    // The order depends on your ONNX model. Adjust indices [0], [1], [2] if needed.
    // Let's assume the order is: scores, boxes, landmarks.
    const float* scores_ptr = output_tensors[0].GetTensorData<float>();
    const float* boxes_ptr = output_tensors[1].GetTensorData<float>();
    const float* landmarks_ptr = output_tensors[2].GetTensorData<float>();

    size_t total_anchors = 0;
    for (int stride : _feat_stride_fpn) {
        int feat_h = (input_height_ + stride - 1) / stride;
        int feat_w = (input_width_ + stride - 1) / stride;
        std::string key = "stride" + std::to_string(stride);
        int num_anchors_per_loc = _num_anchors[key];
        total_anchors += feat_h * feat_w * num_anchors_per_loc;
    }

    size_t current_anchor_offset = 0;
    for (int stride : _feat_stride_fpn) {
        int feat_h = (input_height_ + stride - 1) / stride;
        int feat_w = (input_width_ + stride - 1) / stride;
        std::string key = "stride" + std::to_string(stride);
        int num_anchors_per_loc = _num_anchors[key];
        int num_anchors_on_map = feat_h * feat_w * num_anchors_per_loc;

        std::vector<anchor_box> anchors = anchors_plane(feat_h, feat_w, stride, _anchors_fpn[key]);
        
        for (size_t i = 0; i < anchors.size(); ++i) {
            float score = scores_ptr[current_anchor_offset + i];
            
            if (score < threshold) continue;

            cv::Vec4f regress;
            regress[0] = boxes_ptr[(current_anchor_offset + i) * 4 + 0];
            regress[1] = boxes_ptr[(current_anchor_offset + i) * 4 + 1];
            regress[2] = boxes_ptr[(current_anchor_offset + i) * 4 + 2];
            regress[3] = boxes_ptr[(current_anchor_offset + i) * 4 + 3];

            anchor_box rect = bbox_pred(anchors[i], regress);
            clip_boxes(rect, input_width_, input_height_);

            FacePts pts;
            for (size_t j = 0; j < 5; ++j) {
                pts.x[j] = landmarks_ptr[(current_anchor_offset + i) * 10 + j * 2 + 0];
                pts.y[j] = landmarks_ptr[(current_anchor_offset + i) * 10 + j * 2 + 1];
            }
            FacePts landmarks = landmark_pred(anchors[i], pts);

            FaceDetectInfo info;
            info.score = score;
            info.rect = rect;
            info.pts = landmarks;
            proposals.push_back(info);
        }
        current_anchor_offset += num_anchors_on_map;
    }

    // --- 5. NMS ---
    std::vector<FaceDetectInfo> final_faces = nms(proposals, nms_threshold_);

    // --- 6. Scale results back to original image size ---
    float scale_x = im_w / input_width_;
    float scale_y = im_h / input_height_;
    for (auto& face : final_faces) {
        face.rect.x1 *= scale_x;
        face.rect.y1 *= scale_y;
        face.rect.x2 *= scale_x;
        face.rect.y2 *= scale_y;
        for (int i = 0; i < 5; ++i) {
            face.pts.x[i] *= scale_x;
            face.pts.y[i] *= scale_y;
        }
    }

    return final_faces;
}

void RetinaFace::preprocess(cv::Mat& img, std::vector<float>& input_tensor_values) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_width_, input_height_));

    resized_img.convertTo(resized_img, CV_32FC3);
    resized_img -= cv::Scalar(pixel_means[0], pixel_means[1], pixel_means[2]);

    input_tensor_values.resize(1 * 3 * input_height_ * input_width_);
    
    // HWC to CHW conversion
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                // BGR to RGB is handled implicitly by reversing the channel loop
                input_tensor_values[c * (input_height_ * input_width_) + h * input_width_ + w] =
                    resized_img.at<cv::Vec3f>(h, w)[2-c]; // 2-c for BGR->RGB
            }
        }
    }
}

// --- Framework-Independent Helper Methods (No changes needed) ---

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress) {
    anchor_box rect;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5f * (width - 1.0f);
    float ctr_y = anchor.y1 + 0.5f * (height - 1.0f);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    rect.y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    rect.x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    rect.y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);
    return rect;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt) {
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5f * (width - 1.0f);
    float ctr_y = anchor.y1 + 0.5f * (height - 1.0f);

    for (size_t j = 0; j < 5; j++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }
    return pt;
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