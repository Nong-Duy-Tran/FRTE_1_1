#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <RetinaFace.h>
#include <ArcFace.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <filesystem>

std::string get_executable_dir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    std::string path(result, (count > 0) ? count : 0);
    return std::filesystem::path(path).parent_path().string();
}


int main() {
    std::string exec_dir = get_executable_dir();

    std::string detect_model_path = exec_dir + "/../src/model/retinaface_mobilenet25.onnx";
    std::string recognize_model_path = exec_dir + "/../src/model/arcface.onnx";
    std::string image_path = exec_dir + "/../src/img/images.jpeg";
    std::string output_dir = exec_dir + "/../src/img/";
    std::string output_path = exec_dir + "/../src/img/output.jpeg";

    std::cout << detect_model_path << std::endl << recognize_model_path << std::endl;

    RetinaFace detector(0.4, false);
    detector.initialize(detect_model_path);

    cv::Mat img = cv::imread(image_path);
    std::vector<FaceDetectInfo> faces = detector.detect(img, 0.6f);
    std::cout << "Number of faces detected: " << faces.size() << std::endl;
    int face_counter = 0;

    for (const auto& face : faces) {
        int x1 = static_cast<int>(face.rect.x1);
        int y1 = static_cast<int>(face.rect.y1);
        int x2 = static_cast<int>(face.rect.x2);
        int y2 = static_cast<int>(face.rect.y2);

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(img.cols, x2);
        y2 = std::min(img.rows, y2);

        cv::Point pt1(static_cast<int>(x1), static_cast<int>(y1));
        cv::Point pt2(static_cast<int>(x2), static_cast<int>(y2));
        
        if (x1 < x2 && y1 < y2) {
            // --- 2. Define the Region of Interest (ROI) ---
            cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
            
            // --- 3. Crop the Image using the ROI ---
            // This creates a new cv::Mat that points to the cropped region of the original image.
            cv::Mat cropped_face = img(roi);

            // --- 4. Save the Cropped Face ---
            std::string cropped_filename = "face_" + std::to_string(face_counter) + ".jpg";
            std::filesystem::path cropped_output_path = output_dir + "/" + cropped_filename;
            
            bool success = cv::imwrite(cropped_output_path.string(), cropped_face);
            if (success) {
                std::cout << "Saved cropped face to: " << cropped_output_path << std::endl;
            } else {
                std::cerr << "Error: Failed to save cropped face to: " << cropped_output_path << std::endl;
            }
        }

        face_counter++;

        // Draw the rectangle on the image
        // Parameters: image, point1, point2, color (BGR), thickness
        cv::rectangle(img, pt1, pt2, cv::Scalar(0, 0, 255), 2); // Red rectangle

        // --- 2. Draw the Confidence Score ---
        // Prepare the text to display (e.g., "0.9987")
        std::string score_text = cv::format("%.4f", face.score);
        cv::Point text_origin(pt1.x, pt1.y - 10); // Position text slightly above the box

        // Draw the text
        // Parameters: image, text, origin, font, font_scale, color, thickness
        cv::putText(img, score_text, text_origin, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1); // White text

        // --- 3. Draw the Landmarks ---
        // Define colors for each landmark for better visualization
        cv::Scalar colors[] = {
            cv::Scalar(0, 0, 255),    // Left eye: Red
            cv::Scalar(0, 255, 255),  // Right eye: Yellow
            cv::Scalar(255, 0, 255),  // Nose: Magenta
            cv::Scalar(0, 255, 0),    // Left mouth corner: Green
            cv::Scalar(255, 0, 0)     // Right mouth corner: Blue
        };
        
        for (int i = 0; i < 5; ++i) {
            cv::Point landmark_pt(static_cast<int>(face.pts.x[i]), static_cast<int>(face.pts.y[i]));
            // Draw a circle for each landmark
            // Parameters: image, center, radius, color, thickness (use -1 for filled circle)
            cv::circle(img, landmark_pt, 2, colors[i], -1);
        }
    }

    // --- 4. Save the Resulting Image ---
    bool success = cv::imwrite(output_path, img);
    if (success) {
        std::cout << "Result image saved successfully to: " << output_path << std::endl;
    } else {
        std::cerr << "Error: Failed to save the image to: " << output_path << std::endl;
    }

    ArcFace recognizor;
    recognizor.initialize(recognize_model_path);

    cv::Mat crop_img = cv::imread(output_dir + "/" + "face_0.jpg");
    recognizor.GetEmbedding(crop_img, faces[0].pts);
    return 0;
}