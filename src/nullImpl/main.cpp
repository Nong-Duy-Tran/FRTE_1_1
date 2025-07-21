#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <RetinaFace.h>
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

    std::string model_path = exec_dir + "/../src/model/retinaface_mobilenet25.onnx";
    std::string image_path = exec_dir + "/../src/img/images.jpeg";
    std::string output_path = exec_dir + "/../src/img/output.jpeg";

    std::cout << model_path << std::endl << image_path << std::endl;

    RetinaFace detector(0.4, false);
    detector.initialize(model_path);

    cv::Mat img = cv::imread(image_path);

    std::vector<FaceDetectInfo> faces = detector.detect(img, 0.6f);

    for (const auto& face : faces) {
        // --- 1. Draw the Bounding Box ---
        // Get the rectangle coordinates
        cv::Point pt1(static_cast<int>(face.rect.x1), static_cast<int>(face.rect.y1));
        cv::Point pt2(static_cast<int>(face.rect.x2), static_cast<int>(face.rect.y2));
        
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
    return 0;
}