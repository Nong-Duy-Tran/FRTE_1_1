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
    std::cout << model_path << std::endl << image_path << std::endl;

    RetinaFace detector(0.4, false);
    detector.initialize(model_path);

    std::cout<<"abcabc"<<std::endl;
    cv::Mat img = cv::imread(image_path);
    // std::cout<<img<<std::endl;
    std::vector<FaceDetectInfo> faces = detector.detect(img, 0.6f);
    return 0;
}