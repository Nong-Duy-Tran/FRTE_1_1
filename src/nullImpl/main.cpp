#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <RetinaFace.h>

int main() {
    std::string model_path = "./model/retinaface_mobilenet25.onnx";
    RetinaFace::RetinaFace();
}