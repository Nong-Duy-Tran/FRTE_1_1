/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>

#include "nullimplfrvt11.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {}

double calculateCosineSimilarity(const float* vecA, const float* vecB, size_t size) {
    if (size == 0) {
        return 0.0;
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (size_t i = 0; i < size; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    double magnitude = std::sqrt(normA) * std::sqrt(normB);
    
    if (magnitude < 1e-6) { // Avoid division by zero
        return 0.0;
    }

    return dotProduct / magnitude;
}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
    this->configDir = configDir;
    string arcFaceModelPath = configDir + "/model/arcface.onnx";
    string retinaFaceModelPath = configDir + "/model/retinaface_mobilenet25.onnx";

    recognizor.initialize(arcFaceModelPath);
    detector.initialize(retinaFaceModelPath);

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::createFaceTemplate(
        const std::vector<FRVT::Image> &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{
    vector<float> fv;
    vector<FaceDetectInfo> detectionInfo;
    for (int i = 0; i < faces.size(); i++) 
    {
        int height = faces[i].height - 1;
        int width = faces[i].width - 1;
        cv::Mat faceMat;
        vector<FaceDetectInfo>detectInfo_;
        bool canDetect = true;
        
        if (faces[i].depth == 24) {
            faceMat = cv::Mat(height, width, CV_8UC3, faces[i].data.get());
        } else if (faces[i].depth == 8) {
            faceMat = cv::Mat(height, width, CV_8UC1, faces[i].data.get());
        }

        if (!faceMat.empty()) {
            detectInfo_ = detector.detect(faceMat);
        }

        if (detectInfo_.empty()) {
            canDetect = false;

            FacePts landmarks;
            for (size_t j = 0; j < 5; j++)
            {
                landmarks.x[j] = j;
                landmarks.y[j] = j;
            }

            FaceDetectInfo tmp;
            tmp.score = 0.0;
            // Generate an anchor inside the equation
            tmp.rect = anchor_box{0, 0, (float)width, (float)height};
            tmp.pts = landmarks;

            detectInfo_.push_back(tmp);
        }

        FacePts landmarks =  detectInfo_[0].pts;
        anchor_box face = detectInfo_[0].rect;
        eyeCoordinates.push_back(EyePair(true, true, 
            static_cast<uint16_t>(landmarks.x[0]), 
            static_cast<uint16_t>(landmarks.y[0]),
            static_cast<uint16_t>(landmarks.x[1]), 
            static_cast<uint16_t>(landmarks.y[1])));

        int x1 = max(0, static_cast<int>(face.x1));
        int y1 = max(0, static_cast<int>(face.y1));
        int x2 = min(width, static_cast<int>(face.x2));
        int y2 = min(height, static_cast<int>(face.y2));

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat cropImg;

        if (canDetect) {
            cropImg = faceMat(roi);
        }
        else {
            // If detection failed, use the whole image as crop
            cropImg = faceMat.clone();
        }

        vector<float> faceEmbedding = recognizor.GetEmbedding(cropImg, landmarks);
        fv.insert(fv.end(), faceEmbedding.begin(), faceEmbedding.end());
        cropImg.release();
        faceMat.release();

    }

    const uint8_t *bytes = reinterpret_cast<const uint8_t*>(fv.data());
    int dataSize = sizeof(float) * fv.size();
    templ.resize(dataSize);
    memcpy(templ.data(), bytes, dataSize);


    return ReturnStatus(ReturnCode::Success);
}


ReturnStatus
NullImplFRVT11::createFaceTemplate(
    const FRVT::Image &image,
    FRVT::TemplateRole role,
    std::vector<std::vector<uint8_t>> &templs,
    std::vector<FRVT::EyePair> &eyeCoordinates)
{
    vector<FaceDetectInfo> detectionInfo;

    int height = image.height;
    int width = image.width;
    cv::Mat faceMat;

    bool canDetect = true;

    if (image.depth == 24) {
        faceMat = cv::Mat(height, width, CV_8UC3, image.data.get());
    } else if (image.depth == 8) {
        faceMat = cv::Mat(height, width, CV_8UC1, image.data.get());
    }

    if (faceMat.empty()) {
        return ReturnStatus(ReturnCode::FaceDetectionError);    
    }

    detectionInfo = detector.detect(faceMat);

    if (detectionInfo.empty()) {
        canDetect = false;

        FacePts landmarks;
        for (size_t j = 0; j < 5; j++)
        {
            landmarks.x[j] = j;
            landmarks.y[j] = j;
        }

        FaceDetectInfo tmp;
        tmp.score = 0.0;
        // Generate an anchor inside the equation
        tmp.rect = anchor_box{0, 0, (float)width, (float)height};
        tmp.pts = landmarks;

        detectionInfo.push_back(tmp);
    }

    for (int i = 0; i < detectionInfo.size(); i++) {
        FacePts landmarks =  detectionInfo[i].pts;
        anchor_box face = detectionInfo[i].rect;
        eyeCoordinates.push_back(EyePair(true, true, 
            static_cast<uint16_t>(landmarks.x[0]), // Left eye
            static_cast<uint16_t>(landmarks.y[0]),

            static_cast<uint16_t>(landmarks.x[1]), // Right eye
            static_cast<uint16_t>(landmarks.y[1])));

        int x1 = max(0, static_cast<int>(face.x1));
        int y1 = max(0, static_cast<int>(face.y1));
        int x2 = min(width, static_cast<int>(face.x2));
        int y2 = min(height, static_cast<int>(face.y2));

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat cropImg;

        if (canDetect) {
            cropImg = faceMat(roi);
        }
        else {
            // If detection failed, use the whole image as crop
            cropImg = faceMat.clone();
        }

        vector<uint8_t> templ;
        vector<float> faceEmbedding = recognizor.GetEmbedding(cropImg, landmarks);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(faceEmbedding.data());
        int dataSize = sizeof(float) * faceEmbedding.size();
        templ.resize(dataSize);
        memcpy(templ.data(), bytes, dataSize);
        templs.push_back(templ);
    }

    faceMat.release();
    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &score)
{
    // The expected size in bytes of a single feature vector template.
    const size_t expected_template_size = this->featureVectorSize * sizeof(float);

    if (verifTemplate.size() != expected_template_size || enrollTemplate.size() != expected_template_size) {
        score = -1.0;
        return ReturnStatus(ReturnCode::VerifTemplateError); 
    }

    const float *verifVector = reinterpret_cast<const float*>(verifTemplate.data());
    const float *enrollVector = reinterpret_cast<const float*>(enrollTemplate.data());

    score = calculateCosineSimilarity(enrollVector, verifVector, this->featureVectorSize);
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}

