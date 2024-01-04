#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>

#include "resource.h"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <Windows.h>
#include <iostream>

using namespace std;
using namespace cv;

bool Carregado = false;
cv::Ptr<cv::FaceDetectorYN> FaceDetector = nullptr;
HGLOBAL HData;
cv::dnn::Net ModelONNX;
int Divisor = 32;

// Model Configs
float ScoreThreshold = 0.9f;
float NmsThreshold = 0.3f;
int TopK = 5000;

static void visualize(Mat& input, Mat& faces, int thickness = 2)
{
    for (int i = 0; i < faces.rows; i++)
    {
        // Draw bounding box
        rectangle(input, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
    }
}

Mat postProcess(const std::vector<Mat>& output_blobs, int padW, int padH)
{
    std::vector<int> strides { 8, 16, 32 };
    Mat faces;
    for (size_t i = 0; i < strides.size(); ++i) {
        int cols = int(padW / strides[i]);
        int rows = int(padH / strides[i]);

        // Extract from output_blobs
        Mat cls = output_blobs[i];
        Mat obj = output_blobs[i + strides.size() * 1];
        Mat bbox = output_blobs[i + strides.size() * 2];
        Mat kps = output_blobs[i + strides.size() * 3];

        // Decode from predictions
        float* cls_v = (float*)(cls.data);
        float* obj_v = (float*)(obj.data);
        float* bbox_v = (float*)(bbox.data);
        float* kps_v = (float*)(kps.data);

        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        Mat face(1, 15, CV_32FC1);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                size_t idx = r * cols + c;

                // Get score
                float cls_score = cls_v[idx];
                float obj_score = obj_v[idx];

                // Clamp
                cls_score = MIN(cls_score, 1.f);
                cls_score = MAX(cls_score, 0.f);
                obj_score = MIN(obj_score, 1.f);
                obj_score = MAX(obj_score, 0.f);
                float score = std::sqrt(cls_score * obj_score);
                face.at<float>(0, 14) = score;

                // Get bounding box
                float cx = ((c + bbox_v[idx * 4 + 0]) * strides[i]);
                float cy = ((r + bbox_v[idx * 4 + 1]) * strides[i]);
                float w = exp(bbox_v[idx * 4 + 2]) * strides[i];
                float h = exp(bbox_v[idx * 4 + 3]) * strides[i];

                float x1 = cx - w / 2.f;
                float y1 = cy - h / 2.f;

                face.at<float>(0, 0) = x1;
                face.at<float>(0, 1) = y1;
                face.at<float>(0, 2) = w;
                face.at<float>(0, 3) = h;

                // Get landmarks
                for (int n = 0; n < 5; ++n) {
                    face.at<float>(0, 4 + 2 * n) = (kps_v[idx * 10 + 2 * n] + c) * strides[i];
                    face.at<float>(0, 4 + 2 * n + 1) = (kps_v[idx * 10 + 2 * n + 1] + r) * strides[i];
                }
                faces.push_back(face);
            }
        }
    }

    if (faces.rows > 1)
    {
        // Retrieve boxes and scores
        std::vector<Rect2i> faceBoxes;
        std::vector<float> faceScores;
        for (int rIdx = 0; rIdx < faces.rows; rIdx++)
        {
            faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                int(faces.at<float>(rIdx, 1)),
                int(faces.at<float>(rIdx, 2)),
                int(faces.at<float>(rIdx, 3))));
            faceScores.push_back(faces.at<float>(rIdx, 14));
        }

        std::vector<int> keepIdx;
        dnn::NMSBoxes(faceBoxes, faceScores, ScoreThreshold, NmsThreshold, keepIdx, 1.f, TopK);

        // Get NMS results
        Mat nms_faces;
        for (int idx : keepIdx)
        {
            nms_faces.push_back(faces.row(idx));
        }
        return nms_faces;
    }
    else
    {
        return faces;
    }
}

cv::Mat faceChooser(cv::Mat& faces, cv::Mat& facePrincipal) {

    float areaMaior = 0;

    for (int i = 0; i < faces.rows; i++) {
        float areaAtual = faces.at<float>(i, 2) * faces.at<float>(i, 3);
        if (areaAtual > areaMaior) {
            areaMaior = areaAtual;
            facePrincipal = faces.row(i);
        }
    }
    return facePrincipal;
}

cv::Mat extraiFaces(cv::Mat resizedInput)
{
    cv::Mat inputDetectFace = resizedInput.clone(); // Imagem ja regulada
    cv::Mat pad_image;

    int padW = ((inputDetectFace.cols - 1) / Divisor + 1) * Divisor;
    int padH = ((inputDetectFace.rows - 1) / Divisor + 1) * Divisor;
    int bottom = padH - inputDetectFace.rows;
    int right = padW - inputDetectFace.cols;

    cv::copyMakeBorder(inputDetectFace, pad_image, 0, bottom, 0, right, BORDER_CONSTANT, 0);

    cv::Mat input_blob = cv::dnn::blobFromImage(pad_image);

    std::vector<String> output_names = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
    std::vector<Mat> output_blobs;
    ModelONNX.setInput(input_blob);
    ModelONNX.forward(output_blobs, output_names);

    // Post process
    cv::Mat finalResult;
    Mat results = postProcess(output_blobs, padW, padH);
    results.convertTo(finalResult, CV_32FC1);

    return finalResult;
}



bool initializeModel() {
    try {

        HMODULE hModule;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&visualize, &hModule); // para obter o handle da DLL e nao do consumidor

        HRSRC hRes = FindResource(hModule, MAKEINTRESOURCE(IDR_FACEDETECT), L"BIN");
        if (hRes == NULL) {
            throw std::invalid_argument("Modelo não carregado!");
            std::cerr << "Modelo não carregado!" << std::endl;
            return false;
        }

        HData = LoadResource(hModule, hRes);

        void* pData = LockResource(HData);

        const char* model = reinterpret_cast<const char*>(pData);

        size_t dataSize = SizeofResource(hModule, hRes);

        ModelONNX = cv::dnn::readNetFromONNX(model, dataSize);

        Carregado = true;

        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return false;
    }
    catch (...)
    {
        return false;
    }
}

Mat hwnd2mat(HWND hwnd)
{
    HDC hwindowDC, hwindowCompatibleDC;

    int height, width, srcheight, srcwidth;
    HBITMAP hbwindow;
    Mat src;
    BITMAPINFOHEADER  bi;

    hwindowDC = GetDC(hwnd);
    hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hwnd, &windowsize);

    srcheight = windowsize.bottom;
    srcwidth = windowsize.right;
    height = windowsize.bottom;  //change this to whatever size you want to resize to
    width = windowsize.right;

    src.create(height, width, CV_8UC4);

    // create a bitmap
    hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    // copy from the window device context to the bitmap device context
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

    // avoid memory leak
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hwnd, hwindowDC);

    return src;
}

int main(int argc, char** argv)
{

    if (initializeModel()) {

        HWND hwndDesktop = GetDesktopWindow();
        namedWindow("output", WINDOW_AUTOSIZE);
        int key = 0;


        while (key != 27)
        {
            Mat src = hwnd2mat(hwndDesktop);

            Mat srcWithoutAlpha;
            cvtColor(src, srcWithoutAlpha, cv::COLOR_BGRA2BGR);

            auto channels = srcWithoutAlpha.channels();
            // Mat digs = imread("C:\\Users\\joaodev\\Desktop\\3por4.jpg", IMREAD_UNCHANGED);
            Mat outputFaceDetector = extraiFaces(srcWithoutAlpha);
            
            if (outputFaceDetector.rows != 0) {
                cout << outputFaceDetector.rows << "Faces Detected!" << endl;
            }
            else {
                cout << "No Face Detected!" << endl;
            }

            cv::Mat testin = srcWithoutAlpha.clone();
            visualize(testin , outputFaceDetector);
            
            // you can do some image processing here
            imshow("output", testin);
            key = waitKey(60); // you can change wait time
        }
    }
 

}