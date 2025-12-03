#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// YOLOv8 설정
#define MODEL_INPUT_SIZE 640
#define CONF_THRESHOLD 0.25
#define NMS_THRESHOLD 0.45
#define IOU_THRESHOLD_MAP 0.5

// 검출 결과 구조체
struct Detection {
    int class_id;
    float confidence;
    Rect2f box;
};

// Ground Truth 구조체
struct GroundTruth {
    int class_id;
    Rect2f box;
};

// 타이머 유틸리티
double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// IOU 계산
float calculate_iou(const Rect2f& box1, const Rect2f& box2) {
    float x1 = max(box1.x, box2.x);
    float y1 = max(box1.y, box2.y);
    float x2 = min(box1.x + box1.width, box2.x + box2.width);
    float y2 = min(box1.y + box1.height, box2.y + box2.height);
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float union_area = box1.width * box1.height + box2.width * box2.height - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

// NMS (Non-Maximum Suppression)
vector<Detection> nms(vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return detections;
    
    sort(detections.begin(), detections.end(), 
         [](const Detection& a, const Detection& b) {
             return a.confidence > b.confidence;
         });
    
    vector<Detection> result;
    vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculate_iou(detections[i].box, detections[j].box);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

// YOLOv8 후처리
vector<Detection> postprocess(rknn_output* outputs, int num_outputs,
                              int img_w, int img_h, int input_w, int input_h,
                              float conf_threshold, int num_classes) {
    vector<Detection> detections;
    
    float* output = (float*)outputs[0].buf;
    
    // YOLOv8의 Stride는 (4 + 클래스개수)
    int stride = 4 + num_classes; 
    
    // 박스 개수 계산
    int num_boxes = outputs[0].size / (stride * sizeof(float));

    for (int i = 0; i < num_boxes; i++) {
        float* ptr = output + i * stride;
        
        // [수정 2] YOLOv8은 별도의 objectness score가 없습니다.
        // 클래스 확률 중 가장 높은 것이 곧 confidence입니다.
        
        float max_class_score = -1.0f;
        int class_id = -1;

        // 클래스 점수들 중 최댓값 찾기
        // ptr[0~3]은 좌표, ptr[4]부터 클래스 점수 시작
        for (int c = 0; c < num_classes; c++) {
            float score = ptr[4 + c];
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }

        // 임계값보다 낮으면 무시
        if (max_class_score < conf_threshold) continue;

        // 좌표 복원
        float x = ptr[0] * img_w / input_w;
        float y = ptr[1] * img_h / input_h;
        float w = ptr[2] * img_w / input_w;
        float h = ptr[3] * img_h / input_h;

        Detection det;
        det.class_id = class_id;
        det.confidence = max_class_score;
        det.box = Rect(x - w / 2, y - h / 2, w, h);
        detections.push_back(det);
    }
    return detections;
}

// mAP@50 계산
float calculate_map50(const vector<vector<Detection>>& all_predictions,
                      const vector<vector<GroundTruth>>& all_ground_truths,
                      int num_classes) {
    vector<float> aps(num_classes, 0.0f);
    int valid_classes = 0;
    
    for (int c = 0; c < num_classes; c++) {
        vector<pair<float, bool>> predictions; // (confidence, is_true_positive)
        int total_gt = 0;
        
        // 모든 이미지에 대해 처리
        for (size_t img_idx = 0; img_idx < all_predictions.size(); img_idx++) {
            vector<bool> gt_matched(all_ground_truths[img_idx].size(), false);
            
            // 해당 클래스의 예측 처리
            for (const auto& pred : all_predictions[img_idx]) {
                if (pred.class_id != c) continue;
                
                bool is_tp = false;
                float max_iou = 0.0f;
                int max_idx = -1;
                
                // 가장 높은 IOU를 가진 GT 찾기
                for (size_t j = 0; j < all_ground_truths[img_idx].size(); j++) {
                    if (all_ground_truths[img_idx][j].class_id != c || gt_matched[j]) 
                        continue;
                    
                    float iou = calculate_iou(pred.box, all_ground_truths[img_idx][j].box);
                    if (iou > max_iou) {
                        max_iou = iou;
                        max_idx = j;
                    }
                }
                
                if (max_iou >= IOU_THRESHOLD_MAP && max_idx >= 0) {
                    is_tp = true;
                    gt_matched[max_idx] = true;
                }
                predictions.push_back({pred.confidence, is_tp});
            }
            
            // GT 개수 세기
            for (const auto& gt : all_ground_truths[img_idx]) {
                if (gt.class_id == c) total_gt++;
            }
        }
        
        if (total_gt == 0) continue;
        
        // confidence 기준 정렬
        sort(predictions.begin(), predictions.end(),
             [](const pair<float, bool>& a, const pair<float, bool>& b) {
                 return a.first > b.first;
             });
        
        // AP 계산 (11-point interpolation)
        vector<float> precisions, recalls;
        int tp = 0, fp = 0;
        
        for (const auto& pred : predictions) {
            if (pred.second) tp++;
            else fp++;
            
            float precision = (float)tp / (tp + fp);
            float recall = (float)tp / total_gt;
            
            precisions.push_back(precision);
            recalls.push_back(recall);
        }
        
        // AP 계산 (interpolated)
        float ap = 0.0f;
        for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
            float max_prec = 0.0f;
            for (size_t i = 0; i < recalls.size(); i++) {
                if (recalls[i] >= t) {
                    max_prec = max(max_prec, precisions[i]);
                }
            }
            ap += max_prec;
        }
        ap /= 11.0f;
        
        aps[c] = ap;
        if (ap > 0) valid_classes++;
    }
    
    // mAP 계산
    float sum = 0.0f;
    for (float ap : aps) {
        sum += ap;
    }
    return valid_classes > 0 ? sum / valid_classes : 0.0f;
}

// F1 Score 계산
void calculate_f1_score(const vector<vector<Detection>>& all_predictions,
                        const vector<vector<GroundTruth>>& all_ground_truths,
                        float& precision, float& recall, float& f1) {
    int tp = 0, fp = 0, fn = 0;
    
    for (size_t i = 0; i < all_predictions.size(); i++) {
        vector<bool> gt_matched(all_ground_truths[i].size(), false);
        
        for (const auto& pred : all_predictions[i]) {
            bool matched = false;
            for (size_t j = 0; j < all_ground_truths[i].size(); j++) {
                if (gt_matched[j]) continue;
                if (pred.class_id == all_ground_truths[i][j].class_id) {
                    float iou = calculate_iou(pred.box, all_ground_truths[i][j].box);
                    if (iou >= IOU_THRESHOLD_MAP) {
                        tp++;
                        gt_matched[j] = true;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) fp++;
        }
        
        for (size_t j = 0; j < all_ground_truths[i].size(); j++) {
            if (!gt_matched[j]) fn++;
        }
    }
    
    precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.0f;
    recall = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.0f;
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
}

// Ground Truth 로드 (YOLO 형식)
vector<GroundTruth> load_ground_truth(const string& label_path, int img_w, int img_h) {
    vector<GroundTruth> gts;
    ifstream file(label_path);
    if (!file.is_open()) return gts;
    
    int class_id;
    float x, y, w, h;
    while (file >> class_id >> x >> y >> w >> h) {
        GroundTruth gt;
        gt.class_id = class_id;
        gt.box = Rect2f((x - w/2) * img_w, (y - h/2) * img_h, w * img_w, h * img_h);
        gts.push_back(gt);
    }
    file.close();
    return gts;
}

// 이미지 파일 리스트 가져오기
vector<string> get_image_files(const string& dir_path) {
    vector<string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (dir == NULL) return files;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename.find(".jpg") != string::npos || 
            filename.find(".png") != string::npos ||
            filename.find(".jpeg") != string::npos) {
            files.push_back(dir_path + "/" + filename);
        }
    }
    closedir(dir);
    sort(files.begin(), files.end());
    return files;
}

// letterbox 전처리
Mat letterbox(const Mat& img, int target_size, Scalar color = Scalar(114, 114, 114)) {
    int h = img.rows;
    int w = img.cols;
    float scale = min((float)target_size / w, (float)target_size / h);
    
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    Mat resized;
    resize(img, resized, Size(new_w, new_h));
    
    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;
    
    Mat padded;
    copyMakeBorder(resized, padded, top, bottom, left, right, BORDER_CONSTANT, color);
    
    return padded;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <image_dir> <label_dir> [num_classes]\n", argv[0]);
        printf("Example: %s ./test_images ./test_labels 80\n", argv[0]);
        return -1;
    }
    
    const char* image_dir = argv[1];
    const char* label_dir = argv[2];
    int num_classes = argc > 3 ? atoi(argv[3]) : 80;
    
    const char* model_path = "./yolov8s.rknn";
    
    printf("========================================\n");
    printf("     YOLOv8 RKNN Benchmark Tool\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path);
    printf("Image Directory: %s\n", image_dir);
    printf("Label Directory: %s\n", label_dir);
    printf("Number of Classes: %d\n", num_classes);
    printf("========================================\n\n");
    
    // RKNN 모델 로드
    FILE* fp = fopen(model_path, "rb");
    if (fp == NULL) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* model_data = malloc(model_len);
    fread(model_data, 1, model_len, fp);
    fclose(fp);
    
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }
    printf("✓ Model loaded successfully\n");
    
    // 입출력 정보 가져오기
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("✓ Model has %d inputs and %d outputs\n", io_num.n_input, io_num.n_output);
    
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        printf("  Input[%d]: dims=%d %d %d %d, type=%d\n", i,
               input_attrs[i].dims[0], input_attrs[i].dims[1],
               input_attrs[i].dims[2], input_attrs[i].dims[3],
               input_attrs[i].type);
    }
    
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        printf("  Output[%d]: dims=%d %d %d %d, type=%d\n", i,
               output_attrs[i].dims[0], output_attrs[i].dims[1],
               output_attrs[i].dims[2], output_attrs[i].dims[3],
               output_attrs[i].type);
    }
    
    // 이미지 파일 로드
    vector<string> image_files = get_image_files(image_dir);
    if (image_files.empty()) {
        printf("No images found in %s\n", image_dir);
        rknn_destroy(ctx);
        return -1;
    }
    printf("✓ Found %zu images\n\n", image_files.size());
    
    vector<vector<Detection>> all_predictions;
    vector<vector<GroundTruth>> all_ground_truths;
    
    double total_time = 0;
    int frame_count = 0;
    
    printf("Processing images...\n");
    
    // 추론 실행
    for (size_t idx = 0; idx < image_files.size(); idx++) {
        const string& img_path = image_files[idx];
        Mat img = imread(img_path);
        if (img.empty()) {
            printf("Failed to load: %s\n", img_path.c_str());
            continue;
        }
        
        int orig_w = img.cols;
        int orig_h = img.rows;
        
        // 전처리
        Mat input_img = letterbox(img, MODEL_INPUT_SIZE);
        
        // RKNN 입력 설정
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input_img.data;
        
        double start = get_current_time();
        
        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0) {
            printf("rknn_inputs_set failed! ret=%d\n", ret);
            continue;
        }
        
        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            printf("rknn_run failed! ret=%d\n", ret);
            continue;
        }
        
        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 1;
        }
        
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("rknn_outputs_get failed! ret=%d\n", ret);
            continue;
        }
        
        double end = get_current_time();
        total_time += (end - start);
        frame_count++;
        
        // 후처리
        vector<Detection> dets = postprocess_yolov8(outputs, io_num.n_output,
                                                     orig_w, orig_h, 
                                                     MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                                                     CONF_THRESHOLD, num_classes);
        dets = nms(dets, NMS_THRESHOLD);
        all_predictions.push_back(dets);
        
        // Ground Truth 로드
        string filename = img_path.substr(img_path.find_last_of("/") + 1);
        string label_path = string(label_dir) + "/" + 
                           filename.substr(0, filename.find_last_of(".")) + ".txt";
        vector<GroundTruth> gts = load_ground_truth(label_path, orig_w, orig_h);
        all_ground_truths.push_back(gts);
        
        rknn_outputs_release(ctx, io_num.n_output, outputs);
        
        if ((idx + 1) % 10 == 0 || idx == image_files.size() - 1) {
            printf("\rProcessed: %zu/%zu images", idx + 1, image_files.size());
            fflush(stdout);
        }
    }
    printf("\n\n");
    
    // 벤치마크 결과 계산
    float fps = frame_count / (total_time / 1000.0);
    float avg_inference_time = total_time / frame_count;
    
    float precision, recall, f1;
    calculate_f1_score(all_predictions, all_ground_truths, precision, recall, f1);
    
    float map50 = calculate_map50(all_predictions, all_ground_truths, num_classes);
    
    // 결과 출력
    printf("========================================\n");
    printf("         BENCHMARK RESULTS\n");
    printf("========================================\n");
    printf("Total Images:        %d\n", frame_count);
    printf("Total Time:          %.2f ms\n", total_time);
    printf("Avg Inference Time:  %.2f ms\n", avg_inference_time);
    printf("FPS:                 %.2f\n", fps);
    printf("----------------------------------------\n");
    printf("Precision:           %.4f (%.2f%%)\n", precision, precision * 100);
    printf("Recall:              %.4f (%.2f%%)\n", recall, recall * 100);
    printf("F1 Score:            %.4f (%.2f%%)\n", f1, f1 * 100);
    printf("mAP@50:              %.4f (%.2f%%)\n", map50, map50 * 100);
    printf("========================================\n");
    
    rknn_destroy(ctx);
    return 0;
}
