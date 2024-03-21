// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/classification.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
// Runs face detection on the Edge TPU, using the on-board camera, printing
// results to the serial console and turning on the User LED when a face
// is detected.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e face_detection

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/hand_track_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 16 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  printf("hand track Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  printf("Loading model\r\n");
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Model loaded\r\n");

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);
  printf("Camera started\r\n");
  auto* input_tensor = interpreter.input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];

  while (true) {
    CameraFrameFormat fmt{CameraFormat::kRgb,
                          CameraFilterMethod::kBilinear,
                          CameraRotation::k270,
                          model_width,
                          model_height,
                          false,
                          tflite::GetTensorData<uint8_t>(input_tensor)};
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }

    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    // regressor
    auto* reg_tensor = interpreter.output_tensor(0);
    TfLiteIntArray* reg_output_dims = reg_tensor->dims;
    int reg_rows = reg_output_dims->data[reg_output_dims->size - 2];
    int reg_cols = reg_output_dims->data[reg_output_dims->size - 1];
    float* regressors = reg_tensor->data.f;
    std::vector<float> regressor_vector(reg_rows * reg_cols);

    // classification

    auto* cls_tensor = interpreter.output_tensor(1);
    TfLiteIntArray* cls_output_dims = cls_tensor->dims;
    int cls_rows = cls_output_dims->data[cls_output_dims->size - 2];
    int cls_cols = cls_output_dims->data[cls_output_dims->size - 1];
    float* classifications = cls_tensor->data.f;
    std::vector<float> classification_vector(cls_rows * cls_cols);

    // copy output tensor 
    memcpy(&(regressor_vector[0]), regressors, sizeof(float) * reg_rows * reg_cols);
    memcpy(&(classification_vector[0]), classifications, sizeof(float) * cls_rows * cls_cols);
    // auto results = tensorflow::GetClassificationResults(&interpreter, kThreshold, kTopK); 
    auto results = tensorflow::GetDetectionResults(&interpreter, kThreshold, kTopK);

    // if (!results.empty()) {
    //   for(auto result : results){
    //     printf("Label ID: %d, Score: %f\r\n", result.id, result.score);
    //   }
    //   printf("Found %d face(s):\r\n%s\r\n", results.size(),
    //          tensorflow::FormatClassificationOutput(results).c_str());
    //   LedSet(Led::kUser, true);
    // } else {
    //   LedSet(Led::kUser, false);
    // }
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
