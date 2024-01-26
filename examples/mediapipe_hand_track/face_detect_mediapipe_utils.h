/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LIBS_TENSORFLOW_CLASSIFICATION_H_
#define LIBS_TENSORFLOW_CLASSIFICATION_H_

#include <limits>
#include <vector>

#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"

namespace coralmicro::tensorflow {

// Represents a classification result.
struct Detection {
  float score;
  float class_id;
  float xmin;
  float ymin;
  float width;
  float height;
};

struct Anchor {
  float x_center;
  float y_center;
  float h;
  float w;
};

}

#endif  // LIBS_TENSORFLOW_CLASSIFICATION_H_
