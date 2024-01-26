# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable_m7(mediapipe_face_detect
    face_detect_mediapipe.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/face-detector-quantized_edgetpu.tflite
)

target_link_libraries(mediapipe_face_detect
    libs_base-m7_freertos
)
