// sherpa-onnx/csrc/speaker-segmentation-ascend-compare.cc

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/wave-reader.h"

namespace {

using Clock = std::chrono::steady_clock;
using sherpa_onnx::OfflineSpeakerSegmentationModelConfig;
using sherpa_onnx::OfflineSpeakerSegmentationPyannoteModel;

constexpr int32_t kSampleRate = 16000;
constexpr int64_t kWindowSize = 160000;
constexpr int64_t kNumFrames = 589;
constexpr int64_t kNumClasses = 7;

struct RunResult {
  std::vector<float> output;
  std::vector<int64_t> shape;
  double average_ms = 0;
};

RunResult RunModel(const OfflineSpeakerSegmentationPyannoteModel &model,
                   std::vector<float> *samples, int32_t iterations) {
  RunResult result;
  double elapsed_ms = 0;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 3> input_shape{1, 1, kWindowSize};

  for (int32_t i = 0; i != iterations; ++i) {
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info, samples->data(), samples->size(), input_shape.data(),
        input_shape.size());
    const auto begin = Clock::now();
    Ort::Value output = model.Forward(std::move(input));
    const auto end = Clock::now();
    elapsed_ms +=
        std::chrono::duration<double, std::milli>(end - begin).count();

    result.shape = output.GetTensorTypeAndShapeInfo().GetShape();
    const size_t count = output.GetTensorTypeAndShapeInfo().GetElementCount();
    const float *data = output.GetTensorData<float>();
    result.output.assign(data, data + count);
  }

  result.average_ms = elapsed_ms / iterations;
  return result;
}

int32_t ArgMax(const float *p, int32_t size) {
  return static_cast<int32_t>(
      std::max_element(p, p + size) - p);
}

int Compare(int32_t argc, char **argv, bool *acl_initialized) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " <cpu-model.onnx> <ascend-model.om> <audio.wav> "
                 "[iterations=5]\n";
    return EXIT_FAILURE;
  }

  const int32_t iterations = argc == 5 ? std::stoi(argv[4]) : 5;
  if (iterations <= 0) {
    std::cerr << "iterations must be greater than 0\n";
    return EXIT_FAILURE;
  }

  int32_t sample_rate = 0;
  bool ok = false;
  std::vector<float> wave =
      sherpa_onnx::ReadWave(argv[3], &sample_rate, &ok);
  if (!ok || wave.empty()) {
    std::cerr << "Failed to read WAV: " << argv[3] << "\n";
    return EXIT_FAILURE;
  }
  if (sample_rate != kSampleRate) {
    std::cerr << "Expected a 16 kHz WAV. Given sample rate: " << sample_rate
              << "\n";
    return EXIT_FAILURE;
  }

  std::vector<float> samples(kWindowSize, 0);
  std::copy_n(wave.data(), std::min(wave.size(), samples.size()),
              samples.data());

  OfflineSpeakerSegmentationModelConfig cpu_config;
  cpu_config.pyannote.model = argv[1];
  cpu_config.provider = "cpu";
  cpu_config.num_threads = 4;
  OfflineSpeakerSegmentationPyannoteModel cpu(cpu_config);

  OfflineSpeakerSegmentationModelConfig ascend_config;
  ascend_config.pyannote.model = argv[2];
  ascend_config.provider = "ascend";
  ascend_config.num_threads = 1;
  OfflineSpeakerSegmentationPyannoteModel ascend(ascend_config);
  *acl_initialized = true;

  (void)RunModel(ascend, &samples, 1);
  RunResult cpu_result = RunModel(cpu, &samples, iterations);
  RunResult ascend_result = RunModel(ascend, &samples, iterations);

  const std::vector<int64_t> expected_shape{1, kNumFrames, kNumClasses};
  if (cpu_result.shape != expected_shape ||
      ascend_result.shape != expected_shape ||
      cpu_result.output.size() != ascend_result.output.size()) {
    std::cerr << "Unexpected or mismatched output shape\n";
    return EXIT_FAILURE;
  }

  double abs_sum = 0;
  double squared_sum = 0;
  double max_abs = 0;
  size_t argmax_matches = 0;
  for (size_t i = 0; i != cpu_result.output.size(); ++i) {
    const double diff =
        std::abs(static_cast<double>(cpu_result.output[i]) -
                 static_cast<double>(ascend_result.output[i]));
    abs_sum += diff;
    squared_sum += diff * diff;
    max_abs = std::max(max_abs, diff);
  }
  for (int64_t frame = 0; frame != kNumFrames; ++frame) {
    const size_t offset = static_cast<size_t>(frame * kNumClasses);
    if (ArgMax(cpu_result.output.data() + offset, kNumClasses) ==
        ArgMax(ascend_result.output.data() + offset, kNumClasses)) {
      ++argmax_matches;
    }
  }

  const double count = static_cast<double>(cpu_result.output.size());
  const double frame_count = static_cast<double>(kNumFrames);
  std::cout << std::fixed << std::setprecision(9)
            << "input_samples=" << samples.size() << "\n"
            << "output_frames=" << kNumFrames << "\n"
            << "output_classes=" << kNumClasses << "\n"
            << "mean_absolute_error=" << abs_sum / count << "\n"
            << "root_mean_squared_error="
            << std::sqrt(squared_sum / count) << "\n"
            << "max_absolute_error=" << max_abs << "\n"
            << "argmax_matches=" << argmax_matches << "\n"
            << "argmax_agreement=" << argmax_matches / frame_count << "\n"
            << "cpu_average_ms=" << cpu_result.average_ms << "\n"
            << "ascend_average_ms=" << ascend_result.average_ms << "\n"
            << "speedup="
            << (ascend_result.average_ms >
                        std::numeric_limits<double>::epsilon()
                    ? cpu_result.average_ms / ascend_result.average_ms
                    : 0)
            << "\n";

  return EXIT_SUCCESS;
}

}  // namespace

int main(int32_t argc, char **argv) {
  bool acl_initialized = false;
  const int rc = Compare(argc, argv, &acl_initialized);
  if (acl_initialized) {
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      std::cerr << "aclFinalize failed with error code: " << ret << "\n";
      return EXIT_FAILURE;
    }
  }
  return rc;
}
