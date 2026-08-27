// sherpa-onnx/csrc/ascend/offline-speaker-segmentation-pyannote-model-ascend.cc

#include "sherpa-onnx/csrc/ascend/offline-speaker-segmentation-pyannote-model-ascend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

constexpr int64_t kInputChannels = 1;
constexpr int64_t kWindowSize = 160000;
constexpr int64_t kNumFrames = 589;
constexpr int64_t kNumClasses = 7;
constexpr int64_t kMinBatchSize = 1;
constexpr int64_t kMaxBatchSize = 16;

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kReceptiveFieldSize = 991;
constexpr int32_t kReceptiveFieldShift = 270;
constexpr int32_t kNumSpeakers = 3;
constexpr int32_t kPowersetMaxClasses = 2;

// Scribe owns the process-wide ACL shutdown. Model destruction must not call
// aclFinalize() or reset the device because Whisper CANN and WeSpeaker may be
// using the same runtime concurrently.
void EnsureAclInitialized() {
  static std::once_flag once;
  static aclError result = ACL_SUCCESS;
  std::call_once(once, []() { result = aclInit(nullptr); });
  if (result != ACL_SUCCESS && result != ACL_ERROR_REPEAT_INITIALIZE) {
    SHERPA_ONNX_ASCEND_CHECK(result, "Failed to call aclInit");
  }
}

}  // namespace

class OfflineSpeakerSegmentationPyannoteModelAscend::Impl {
 public:
  explicit Impl(const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config) {
    EnsureAclInitialized();

    aclError ret = aclrtSetDevice(config_.device);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d",
        config_.device);

    context_ = std::make_unique<AclContext>(config_.device);
    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    model_ = std::make_unique<AclModel>(config_.pyannote.model);
    ValidateModel();

    input_bytes_ = static_cast<size_t>(model_batch_size_ * kInputChannels *
                                       kWindowSize) *
                   sizeof(float);
    output_bytes_ =
        static_cast<size_t>(model_batch_size_ * kNumFrames * kNumClasses) *
        sizeof(float);
    input_ = std::make_unique<AclDevicePtr>(input_bytes_);
    output_ = std::make_unique<AclDevicePtr>(output_bytes_);
    padded_input_.resize(input_bytes_ / sizeof(float));
    InitModelMetaData();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Ascend Pyannote segmentation OM:\n%s",
                       model_->GetInfo().c_str());
    }
  }

  Ort::Value Forward(Ort::Value x) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto tensor_info = x.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = tensor_info.GetShape();
    if (tensor_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] < kMinBatchSize ||
        shape[0] > model_batch_size_ ||
        shape[1] != kInputChannels || shape[2] != kWindowSize) {
      SHERPA_ONNX_LOGE(
          "Ascend Pyannote segmentation currently expects float32 input "
          "[B,1,160000], 1 <= B <= model batch %lld. Given shape: "
          "[%lld,%lld,%lld]",
          static_cast<long long>(model_batch_size_),
          shape.size() > 0 ? static_cast<long long>(shape[0]) : -1LL,
          shape.size() > 1 ? static_cast<long long>(shape[1]) : -1LL,
          shape.size() > 2 ? static_cast<long long>(shape[2]) : -1LL);
      SHERPA_ONNX_EXIT(-1);
    }

    aclError ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    const int64_t actual_batch_size = shape[0];
    const size_t actual_input_bytes =
        static_cast<size_t>(actual_batch_size * kInputChannels * kWindowSize) *
        sizeof(float);
    const float *input_data = x.GetTensorData<float>();
    if (actual_batch_size != model_batch_size_) {
      std::copy(input_data, input_data + actual_input_bytes / sizeof(float),
                padded_input_.begin());
      const size_t samples_per_item = kInputChannels * kWindowSize;
      const float *last_item =
          input_data + (actual_batch_size - 1) * samples_per_item;
      for (int64_t i = actual_batch_size; i != model_batch_size_; ++i) {
        std::copy(last_item, last_item + samples_per_item,
                  padded_input_.begin() + i * samples_per_item);
      }
      input_data = padded_input_.data();
    }
    ret = aclrtMemcpy(*input_, input_->Size(), input_data, input_bytes_,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to copy Pyannote segmentation input to NPU");

    AclMdlDataset input_dataset;
    AclDataBuffer input_buffer(*input_, input_->Size());
    input_dataset.AddBuffer(input_buffer);
    const std::array<int64_t, 3> model_input_shape{
        model_batch_size_, kInputChannels, kWindowSize};
    AclTensorDesc input_desc(ACL_FLOAT, model_input_shape.size(),
                             model_input_shape.data(),
                             ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(input_desc, 0);

    AclMdlDataset output_dataset;
    AclDataBuffer output_buffer(*output_, output_->Size());
    output_dataset.AddBuffer(output_buffer);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to execute Pyannote segmentation OM");

    std::array<int64_t, 3> output_shape{actual_batch_size, kNumFrames,
                                        kNumClasses};
    Ort::Value ans = Ort::Value::CreateTensor<float>(
        allocator_, output_shape.data(), output_shape.size());
    const size_t actual_output_bytes =
        static_cast<size_t>(actual_batch_size * kNumFrames * kNumClasses) *
        sizeof(float);
    ret = aclrtMemcpy(ans.GetTensorMutableData<float>(), actual_output_bytes,
                      *output_, actual_output_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to copy Pyannote segmentation output from NPU");

    return ans;
  }

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

 private:
  void ValidateModel() {
    const auto &input_shapes = model_->GetInputShapes();
    const auto &output_shapes = model_->GetOutputShapes();
    const auto &input_data_types = model_->GetInputDataTypes();
    const auto &output_data_types = model_->GetOutputDataTypes();
    const bool valid_input_shape =
        input_shapes.size() == 1 && input_shapes[0].size() == 3 &&
        input_shapes[0][0] >= kMinBatchSize &&
        input_shapes[0][0] <= kMaxBatchSize &&
        input_shapes[0][1] == kInputChannels &&
        input_shapes[0][2] == kWindowSize;
    const bool valid_output_shape =
        output_shapes.size() == 1 && output_shapes[0].size() == 3 &&
        valid_input_shape && output_shapes[0][0] == input_shapes[0][0] &&
        output_shapes[0][1] == kNumFrames &&
        output_shapes[0][2] == kNumClasses;
    if (input_shapes.size() != 1 || output_shapes.size() != 1 ||
        input_data_types.size() != 1 || output_data_types.size() != 1 ||
        input_data_types[0] != ACL_FLOAT ||
        output_data_types[0] != ACL_FLOAT ||
        !valid_input_shape || !valid_output_shape) {
      SHERPA_ONNX_LOGE(
          "Unsupported Ascend Pyannote segmentation model. Expected one "
          "float32 [B,1,160000] input and one float32 [B,589,7] output, "
          "where 1 <= B <= 16.");
      SHERPA_ONNX_EXIT(-1);
    }
    model_batch_size_ = input_shapes[0][0];
  }

  void InitModelMetaData() {
    meta_data_.sample_rate = kSampleRate;
    meta_data_.window_size = kWindowSize;
    meta_data_.receptive_field_size = kReceptiveFieldSize;
    meta_data_.receptive_field_shift = kReceptiveFieldShift;
    meta_data_.num_speakers = kNumSpeakers;
    meta_data_.powerset_max_classes = kPowersetMaxClasses;
    meta_data_.num_classes = kNumClasses;

    const double window_shift =
        static_cast<double>(config_.pyannote.window_shift_ratio) *
        meta_data_.window_size;
    if (std::isnan(window_shift) || window_shift < 1) {
      meta_data_.window_shift = 1;
    } else if (window_shift > meta_data_.window_size) {
      meta_data_.window_shift = meta_data_.window_size;
    } else {
      meta_data_.window_shift = static_cast<int32_t>(window_shift);
    }
  }

 private:
  std::mutex mutex_;
  OfflineSpeakerSegmentationModelConfig config_;
  std::unique_ptr<AclContext> context_;
  std::unique_ptr<AclModel> model_;
  std::unique_ptr<AclDevicePtr> input_;
  std::unique_ptr<AclDevicePtr> output_;
  int64_t model_batch_size_ = 0;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  std::vector<float> padded_input_;
  Ort::AllocatorWithDefaultOptions allocator_;
  OfflineSpeakerSegmentationPyannoteModelMetaData meta_data_;
};

OfflineSpeakerSegmentationPyannoteModelAscend::
    OfflineSpeakerSegmentationPyannoteModelAscend(
        const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineSpeakerSegmentationPyannoteModelAscend::
    ~OfflineSpeakerSegmentationPyannoteModelAscend() = default;

const OfflineSpeakerSegmentationPyannoteModelMetaData &
OfflineSpeakerSegmentationPyannoteModelAscend::GetModelMetaData() const {
  return impl_->GetModelMetaData();
}

Ort::Value OfflineSpeakerSegmentationPyannoteModelAscend::Forward(
    Ort::Value x) const {
  return impl_->Forward(std::move(x));
}

}  // namespace sherpa_onnx
