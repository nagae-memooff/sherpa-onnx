// sherpa-onnx/csrc/ascend/offline-speaker-segmentation-pyannote-model-ascend.cc

#include "sherpa-onnx/csrc/ascend/offline-speaker-segmentation-pyannote-model-ascend.h"

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

constexpr int32_t kDeviceId = 0;
constexpr int64_t kBatchSize = 1;
constexpr int64_t kInputChannels = 1;
constexpr int64_t kWindowSize = 160000;
constexpr int64_t kNumFrames = 589;
constexpr int64_t kNumClasses = 7;

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kReceptiveFieldSize = 991;
constexpr int32_t kReceptiveFieldShift = 270;
constexpr int32_t kNumSpeakers = 3;
constexpr int32_t kPowersetMaxClasses = 2;

constexpr size_t kInputBytes =
    kBatchSize * kInputChannels * kWindowSize * sizeof(float);
constexpr size_t kOutputBytes =
    kBatchSize * kNumFrames * kNumClasses * sizeof(float);

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

    aclError ret = aclrtSetDevice(kDeviceId);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d", kDeviceId);

    context_ = std::make_unique<AclContext>(kDeviceId);
    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    model_ = std::make_unique<AclModel>(config_.pyannote.model);
    ValidateModel();

    input_ = std::make_unique<AclDevicePtr>(kInputBytes);
    output_ = std::make_unique<AclDevicePtr>(kOutputBytes);
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
        shape.size() != 3 || shape[0] != kBatchSize ||
        shape[1] != kInputChannels || shape[2] != kWindowSize) {
      SHERPA_ONNX_LOGE(
          "Ascend Pyannote segmentation currently expects float32 input "
          "[1,1,160000] and requires segmentation_batch_size=1. Given "
          "shape: [%lld,%lld,%lld]",
          shape.size() > 0 ? static_cast<long long>(shape[0]) : -1LL,
          shape.size() > 1 ? static_cast<long long>(shape[1]) : -1LL,
          shape.size() > 2 ? static_cast<long long>(shape[2]) : -1LL);
      SHERPA_ONNX_EXIT(-1);
    }

    aclError ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    ret = aclrtMemcpy(*input_, input_->Size(), x.GetTensorData<float>(),
                      kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to copy Pyannote segmentation input to NPU");

    AclMdlDataset input_dataset;
    AclDataBuffer input_buffer(*input_, input_->Size());
    input_dataset.AddBuffer(input_buffer);
    AclTensorDesc input_desc(ACL_FLOAT, shape.size(), shape.data(),
                             ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(input_desc, 0);

    AclMdlDataset output_dataset;
    AclDataBuffer output_buffer(*output_, output_->Size());
    output_dataset.AddBuffer(output_buffer);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to execute Pyannote segmentation OM");

    std::array<int64_t, 3> output_shape{kBatchSize, kNumFrames, kNumClasses};
    Ort::Value ans = Ort::Value::CreateTensor<float>(
        allocator_, output_shape.data(), output_shape.size());
    ret = aclrtMemcpy(ans.GetTensorMutableData<float>(), kOutputBytes,
                      *output_, output_->Size(), ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to copy Pyannote segmentation output from NPU");

    return ans;
  }

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

 private:
  void ValidateModel() const {
    const auto &input_shapes = model_->GetInputShapes();
    const auto &output_shapes = model_->GetOutputShapes();
    const auto &input_data_types = model_->GetInputDataTypes();
    const auto &output_data_types = model_->GetOutputDataTypes();
    const std::vector<int64_t> expected_input{kBatchSize, kInputChannels,
                                              kWindowSize};
    const std::vector<int64_t> expected_output{kBatchSize, kNumFrames,
                                               kNumClasses};
    if (input_shapes.size() != 1 || output_shapes.size() != 1 ||
        input_data_types.size() != 1 || output_data_types.size() != 1 ||
        input_data_types[0] != ACL_FLOAT ||
        output_data_types[0] != ACL_FLOAT ||
        input_shapes[0] != expected_input ||
        output_shapes[0] != expected_output) {
      SHERPA_ONNX_LOGE(
          "Unsupported Ascend Pyannote segmentation model. Expected one "
          "float32 [1,1,160000] input and one float32 [1,589,7] output.");
      SHERPA_ONNX_EXIT(-1);
    }
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
