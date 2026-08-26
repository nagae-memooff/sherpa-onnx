// sherpa-onnx/csrc/ascend/speaker-embedding-extractor-model-ascend.cc

#include "sherpa-onnx/csrc/ascend/speaker-embedding-extractor-model-ascend.h"

#include <array>
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
constexpr int64_t kFeatureDim = 80;
constexpr int64_t kEmbeddingDim = 256;
constexpr int64_t kMinFrames = 1;
constexpr int64_t kMaxFrames = 1000;

// ggml CANN and sherpa-onnx share the process-wide ACL runtime. ggml may have
// initialized ACL before this model is constructed, so repeated aclInit is
// intentionally accepted. Finalization is left to process shutdown; calling
// aclFinalize when the embedding model is released could invalidate a Whisper
// inference that is still using CANN on another thread.
void EnsureAclInitialized() {
  static std::once_flag once;
  std::call_once(once, []() { (void)aclInit(nullptr); });
}

}  // namespace

class SpeakerEmbeddingExtractorModelAscend::Impl {
 public:
  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config) {
    EnsureAclInitialized();

    aclError ret = aclrtSetDevice(kDeviceId);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d", kDeviceId);

    context_ = std::make_unique<AclContext>(kDeviceId);
    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    model_ = std::make_unique<AclModel>(config_.model);
    ValidateModel();

    input_ = std::make_unique<AclDevicePtr>(
        kBatchSize * kMaxFrames * kFeatureDim * sizeof(float));
    output_ = std::make_unique<AclDevicePtr>(
        kBatchSize * kEmbeddingDim * sizeof(float));

    meta_data_.output_dim = kEmbeddingDim;
    meta_data_.sample_rate = 16000;
    meta_data_.normalize_samples = 0;
    meta_data_.language = "English";
    meta_data_.feature_normalize_type = "global-mean";

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Ascend WeSpeaker OM:\n%s",
                       model_->GetInfo().c_str());
    }
  }

  Ort::Value Compute(Ort::Value x) {
    std::lock_guard<std::mutex> lock(mutex_);

    aclError ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

    const std::vector<int64_t> shape =
        x.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != kBatchSize ||
        shape[2] != kFeatureDim || shape[1] < kMinFrames ||
        shape[1] > kMaxFrames) {
      SHERPA_ONNX_LOGE(
          "Ascend WeSpeaker expects float32 input [1,T,80], 1 <= T <= "
          "1000. Given shape: [%lld,%lld,%lld]",
          shape.size() > 0 ? static_cast<long long>(shape[0]) : -1LL,
          shape.size() > 1 ? static_cast<long long>(shape[1]) : -1LL,
          shape.size() > 2 ? static_cast<long long>(shape[2]) : -1LL);
      SHERPA_ONNX_EXIT(-1);
    }

    const size_t input_bytes = static_cast<size_t>(shape[0] * shape[1] *
                                                   shape[2]) *
                               sizeof(float);
    ret = aclrtMemcpy(*input_, input_->Size(), x.GetTensorData<float>(),
                      input_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to copy WeSpeaker input to NPU");

    AclMdlDataset input_dataset;
    AclDataBuffer input_buffer(*input_, input_bytes);
    input_dataset.AddBuffer(input_buffer);
    AclTensorDesc input_desc(ACL_FLOAT, shape.size(), shape.data(),
                             ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(input_desc, 0);

    AclMdlDataset output_dataset;
    AclDataBuffer output_buffer(*output_, output_->Size());
    output_dataset.AddBuffer(output_buffer);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to execute WeSpeaker OM");

    std::array<int64_t, 2> output_shape{kBatchSize, kEmbeddingDim};
    Ort::Value ans = Ort::Value::CreateTensor<float>(
        allocator_, output_shape.data(), output_shape.size());
    ret = aclrtMemcpy(ans.GetTensorMutableData<float>(), output_->Size(),
                      *output_, output_->Size(), ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to copy WeSpeaker output from NPU");

    return ans;
  }

  const SpeakerEmbeddingExtractorModelMetaData &GetMetaData() const {
    return meta_data_;
  }

 private:
  void ValidateModel() const {
    const auto &input_shapes = model_->GetInputShapes();
    const auto &output_shapes = model_->GetOutputShapes();
    if (input_shapes.size() != 1 || output_shapes.size() != 1 ||
        input_shapes[0].size() != 3 || output_shapes[0].size() != 2 ||
        input_shapes[0].back() != kFeatureDim ||
        output_shapes[0].back() != kEmbeddingDim) {
      SHERPA_ONNX_LOGE(
          "Unsupported Ascend speaker embedding model. Expected one "
          "[1,T,80] input and one [1,256] output.");
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  std::mutex mutex_;
  SpeakerEmbeddingExtractorConfig config_;
  std::unique_ptr<AclContext> context_;
  std::unique_ptr<AclModel> model_;
  std::unique_ptr<AclDevicePtr> input_;
  std::unique_ptr<AclDevicePtr> output_;
  Ort::AllocatorWithDefaultOptions allocator_;
  SpeakerEmbeddingExtractorModelMetaData meta_data_;
};

SpeakerEmbeddingExtractorModelAscend::SpeakerEmbeddingExtractorModelAscend(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerEmbeddingExtractorModelAscend::~SpeakerEmbeddingExtractorModelAscend() =
    default;

const SpeakerEmbeddingExtractorModelMetaData &
SpeakerEmbeddingExtractorModelAscend::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value SpeakerEmbeddingExtractorModelAscend::Compute(Ort::Value x) const {
  return impl_->Compute(std::move(x));
}

}  // namespace sherpa_onnx
