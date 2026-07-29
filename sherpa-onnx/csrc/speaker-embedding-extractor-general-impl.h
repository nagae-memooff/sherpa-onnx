// sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorGeneralImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorGeneralImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorGeneralImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.normalize_samples = meta_data.normalize_samples;

    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    return ComputeImpl<false>(s, nullptr);
  }

  std::vector<float> ComputeWithProfiling(
      OnlineStream *s,
      SpeakerEmbeddingExtractorProfilingInfo *profiling) const override {
    return ComputeImpl<true>(s, profiling);
  }

 private:
  template <bool EnableProfiling, typename Func>
  static auto RunProfiled(double *seconds, Func &&func) -> decltype(func()) {
    if constexpr (EnableProfiling) {
      auto start = std::chrono::steady_clock::now();
      auto ans = func();
      *seconds += std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
      return ans;
    } else {
      return func();
    }
  }

  template <bool EnableProfiling>
  std::vector<float> ComputeImpl(
      OnlineStream *s,
      SpeakerEmbeddingExtractorProfilingInfo *profiling) const {
    int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
    if (num_frames <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %{public}d",
          num_frames);
#else
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %d",
          num_frames);
#endif
      return {};
    }

    std::vector<float> features = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->get_frames_seconds : nullptr, [&]() {
          return s->GetFrames(s->GetNumProcessedFrames(), num_frames);
        });

    s->GetNumProcessedFrames() += num_frames;

    int32_t feat_dim = features.size() / num_frames;
    if constexpr (EnableProfiling) {
      profiling->num_frames = num_frames;
      profiling->feature_dim = feat_dim;
    }

    const auto &meta_data = model_.GetMetaData();
    if (!meta_data.feature_normalize_type.empty()) {
      if (meta_data.feature_normalize_type == "global-mean") {
        RunProfiled<EnableProfiling>(
            EnableProfiling ? &profiling->normalize_seconds : nullptr, [&]() {
              SubtractGlobalMean(features.data(), num_frames, feat_dim);
              return 0;
            });
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                         meta_data.feature_normalize_type.c_str());
#else
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                         meta_data.feature_normalize_type.c_str());
#endif
        SHERPA_ONNX_EXIT(-1);
      }
    }

    Ort::Value x = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->prepare_tensor_seconds : nullptr, [&]() {
          auto memory_info =
              Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
          std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
          return Ort::Value::CreateTensor(memory_info, features.data(),
                                          features.size(), x_shape.data(),
                                          x_shape.size());
        });
    Ort::Value embedding = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->inference_seconds : nullptr,
        [&]() { return model_.Compute(std::move(x)); });
    std::vector<float> ans = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->output_copy_seconds : nullptr, [&]() {
          std::vector<int64_t> embedding_shape =
              embedding.GetTensorTypeAndShapeInfo().GetShape();
          std::vector<float> output(embedding_shape[1]);
          std::copy(embedding.GetTensorData<float>(),
                    embedding.GetTensorData<float>() + output.size(),
                    output.begin());
          return output;
        });

    return ans;
  }

 private:
  void SubtractGlobalMean(float *p, int32_t num_frames,
                          int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    m = m.rowwise() - m.colwise().mean();
  }

 private:
  SpeakerEmbeddingExtractorModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
