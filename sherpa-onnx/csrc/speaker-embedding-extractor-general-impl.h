// sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
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

  int32_t NumFramesForSamples(int64_t num_samples) const override {
    int64_t frame_shift = model_.GetMetaData().sample_rate / 100;
    return static_cast<int32_t>((num_samples + frame_shift / 2) / frame_shift);
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    return ComputeImpl<false>(s, nullptr);
  }

  std::vector<float> ComputeWithProfiling(
      OnlineStream *s,
      SpeakerEmbeddingExtractorProfilingInfo *profiling) const override {
    return ComputeImpl<true>(s, profiling);
  }

  std::vector<std::vector<float>> ComputeBatch(
      const std::vector<OnlineStream *> &streams) const override {
    return ComputeBatchImpl<false>(streams, nullptr, nullptr);
  }

  std::vector<std::vector<float>> ComputeBatchWithProfiling(
      const std::vector<OnlineStream *> &streams,
      std::vector<SpeakerEmbeddingExtractorProfilingInfo> *stream_profiling,
      SpeakerEmbeddingExtractorBatchProfilingInfo *batch_profiling)
      const override {
    return ComputeBatchImpl<true>(streams, stream_profiling, batch_profiling);
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

  template <bool EnableProfiling>
  std::vector<std::vector<float>> ComputeBatchImpl(
      const std::vector<OnlineStream *> &streams,
      std::vector<SpeakerEmbeddingExtractorProfilingInfo> *stream_profiling,
      SpeakerEmbeddingExtractorBatchProfilingInfo *batch_profiling) const {
    if (streams.empty()) {
      return {};
    }

    int32_t num_frames =
        streams[0]->NumFramesReady() - streams[0]->GetNumProcessedFrames();
    if (num_frames <= 0) {
      throw std::runtime_error(
          "Speaker embedding batch contains an empty stream");
    }
    for (OnlineStream *s : streams) {
      int32_t n = s->NumFramesReady() - s->GetNumProcessedFrames();
      if (n != num_frames) {
        throw std::runtime_error(
            "Speaker embedding batch requires equal frame counts");
      }
    }

    if constexpr (EnableProfiling) {
      stream_profiling->resize(streams.size());
      batch_profiling->batch_size = static_cast<int32_t>(streams.size());
    }

    std::vector<float> batch_features;
    int32_t feat_dim = 0;
    for (size_t i = 0; i != streams.size(); ++i) {
      OnlineStream *s = streams[i];
      auto *p = EnableProfiling ? &(*stream_profiling)[i] : nullptr;
      std::vector<float> features = RunProfiled<EnableProfiling>(
          EnableProfiling ? &p->get_frames_seconds : nullptr, [&]() {
            return s->GetFrames(s->GetNumProcessedFrames(), num_frames);
          });
      s->GetNumProcessedFrames() += num_frames;

      int32_t current_feat_dim =
          static_cast<int32_t>(features.size()) / num_frames;
      if (i == 0) {
        feat_dim = current_feat_dim;
        batch_features.resize(streams.size() * features.size());
      } else if (current_feat_dim != feat_dim) {
        throw std::runtime_error(
            "Speaker embedding batch requires equal feature dimensions");
      }
      if constexpr (EnableProfiling) {
        p->num_frames = num_frames;
        p->feature_dim = feat_dim;
      }

      const auto &meta_data = model_.GetMetaData();
      if (!meta_data.feature_normalize_type.empty()) {
        if (meta_data.feature_normalize_type == "global-mean") {
          RunProfiled<EnableProfiling>(
              EnableProfiling ? &p->normalize_seconds : nullptr, [&]() {
                SubtractGlobalMean(features.data(), num_frames, feat_dim);
                return 0;
              });
        } else {
          throw std::runtime_error(
              "Unsupported speaker embedding feature normalization");
        }
      }

      std::copy(features.begin(), features.end(),
                batch_features.begin() + i * features.size());
    }

    Ort::Value x = RunProfiled<EnableProfiling>(
        EnableProfiling ? &batch_profiling->prepare_tensor_seconds : nullptr,
        [&]() {
          auto memory_info =
              Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
          std::array<int64_t, 3> shape = {
              static_cast<int64_t>(streams.size()), num_frames, feat_dim};
          return Ort::Value::CreateTensor(
              memory_info, batch_features.data(), batch_features.size(),
              shape.data(), shape.size());
        });

    Ort::Value embedding = RunProfiled<EnableProfiling>(
        EnableProfiling ? &batch_profiling->inference_seconds : nullptr,
        [&]() { return model_.Compute(std::move(x)); });

    return RunProfiled<EnableProfiling>(
        EnableProfiling ? &batch_profiling->output_copy_seconds : nullptr,
        [&]() {
          std::vector<int64_t> shape =
              embedding.GetTensorTypeAndShapeInfo().GetShape();
          if (shape.size() != 2 ||
              shape[0] != static_cast<int64_t>(streams.size()) ||
              shape[1] <= 0) {
            throw std::runtime_error(
                "Unexpected speaker embedding batch output shape");
          }
          int32_t dim = static_cast<int32_t>(shape[1]);
          const float *data = embedding.GetTensorData<float>();
          std::vector<std::vector<float>> ans(streams.size());
          for (size_t i = 0; i != streams.size(); ++i) {
            ans[i].assign(data + i * dim, data + (i + 1) * dim);
          }
          return ans;
        });
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
