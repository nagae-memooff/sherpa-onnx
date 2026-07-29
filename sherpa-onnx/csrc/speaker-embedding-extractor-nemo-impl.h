// sherpa-onnx/csrc/speaker-embedding-extractor-nemo-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorNeMoImpl : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorNeMoImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorNeMoImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.feature_dim = meta_data.feat_dim;
    feat_config.normalize_samples = true;
    feat_config.snip_edges = true;
    feat_config.frame_shift_ms = meta_data.window_stride_ms;
    feat_config.frame_length_ms = meta_data.window_size_ms;
    feat_config.low_freq = 0;
    feat_config.is_librosa = true;
    feat_config.remove_dc_offset = false;
    feat_config.window_type = meta_data.window_type;

    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  int32_t NumFramesForSamples(int64_t num_samples) const override {
    const auto &meta_data = model_.GetMetaData();
    int64_t frame_shift = static_cast<int64_t>(
        meta_data.sample_rate * 0.001f * meta_data.window_stride_ms);
    int64_t frame_length = static_cast<int64_t>(
        meta_data.sample_rate * 0.001f * meta_data.window_size_ms);
    if (num_samples < frame_length) {
      return 0;
    }
    return static_cast<int32_t>(
        1 + (num_samples - frame_length) / frame_shift);
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
      if (meta_data.feature_normalize_type == "per_feature") {
        RunProfiled<EnableProfiling>(
            EnableProfiling ? &profiling->normalize_seconds : nullptr, [&]() {
              NormalizePerFeature(features.data(), num_frames, feat_dim);
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

    int64_t x_lens = num_frames;
    auto tensors = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->prepare_tensor_seconds : nullptr, [&]() {
          if (num_frames % 16 != 0) {
            int32_t pad = 16 - num_frames % 16;
            features.resize((num_frames + pad) * feat_dim);
          }

          auto memory_info =
              Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
          std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
          Ort::Value x =
              Ort::Value::CreateTensor(memory_info, features.data(),
                                       features.size(), x_shape.data(),
                                       x_shape.size());
          x = Transpose12(model_.Allocator(), &x);

          std::array<int64_t, 1> x_lens_shape{1};
          Ort::Value x_lens_tensor = Ort::Value::CreateTensor(
              memory_info, &x_lens, 1, x_lens_shape.data(),
              x_lens_shape.size());
          return std::make_pair(std::move(x), std::move(x_lens_tensor));
        });

    Ort::Value embedding = RunProfiled<EnableProfiling>(
        EnableProfiling ? &profiling->inference_seconds : nullptr, [&]() {
          return model_.Compute(std::move(tensors.first),
                                std::move(tensors.second));
        });
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
        if (meta_data.feature_normalize_type == "per_feature") {
          RunProfiled<EnableProfiling>(
              EnableProfiling ? &p->normalize_seconds : nullptr, [&]() {
                NormalizePerFeature(features.data(), num_frames, feat_dim);
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

    std::vector<int64_t> lengths(streams.size(), num_frames);
    auto tensors = RunProfiled<EnableProfiling>(
        EnableProfiling ? &batch_profiling->prepare_tensor_seconds : nullptr,
        [&]() {
          auto memory_info =
              Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
          std::array<int64_t, 3> x_shape = {
              static_cast<int64_t>(streams.size()), num_frames, feat_dim};
          Ort::Value x = Ort::Value::CreateTensor(
              memory_info, batch_features.data(), batch_features.size(),
              x_shape.data(), x_shape.size());
          x = Transpose12(model_.Allocator(), &x);

          std::array<int64_t, 1> lengths_shape = {
              static_cast<int64_t>(streams.size())};
          Ort::Value lengths_tensor = Ort::Value::CreateTensor(
              memory_info, lengths.data(), lengths.size(), lengths_shape.data(),
              lengths_shape.size());
          return std::make_pair(std::move(x), std::move(lengths_tensor));
        });

    Ort::Value embedding = RunProfiled<EnableProfiling>(
        EnableProfiling ? &batch_profiling->inference_seconds : nullptr,
        [&]() {
          return model_.Compute(std::move(tensors.first),
                                std::move(tensors.second));
        });

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
  void NormalizePerFeature(float *p, int32_t num_frames,
                           int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    auto EX = m.colwise().mean();
    auto EX2 = m.array().pow(2).colwise().sum() / num_frames;
    auto variance = (EX2 - EX.array().pow(2)).max(1e-5f);

    auto stddev = variance.array().sqrt();

    m = (m.rowwise() - EX).array().rowwise() / (stddev.array() + 1e-5f);
  }

 private:
  SpeakerEmbeddingExtractorNeMoModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
