// sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/fast-clustering.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/csrc/timer.h"

namespace sherpa_onnx {

namespace {  // NOLINT

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L41
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L47
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};

class StageResourceGuard {
 public:
  explicit StageResourceGuard(std::function<void()> release)
      : release_(std::move(release)) {}

  StageResourceGuard(const StageResourceGuard &) = delete;
  StageResourceGuard &operator=(const StageResourceGuard &) = delete;

  ~StageResourceGuard() { ReleaseNow(); }

  void ReleaseNow() {
    if (release_) {
      auto release = std::move(release_);
      release();
    }
  }

 private:
  std::function<void()> release_;
};
}  // namespace

using Matrix2D = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;  // NOLINT

using Matrix2DInt32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>;  // NOLINT

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using Int32RowVector = Eigen::Matrix<int32_t, 1, Eigen::Dynamic>;

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(
            std::make_unique<OfflineSpeakerSegmentationPyannoteModel>(
                config_.segmentation)),
        segmentation_meta_data_(segmentation_model_->GetModelMetaData()),
        embedding_factory_([embedding_config = config_.embedding]() {
          return std::make_unique<SpeakerEmbeddingExtractor>(embedding_config);
        }),
        embedding_extractor_(
            config_.release_model_resources_after_use
                ? nullptr
                : embedding_factory_()),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  template <typename Manager>
  OfflineSpeakerDiarizationPyannoteImpl(
      Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(
            std::make_unique<OfflineSpeakerSegmentationPyannoteModel>(
                mgr, config_.segmentation)),
        segmentation_meta_data_(segmentation_model_->GetModelMetaData()),
        embedding_extractor_(
            std::make_unique<SpeakerEmbeddingExtractor>(mgr,
                                                        config_.embedding)),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  int32_t SampleRate() const override {
    return segmentation_meta_data_.sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig &config) override {
    if (!config.clustering.Validate()) {
      SHERPA_ONNX_LOGE("Invalid clustering config. Skip it");
      return;
    }
    clustering_ = std::make_unique<FastClustering>(config.clustering);
    config_.clustering = config.clustering;
  }

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    if (config_.release_model_resources_after_use &&
        process_started_.exchange(true, std::memory_order_acq_rel)) {
      throw std::runtime_error(
          "OfflineSpeakerDiarization with "
          "release_model_resources_after_use=true can be processed only once");
    }

    Timer total_timer;
    Timer segmentation_timer;
    bool stopped = false;
    StageResourceGuard segmentation_guard(
        [this]() { ReleaseSegmentationModel(); });
    std::vector<Matrix2D> segmentations =
        RunSpeakerSegmentationModel(audio, n, callback, callback_arg, &stopped);
    segmentation_guard.ReleaseNow();
    LogStageTime("segmentation", segmentation_timer.Elapsed());
    // segmentations[i] is for chunk_i
    // Each matrix is of shape (num_frames, num_powerset_classes)
    if (stopped) {
      return CreateStoppedResult();
    }

    if (segmentations.empty()) {
      return {};
    }

    std::vector<Matrix2DInt32> labels;
    labels.reserve(segmentations.size());

    for (const auto &m : segmentations) {
      labels.push_back(ToMultiLabel(m));
    }

    segmentations.clear();

    if (labels.size() == 1) {
      return HandleOneChunkSpecialCase(labels[0], n);
    }

    // labels[i] is a 0-1 matrix of shape (num_frames, num_speakers)

    // speaker count per frame
    Int32RowVector speakers_per_frame = ComputeSpeakersPerFrame(labels);

    if (speakers_per_frame.maxCoeff() == 0) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }

    auto chunk_speaker_samples_list_pair = GetChunkSpeakerSampleIndexes(labels);

    if (chunk_speaker_samples_list_pair.second.empty()) {
      SHERPA_ONNX_LOGE("No valid speaker embeddings found in the audio samples");
      return {};
    }

    // The embedding model may output NaN. valid_indexes contains indexes
    // in chunk_speaker_samples_list_pair.second that don't lead to
    // NaN embeddings.
    std::vector<int32_t> valid_indexes;
    valid_indexes.reserve(chunk_speaker_samples_list_pair.second.size());

    EnsureEmbeddingExtractor();
    StageResourceGuard embedding_guard(
        [this]() { ReleaseEmbeddingExtractor(); });
    Timer embedding_timer;
    Matrix2D embeddings =
        ComputeEmbeddings(audio, n, chunk_speaker_samples_list_pair.second,
                          &valid_indexes, callback, callback_arg, &stopped);
    embedding_guard.ReleaseNow();
    LogStageTime("embedding", embedding_timer.Elapsed());

    if (stopped) {
      return CreateStoppedResult();
    }

    if (valid_indexes.size() != chunk_speaker_samples_list_pair.second.size()) {
      std::vector<Int32Pair> chunk_speaker_pair;

      chunk_speaker_pair.reserve(valid_indexes.size());
      for (auto i : valid_indexes) {
        if (i < 0 ||
            i >= static_cast<int32_t>(
                     chunk_speaker_samples_list_pair.first.size())) {
          SHERPA_ONNX_LOGE("valid_indexes: index %d out of range [0, %d)",
                           i,
                           static_cast<int32_t>(
                               chunk_speaker_samples_list_pair.first.size()));
          continue;
        }
        chunk_speaker_pair.push_back(chunk_speaker_samples_list_pair.first[i]);
      }

      chunk_speaker_samples_list_pair.first = std::move(chunk_speaker_pair);
    }
    std::vector<std::vector<Int32Pair>>().swap(
        chunk_speaker_samples_list_pair.second);

    if (embeddings.rows() == 0) {
      SHERPA_ONNX_LOGE("No valid speaker embeddings found in the audio samples");
      return {};
    }

    if (ShouldStop(callback, valid_indexes.size(), embeddings.rows(),
                   callback_arg)) {
      return CreateStoppedResult();
    }

    Timer clustering_timer;
    std::vector<int32_t> cluster_labels = clustering_->Cluster(
        &embeddings(0, 0), embeddings.rows(), embeddings.cols());
    LogStageTime("clustering", clustering_timer.Elapsed());

    if (ShouldStop(callback, valid_indexes.size(), embeddings.rows(),
                   callback_arg)) {
      return CreateStoppedResult();
    }
    embeddings.resize(0, 0);
    std::vector<int32_t>().swap(valid_indexes);

    if (cluster_labels.empty()) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }

    // 如果只得到 1 个簇，直接构造结果返回，避免后续重标记/合并路径
    {
      int32_t max_cluster_index =
          *std::max_element(cluster_labels.begin(), cluster_labels.end());
      if (max_cluster_index <= 0) {
        // speakers_per_frame 已反映每帧说话人数，强制单簇输出到 speaker 0
        int32_t num_frames = speakers_per_frame.size();
        Matrix2DInt32 final_labels(num_frames, 1);
        final_labels.setZero();
        for (int32_t i = 0; i != num_frames; ++i) {
          if (speakers_per_frame[i] > 0) {
            final_labels(i, 0) = 1;
          }
        }
        return ComputeResult(final_labels);
      }
    }

    int32_t max_cluster_index =
        *std::max_element(cluster_labels.begin(), cluster_labels.end());

    auto chunk_speaker_to_cluster = ConvertChunkSpeakerToCluster(
        chunk_speaker_samples_list_pair.first, cluster_labels);

    auto new_labels =
        ReLabel(labels, max_cluster_index, chunk_speaker_to_cluster);

    Matrix2DInt32 speaker_count = ComputeSpeakerCount(new_labels, n);

    Matrix2DInt32 final_labels =
        FinalizeLabels(speaker_count, speakers_per_frame);

    auto result = ComputeResult(final_labels);

    LogTotalTime(total_timer.Elapsed(), n);
    return result;
  }

 private:
  void LogStageTime(const char *stage, double seconds) const {
    if (config_.segmentation.debug || config_.embedding.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("OfflineSpeakerDiarization: %{public}s %{public}.3f s",
                       stage, seconds);
#else
      SHERPA_ONNX_LOGE("OfflineSpeakerDiarization: %s %.3f s", stage, seconds);
#endif
    }
  }

  void LogTotalTime(double seconds, int32_t num_samples) const {
    if (!(config_.segmentation.debug || config_.embedding.debug) ||
        num_samples <= 0) {
      return;
    }

    double audio_duration = static_cast<double>(num_samples) / SampleRate();
#if __OHOS__
    SHERPA_ONNX_LOGE(
        "OfflineSpeakerDiarization: total %{public}.3f s, audio %{public}.3f s, "
        "RTF %{public}.3f",
        seconds, audio_duration, seconds / audio_duration);
#else
    SHERPA_ONNX_LOGE(
        "OfflineSpeakerDiarization: total %.3f s, audio %.3f s, RTF %.3f",
        seconds, audio_duration, seconds / audio_duration);
#endif
  }

  void EnsureEmbeddingExtractor() const {
    if (!embedding_extractor_ && embedding_factory_) {
      embedding_extractor_ = embedding_factory_();
    }
    if (!embedding_extractor_) {
      throw std::runtime_error(
          "Failed to initialize speaker embedding extractor");
    }
  }

  void ReleaseSegmentationModel() const {
    if (!config_.release_model_resources_after_use || !segmentation_model_) {
      return;
    }
    segmentation_model_.reset();
    if (config_.segmentation.debug || config_.embedding.debug) {
      SHERPA_ONNX_LOGE(
          "OfflineSpeakerDiarization: released segmentation model resources");
    }
  }

  void ReleaseEmbeddingExtractor() const {
    if (!config_.release_model_resources_after_use || !embedding_extractor_) {
      return;
    }
    embedding_extractor_.reset();
    if (config_.segmentation.debug || config_.embedding.debug) {
      SHERPA_ONNX_LOGE(
          "OfflineSpeakerDiarization: released embedding model resources");
    }
  }

  void Init() { InitPowersetMapping(); }

  static bool ShouldStop(OfflineSpeakerDiarizationProgressCallback callback,
                         int32_t processed, int32_t total,
                         void *callback_arg) {
    return callback && callback(processed, total, callback_arg) != 0;
  }

  static OfflineSpeakerDiarizationResult CreateStoppedResult() {
    OfflineSpeakerDiarizationResult ans;
    ans.SetStopped(true);
    return ans;
  }

  // see also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L68
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_meta_data_;
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ = Matrix2DInt32(num_classes, num_speakers);
    powerset_mapping_.setZero();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          powerset_mapping_(k, j) = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            powerset_mapping_(k, j) = 1;
            powerset_mapping_(k, m) = 1;
          }
        }
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %{public}d is currently not supported!", i);
#else
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %d is currently not supported!", i);
#endif
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  std::vector<Matrix2D> RunSpeakerSegmentationModel(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback, void *callback_arg,
      bool *stopped) const {
    std::vector<Matrix2D> ans;

    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    if (n <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "number of audio samples is %{public}d (<= 0). Please provide a "
          "positive number",
          n);
#else
      SHERPA_ONNX_LOGE(
          "number of audio samples is %d (<= 0). Please provide a positive "
          "number",
          n);
#endif
      return {};
    }

    int32_t num_chunks = 1;
    bool has_last_chunk = false;
    if (n > window_size) {
      num_chunks = (n - window_size) / window_shift + 1;
      has_last_chunk = ((n - window_size) % window_shift) > 0;
    }
    int32_t total_chunks = num_chunks + static_cast<int32_t>(has_last_chunk);

    ans.reserve(total_chunks);

    int32_t batch_size = std::min(config_.segmentation_batch_size, total_chunks);
    if (config_.segmentation.debug) {
      int32_t num_batches = (total_chunks + batch_size - 1) / batch_size;
      SHERPA_ONNX_LOGE(
          "OfflineSpeakerDiarization: segmentation chunks=%d, batch_size=%d, "
          "batches=%d",
          total_chunks, batch_size, num_batches);
    }

    for (int32_t batch_start = 0; batch_start < total_chunks;
         batch_start += batch_size) {
      int32_t current_batch =
          std::min(batch_size, total_chunks - batch_start);
      size_t batch_num_samples =
          static_cast<size_t>(current_batch) * window_size;
      std::vector<float> batch_audio(batch_num_samples);

      for (int32_t i = 0; i != current_batch; ++i) {
        int32_t chunk_index = batch_start + i;
        int64_t start = static_cast<int64_t>(chunk_index) * window_shift;
        int64_t end =
            std::min<int64_t>(start + window_size, static_cast<int64_t>(n));
        if (start < n && end > start) {
          std::copy(audio + start, audio + end,
                    batch_audio.data() + static_cast<size_t>(i) * window_size);
        }
      }

      std::vector<Matrix2D> batch_results =
          ProcessChunkBatch(batch_audio.data(), current_batch);
      if (static_cast<int32_t>(batch_results.size()) != current_batch) {
        throw std::runtime_error(
            "Speaker segmentation output batch size does not match input");
      }

      for (auto &m : batch_results) {
        ans.push_back(std::move(m));
        if (ShouldStop(callback, static_cast<int32_t>(ans.size()),
                       total_chunks, callback_arg)) {
          *stopped = true;
          return ans;
        }
      }
    }

    return ans;
  }

  std::vector<Matrix2D> ProcessChunkBatch(const float *p,
                                          int32_t batch_size) const {
    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> shape = {batch_size, 1, window_size};
    size_t num_samples = static_cast<size_t>(batch_size) * window_size;

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(p),
                                 num_samples, shape.data(), shape.size());

    Ort::Value out = segmentation_model_->Forward(std::move(x));
    std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[0] != batch_size ||
        out_shape[1] <= 0 || out_shape[2] <= 0) {
      throw std::runtime_error(
          "Unexpected speaker segmentation output tensor shape");
    }

    size_t one_output_size =
        static_cast<size_t>(out_shape[1]) * out_shape[2];
    const float *out_data = out.GetTensorData<float>();
    std::vector<Matrix2D> ans;
    ans.reserve(batch_size);
    for (int32_t i = 0; i != batch_size; ++i) {
      Matrix2D m(out_shape[1], out_shape[2]);
      const float *begin = out_data + static_cast<size_t>(i) * one_output_size;
      std::copy(begin, begin + one_output_size, &m(0, 0));
      ans.push_back(std::move(m));
    }
    return ans;
  }

  Matrix2DInt32 ToMultiLabel(const Matrix2D &m) const {
    int32_t num_rows = m.rows();
    Matrix2DInt32 ans(num_rows, powerset_mapping_.cols());

    std::ptrdiff_t col_id;

    for (int32_t i = 0; i != num_rows; ++i) {
      m.row(i).maxCoeff(&col_id);
      ans.row(i) = powerset_mapping_.row(col_id);
    }

    return ans;
  }

  // See also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/utils/diarization.py#L122
  Int32RowVector ComputeSpeakersPerFrame(
      const std::vector<Matrix2DInt32> &labels) const {
    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    FloatRowVector count(num_frames);
    FloatRowVector weight(num_frames);
    count.setZero();
    weight.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq).array() += labels[i].rowwise().sum().array().cast<float>();

      weight(seq).array() += 1;
    }

    return ((count.array() / (weight.array() + 1e-12f)) + 0.5).cast<int32_t>();
  }

  // ans.first: a list of (chunk_id, speaker_id)
  // ans.second: a list of list of (start_sample_index, end_sample_index)
  //
  // ans.first[i] corresponds to ans.second[i]
  std::pair<std::vector<Int32Pair>, std::vector<std::vector<Int32Pair>>>
  GetChunkSpeakerSampleIndexes(const std::vector<Matrix2DInt32> &labels) const {
    auto new_labels = ExcludeOverlap(labels);

    std::vector<Int32Pair> chunk_speaker_list;
    std::vector<std::vector<Int32Pair>> samples_index_list;

    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t num_speakers = meta_data.num_speakers;

    int32_t chunk_index = 0;
    for (const auto &label : new_labels) {
      Matrix2DInt32 tmp = label.transpose();
      // tmp: (num_speakers, num_frames)
      int32_t num_frames = tmp.cols();

      int32_t sample_offset = chunk_index * window_shift;

      for (int32_t speaker_index = 0; speaker_index != num_speakers;
           ++speaker_index) {
        auto d = tmp.row(speaker_index);
        if (d.sum() < 10) {
          // skip segments less than 10 frames
          continue;
        }

        Int32Pair this_chunk_speaker = {chunk_index, speaker_index};
        std::vector<Int32Pair> this_speaker_samples;

        bool is_active = false;
        int32_t start_index;

        for (int32_t k = 0; k != num_frames; ++k) {
          if (d[k] != 0) {
            if (!is_active) {
              is_active = true;
              start_index = k;
            }
          } else if (is_active) {
            is_active = false;

            int32_t start_samples =
                static_cast<float>(start_index) / num_frames * window_size +
                sample_offset;
            int32_t end_samples =
                static_cast<float>(k) / num_frames * window_size +
                sample_offset;

            this_speaker_samples.emplace_back(start_samples, end_samples);
          }
        }

        if (is_active) {
          int32_t start_samples =
              static_cast<float>(start_index) / num_frames * window_size +
              sample_offset;
          int32_t end_samples =
              static_cast<float>(num_frames - 1) / num_frames * window_size +
              sample_offset;
          this_speaker_samples.emplace_back(start_samples, end_samples);
        }

        chunk_speaker_list.push_back(std::move(this_chunk_speaker));
        samples_index_list.push_back(std::move(this_speaker_samples));
      }  // for (int32_t speaker_index = 0;
      chunk_index += 1;
    }  // for (const auto &label : new_labels)

    return {chunk_speaker_list, samples_index_list};
  }

  // If there are multiple speakers at a frame, then this frame is excluded.
  std::vector<Matrix2DInt32> ExcludeOverlap(
      const std::vector<Matrix2DInt32> &labels) const {
    int32_t num_chunks = labels.size();
    std::vector<Matrix2DInt32> ans;
    ans.reserve(num_chunks);

    for (const auto &label : labels) {
      Matrix2DInt32 new_label(label.rows(), label.cols());
      new_label.setZero();
      Int32RowVector v = label.rowwise().sum();

      for (int32_t i = 0; i != v.cols(); ++i) {
        if (v[i] < 2) {
          new_label.row(i) = label.row(i);
        }
      }

      ans.push_back(std::move(new_label));
    }

    return ans;
  }

  /**
   * @param sample_indexes[i] contains the sample segment start and end indexes
   *                          for the i-th (chunk, speaker) pair
   * @return Return a matrix of shape (sample_indexes.size(), embedding_dim)
   *         where ans.row[i] contains the embedding for the
   *         i-th (chunk, speaker) pair
   */
  Matrix2D ComputeEmbeddings(
      const float *audio, int32_t n,
      const std::vector<std::vector<Int32Pair>> &sample_indexes,
      std::vector<int32_t> *valid_indexes,
      OfflineSpeakerDiarizationProgressCallback callback,
      void *callback_arg, bool *stopped) const {
    const auto &meta_data = segmentation_meta_data_;
    int32_t sample_rate = meta_data.sample_rate;
    Matrix2D ans(sample_indexes.size(), embedding_extractor_->Dim());

    auto IsNaNWrapper = [](float f) -> bool { return std::isnan(f); };

    if (sample_indexes.empty()) {
      return ans;
    }

    int32_t num_jobs = static_cast<int32_t>(sample_indexes.size());
    int32_t effective_workers =
        std::min(config_.embedding_num_workers, num_jobs);
    if (config_.embedding.debug) {
      SHERPA_ONNX_LOGE(
          "OfflineSpeakerDiarization: embedding jobs=%d, workers=%d", num_jobs,
          effective_workers);
    }

    if (config_.embedding_num_workers > 1) {
      int32_t num_workers = effective_workers;
      std::vector<std::vector<float>> embeddings(num_jobs);
      std::atomic<int32_t> next_job{0};
      std::atomic<int32_t> active_workers{num_workers};
      std::atomic<bool> cancel{false};
      std::mutex completion_mutex;
      std::condition_variable completion_cv;
      std::deque<int32_t> completed_jobs;
      std::exception_ptr worker_exception;

      auto worker = [&]() {
        try {
          while (!cancel.load(std::memory_order_acquire)) {
            int32_t job = next_job.fetch_add(1, std::memory_order_relaxed);
            if (job >= num_jobs) {
              break;
            }

            embeddings[job] = ComputeOneEmbedding(
                audio, n, sample_indexes[job], sample_rate);
            {
              std::lock_guard<std::mutex> lock(completion_mutex);
              completed_jobs.push_back(job);
            }
            completion_cv.notify_one();
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(completion_mutex);
            if (!worker_exception) {
              worker_exception = std::current_exception();
            }
          }
          cancel.store(true, std::memory_order_release);
        }

        active_workers.fetch_sub(1, std::memory_order_acq_rel);
        completion_cv.notify_one();
      };

      std::vector<std::thread> workers;
      workers.reserve(num_workers);
      try {
        for (int32_t i = 0; i != num_workers; ++i) {
          workers.emplace_back(worker);
        }
      } catch (...) {
        cancel.store(true, std::memory_order_release);
        for (auto &t : workers) {
          t.join();
        }
        throw;
      }

      int32_t reported = 0;
      std::exception_ptr coordinator_exception;
      try {
        while (reported < num_jobs) {
          std::unique_lock<std::mutex> lock(completion_mutex);
          completion_cv.wait(lock, [&]() {
            return !completed_jobs.empty() || worker_exception ||
                   active_workers.load(std::memory_order_acquire) == 0;
          });

          if (worker_exception) {
            cancel.store(true, std::memory_order_release);
            break;
          }

          while (!completed_jobs.empty()) {
            completed_jobs.pop_front();
            ++reported;
            lock.unlock();
            if (ShouldStop(callback, reported, num_jobs, callback_arg)) {
              *stopped = true;
              cancel.store(true, std::memory_order_release);
            }
            lock.lock();
            if (*stopped) {
              break;
            }
          }

          if (*stopped ||
              (completed_jobs.empty() &&
               active_workers.load(std::memory_order_acquire) == 0)) {
            break;
          }
        }
      } catch (...) {
        coordinator_exception = std::current_exception();
        cancel.store(true, std::memory_order_release);
      }

      for (auto &t : workers) {
        t.join();
      }

      if (coordinator_exception) {
        std::rethrow_exception(coordinator_exception);
      }
      if (worker_exception) {
        std::rethrow_exception(worker_exception);
      }
      if (*stopped) {
        return {};
      }

      int32_t cur_row_index = 0;
      for (int32_t i = 0; i != num_jobs; ++i) {
        const auto &embedding = embeddings[i];
        if (std::none_of(embedding.begin(), embedding.end(), IsNaNWrapper)) {
          std::copy(embedding.begin(), embedding.end(),
                    &ans(cur_row_index, 0));
          ++cur_row_index;
          valid_indexes->push_back(i);
        }
      }

      if (cur_row_index != ans.rows()) {
        auto seq = Eigen::seqN(0, cur_row_index);
        ans = ans(seq, Eigen::placeholders::all);
      }
      return ans;
    }

    int32_t k = 0;
    int32_t cur_row_index = 0;
    for (const auto &v : sample_indexes) {
      std::vector<float> embedding =
          ComputeOneEmbedding(audio, n, v, sample_rate);

      if (std::none_of(embedding.begin(), embedding.end(), IsNaNWrapper)) {
        // a valid embedding
        std::copy(embedding.begin(), embedding.end(), &ans(cur_row_index, 0));
        cur_row_index += 1;
        valid_indexes->push_back(k);
      }

      k += 1;

      if (ShouldStop(callback, k, ans.rows(), callback_arg)) {
        *stopped = true;
        break;
      }
    }

    if (k != cur_row_index) {
      auto seq = Eigen::seqN(0, cur_row_index);
      ans = ans(seq, Eigen::placeholders::all);
    }

    return ans;
  }

  std::vector<float> ComputeOneEmbedding(
      const float *audio, int32_t n, const std::vector<Int32Pair> &indexes,
      int32_t sample_rate) const {
    auto stream = embedding_extractor_->CreateStream();
    for (const auto &p : indexes) {
      int32_t end = (p.second <= n) ? p.second : n;
      int32_t num_samples = end - p.first;

      if (num_samples > 0) {
        stream->AcceptWaveform(sample_rate, audio + p.first, num_samples);
      }
    }

    stream->InputFinished();
    if (!embedding_extractor_->IsReady(stream.get())) {
      throw std::runtime_error(
          "Speaker embedding segment is unexpectedly too short");
    }

    return embedding_extractor_->Compute(stream.get());
  }

  std::unordered_map<Int32Pair, int32_t, PairHash> ConvertChunkSpeakerToCluster(
      const std::vector<Int32Pair> &chunk_speaker_pair,
      const std::vector<int32_t> &cluster_labels) const {
    std::unordered_map<Int32Pair, int32_t, PairHash> ans;

    int32_t k = 0;
    for (const auto &p : chunk_speaker_pair) {
      if (k >= static_cast<int32_t>(cluster_labels.size())) {
        SHERPA_ONNX_LOGE("cluster_labels size mismatch: k=%d, size=%d", k,
                         static_cast<int32_t>(cluster_labels.size()));
        break;
      }
      ans[p] = cluster_labels[k];
      k += 1;
    }

    return ans;
  }

  std::vector<Matrix2DInt32> ReLabel(
      const std::vector<Matrix2DInt32> &labels, int32_t max_cluster_index,
      std::unordered_map<Int32Pair, int32_t, PairHash> chunk_speaker_to_cluster)
      const {
    std::vector<Matrix2DInt32> new_labels;
    new_labels.reserve(labels.size());

    int32_t chunk_index = 0;
    for (const auto &label : labels) {
      Matrix2DInt32 new_label(label.rows(), max_cluster_index + 1);
      new_label.setZero();

      Matrix2DInt32 t = label.transpose();
      // t: (num_speakers, num_frames)

      for (int32_t speaker_index = 0; speaker_index != t.rows();
           ++speaker_index) {
        if (chunk_speaker_to_cluster.count({chunk_index, speaker_index}) == 0) {
          continue;
        }

        int32_t new_speaker_index =
            chunk_speaker_to_cluster.at({chunk_index, speaker_index});

        if (new_speaker_index < 0 ||
            new_speaker_index >= new_label.cols()) {
          SHERPA_ONNX_LOGE(
              "ReLabel: new_speaker_index %d out of range [0, %d), skipping",
              new_speaker_index,
              static_cast<int32_t>(new_label.cols()));
          continue;
        }

        for (int32_t k = 0; k != t.cols(); ++k) {
          if (t(speaker_index, k) == 1) {
            new_label(k, new_speaker_index) = 1;
          }
        }
      }

      new_labels.push_back(std::move(new_label));

      chunk_index += 1;
    }

    return new_labels;
  }

  Matrix2DInt32 ComputeSpeakerCount(const std::vector<Matrix2DInt32> &labels,
                                    int32_t num_samples) const {
    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    Matrix2DInt32 count(num_frames, labels[0].cols());
    count.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq, Eigen::placeholders::all).array() += labels[i].array();
    }

    bool has_last_chunk = ((num_samples - window_size) % window_shift) > 0;

    if (!has_last_chunk) {
      return count;
    }

    int32_t last_frame = num_samples / receptive_field_shift;
    if (last_frame >= count.rows()) {
      last_frame = count.rows() - 1;
    }
    return count(Eigen::seqN(0, last_frame + 1), Eigen::placeholders::all);
  }

  Matrix2DInt32 FinalizeLabels(const Matrix2DInt32 &count,
                               const Int32RowVector &speakers_per_frame) const {
    int32_t num_rows = count.rows();
    int32_t num_cols = count.cols();

    Matrix2DInt32 ans(num_rows, num_cols);
    ans.setZero();

    for (int32_t i = 0; i != num_rows; ++i) {
      int32_t k = speakers_per_frame[i];
      if (k == 0) {
        continue;
      }
      auto top_k = TopkIndex(&count(i, 0), num_cols, k);

      for (int32_t m : top_k) {
        if (m >= 0 && m < num_cols) {
          ans(i, m) = 1;
        }
      }
    }

    return ans;
  }

  OfflineSpeakerDiarizationResult ComputeResult(
      const Matrix2DInt32 &final_labels) const {
    Matrix2DInt32 final_labels_t = final_labels.transpose();
    int32_t num_speakers = final_labels_t.rows();
    int32_t num_frames = final_labels_t.cols();

    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t receptive_field_size = meta_data.receptive_field_size;
    int32_t sample_rate = meta_data.sample_rate;

    float scale = static_cast<float>(receptive_field_shift) / sample_rate;
    float scale_offset = 0.5 * receptive_field_size / sample_rate;

    OfflineSpeakerDiarizationResult ans;

    for (int32_t speaker_index = 0; speaker_index != num_speakers;
         ++speaker_index) {
      std::vector<OfflineSpeakerDiarizationSegment> this_speaker;

      bool is_active = final_labels_t(speaker_index, 0) > 0;
      int32_t start_index = is_active ? 0 : -1;

      for (int32_t frame_index = 1; frame_index != num_frames; ++frame_index) {
        if (is_active) {
          if (final_labels_t(speaker_index, frame_index) == 0) {
            float start_time = start_index * scale + scale_offset;
            float end_time = frame_index * scale + scale_offset;

            OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                     speaker_index);
            this_speaker.push_back(segment);

            is_active = false;
          }
        } else if (final_labels_t(speaker_index, frame_index) == 1) {
          is_active = true;
          start_index = frame_index;
        }
      }

      if (is_active) {
        float start_time = start_index * scale + scale_offset;
        float end_time = (num_frames - 1) * scale + scale_offset;

        OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                 speaker_index);
        this_speaker.push_back(segment);
      }

      // merge segments if the gap between them is less than min_duration_off
      MergeSegments(&this_speaker);

      for (const auto &seg : this_speaker) {
        if (seg.Duration() > config_.min_duration_on) {
          ans.Add(seg);
        }
      }
    }  // for (int32_t speaker_index = 0; speaker_index != num_speakers;

    return ans;
  }

  OfflineSpeakerDiarizationResult HandleOneChunkSpecialCase(
      const Matrix2DInt32 &final_labels, int32_t num_samples) const {
    const auto &meta_data = segmentation_meta_data_;
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    bool has_last_chunk = (num_samples - window_size) % window_shift > 0;
    if (!has_last_chunk) {
      return ComputeResult(final_labels);
    }

    int32_t num_frames = final_labels.rows();

    int32_t new_num_frames = num_samples / receptive_field_shift;

    int32_t num_frames_to_keep = std::min(new_num_frames, num_frames);

    return ComputeResult(
        final_labels(Eigen::seqN(0, num_frames_to_keep), Eigen::placeholders::all));
  }

  void MergeSegments(
      std::vector<OfflineSpeakerDiarizationSegment> *segments) const {
    float min_duration_off = config_.min_duration_off;
    bool changed = true;
    while (changed) {
      changed = false;
      for (int32_t i = 0; i < static_cast<int32_t>(segments->size()) - 1; ++i) {
        auto s = (*segments)[i].Merge((*segments)[i + 1], min_duration_off);
        if (s) {
          (*segments)[i] = s.value();
          segments->erase(segments->begin() + i + 1);

          changed = true;
          break;
        }
      }
    }
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  mutable std::unique_ptr<OfflineSpeakerSegmentationPyannoteModel>
      segmentation_model_;
  OfflineSpeakerSegmentationPyannoteModelMetaData segmentation_meta_data_;
  std::function<std::unique_ptr<SpeakerEmbeddingExtractor>()>
      embedding_factory_;
  mutable std::unique_ptr<SpeakerEmbeddingExtractor> embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
  Matrix2DInt32 powerset_mapping_;
  mutable std::atomic<bool> process_started_{false};
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
