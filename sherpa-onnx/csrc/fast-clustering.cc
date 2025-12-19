// sherpa-onnx/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering.h"

#include <vector>
#include <unordered_map>
#include <limits>
#include <cmath>

#include "Eigen/Dense"
#include "fastcluster-all-in-one.h"  // NOLINT

namespace sherpa_onnx {

class FastClustering::Impl {
 public:
  explicit Impl(const FastClusteringConfig &config) : config_(config) {}

  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols) const {
    if (num_rows <= 0) {
      return {};
    }

    if (num_rows == 1) {
      return {0};
    }

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m(features, num_rows, num_cols);

    std::vector<double> distance((num_rows * (num_rows - 1)) / 2);

    // 通过环境变量控制距离构造与算法：
    // - SHERPA_CLUSTER_METHOD=centroid  -> centroid/欧氏距离路径
    // - 其他或未设置                 -> average/1-cosine 路径
    const char *env_method = std::getenv("SHERPA_CLUSTER_METHOD");
    const bool use_centroid = (env_method == nullptr) ? false : (std::string(env_method) == "centroid");


    // cosine 路径需要单位化，centroid 路径也建议单位化以对齐 pyannote
    if (use_centroid) {
      const char *env_normalize = std::getenv("SHERPA_CLUSTER_NORMALIZE");
      const bool use_normalize = (env_normalize == nullptr) ? false : (std::string(env_normalize) == "true");

      if (use_normalize) {
        m.rowwise().normalize();
        printf("use cluster method is 'centroid', will normalize\n");
      } else {
        printf("use cluster method is 'centroid', will not normalize\n");
      }

    } else {
      printf("use cluster method is 'average', will normalize.\n");
      m.rowwise().normalize();
    }

    int32_t k = 0;
    for (int32_t i = 0; i != num_rows; ++i) {
      auto v = m.row(i);
      for (int32_t j = i + 1; j != num_rows; ++j) {
        if (use_centroid) {
          // 欧氏距离按原始定义传入，fastcluster 内部再做处理
          double euclidean = (v - m.row(j)).norm();
          distance[k] = euclidean;
        } else {
          double cosine_similarity = v.dot(m.row(j));
          double consine_dissimilarity = 1 - cosine_similarity;
          if (consine_dissimilarity < 0) {
            consine_dissimilarity = 0;
          }
          distance[k] = consine_dissimilarity;
        }
        ++k;
      }
    }

    std::vector<int32_t> merge(2 * (num_rows - 1));
    std::vector<double> height(num_rows - 1);

    if (use_centroid) {
      fastclustercpp::hclust_fast(num_rows, distance.data(),
                                  fastclustercpp::HCLUST_METHOD_CENTROID,
                                  merge.data(), height.data());
    } else {
      fastclustercpp::hclust_fast(num_rows, distance.data(),
                                  fastclustercpp::HCLUST_METHOD_AVERAGE,
                                  merge.data(), height.data());
    }

    std::vector<int32_t> labels(num_rows);
    if (config_.num_clusters > 0) {
      fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters,
                               labels.data());
    } else {
      fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                   config_.threshold, labels.data());
    }

    const char *env_pyannote_like = std::getenv("SHERPA_PYANNOTE_LIKE");
    const bool use_pyannote_like = (env_pyannote_like == nullptr) ? false : (std::string(env_pyannote_like) == "true");

    if (use_pyannote_like) {
      printf("use pyannote like.\n");
      // 参考 pyannote：对小簇做二次合并，避免阈值导致碎片化
      // min_cluster_size 默认 5，且不超过样本数的 10%
      int32_t min_cluster_size =
        std::min<int32_t>(5, std::max<int32_t>(1, static_cast<int32_t>(std::round(0.1 * num_rows))));
      if (min_cluster_size < 1) min_cluster_size = 1;

      // 统计簇大小
      std::unordered_map<int32_t, int32_t> cluster_size;
      for (int32_t lbl : labels) cluster_size[lbl]++;

      std::vector<int32_t> large_clusters;
      std::vector<int32_t> small_clusters;
      for (auto &kv : cluster_size) {
        if (kv.second >= min_cluster_size) large_clusters.push_back(kv.first);
        else small_clusters.push_back(kv.first);
      }

      if (large_clusters.empty()) {
        // pyannote: 若没有大簇，则直接归为一个簇，避免碎片化
        for (auto &lbl : labels) lbl = 0;
      } else if (!small_clusters.empty()) {
        // 计算各簇质心（基于已归一化的 m）
        const int32_t dim = num_cols;
        std::unordered_map<int32_t, Eigen::VectorXd> centroids;
        for (auto &kv : cluster_size) {
          centroids.emplace(kv.first, Eigen::VectorXd::Zero(dim));
        }
        for (int32_t i = 0; i < num_rows; ++i) {
          centroids[labels[i]] += m.row(i).cast<double>();
        }
        for (auto &kv : cluster_size) {
          centroids[kv.first] /= static_cast<double>(kv.second);
        }

        // 将小簇指派到最近的大簇（欧氏距离）
        for (int32_t sc : small_clusters) {
          double best_dist = std::numeric_limits<double>::infinity();
          int32_t best_lc = large_clusters.front();
          for (int32_t lc : large_clusters) {
            double d = (centroids[sc] - centroids[lc]).norm();
            if (d < best_dist) {
              best_dist = d;
              best_lc = lc;
            }
          }
          for (int32_t i = 0; i < num_rows; ++i) {
            if (labels[i] == sc) labels[i] = best_lc;
          }
        }

        // 重编号为 0..K-1
        std::unordered_map<int32_t, int32_t> remap;
        int32_t idx = 0;
        for (int32_t lc : large_clusters) remap[lc] = idx++;
        for (int32_t i = 0; i < num_rows; ++i) labels[i] = remap[labels[i]];
      }
    } else {
      printf("not use pyannote like.\n");
    }

    return labels;
  }

 private:
  FastClusteringConfig config_;
};

FastClustering::FastClustering(const FastClusteringConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FastClustering::~FastClustering() = default;

std::vector<int32_t> FastClustering::Cluster(float *features, int32_t num_rows,
                                             int32_t num_cols) const {
  return impl_->Cluster(features, num_rows, num_cols);
}
}  // namespace sherpa_onnx
