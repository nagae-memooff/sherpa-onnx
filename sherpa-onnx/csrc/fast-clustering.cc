// sherpa-onnx/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

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
    m.rowwise().normalize();

    std::vector<double> distance((num_rows * (num_rows - 1)) / 2);

    int32_t k = 0;
    for (int32_t i = 0; i != num_rows; ++i) {
      auto v = m.row(i);
      for (int32_t j = i + 1; j != num_rows; ++j) {
        double cosine_similarity = v.dot(m.row(j));
        double consine_dissimilarity = 1 - cosine_similarity;

        if (consine_dissimilarity < 0) {
          consine_dissimilarity = 0;
        }

        distance[k] = consine_dissimilarity;
        ++k;
      }
    }

    std::vector<int32_t> merge(2 * (num_rows - 1));
    std::vector<double> height(num_rows - 1);

    fastclustercpp::hclust_fast(num_rows, distance.data(),
                                fastclustercpp::HCLUST_METHOD_AVERAGE,
                                merge.data(), height.data());

    std::vector<int32_t> labels(num_rows);
    if (config_.num_clusters > 0) {
      fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters,
                               labels.data());
    } else {
      fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                   config_.threshold, labels.data());
    }

    // 若只有一个簇，直接返回，避免后续处理在 k=1 情况下再做多余操作
    {
      std::unordered_map<int32_t, int32_t> uniq;
      for (int32_t lbl : labels) uniq[lbl] += 1;
      if (uniq.size() <= 1) {
        return labels;
      }
    }

    const char *env_pyannote_like = std::getenv("SHERPA_PYANNOTE_LIKE");
    // 默认开启 pyannote_like，如需关闭显式设为 "false"/"0"
    bool use_pyannote_like = true;
    if (env_pyannote_like != nullptr) {
      std::string v = env_pyannote_like;
      for (auto &c : v) c = static_cast<char>(::tolower(c));
      if (v == "0" || v == "false") use_pyannote_like = false;
      if (v == "true") use_pyannote_like = true;
    }

    if (use_pyannote_like) {
      printf("use pyannote like.\n");
      // 参考 pyannote：对小簇做二次合并，避免阈值导致碎片化
      // min_cluster_size：基础值来自 SHERPA_MIN_CLUSTER_SIZE（默认 24），
      // 再按 pyannote 的启发式裁剪：min(base, max(1, round(0.1 * N))).
      int32_t base_min_cluster_size = 24;
      if (const char *env_min = std::getenv("SHERPA_MIN_CLUSTER_SIZE")) {
        try {
          int32_t v = std::stoi(env_min);
          if (v > 0) base_min_cluster_size = v;
        } catch (...) {
        }
      }
      int32_t min_cluster_size =
          std::min<int32_t>(base_min_cluster_size,
                            std::max<int32_t>(1, static_cast<int32_t>(std::round(0.1 * num_rows))));
      if (min_cluster_size > num_rows) min_cluster_size = num_rows;

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
        // 计算各簇质心（使用前面统一处理过的 m）
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

        // 将小簇指派到最近的大簇（与上游相同的度量：1-cosine）
        for (int32_t sc : small_clusters) {
          double best_dist = std::numeric_limits<double>::infinity();
          int32_t best_lc = large_clusters.front();
          for (int32_t lc : large_clusters) {
            double n1 = centroids[sc].norm();
            double n2 = centroids[lc].norm();
            double cos = 0.0;
            if (n1 > 1e-12 && n2 > 1e-12) {
              cos = centroids[sc].dot(centroids[lc]) / (n1 * n2);
              cos = std::clamp(cos, -1.0, 1.0);
            }
            double d = 1.0 - cos;
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

    // 调试输出：打印簇间距离（最近/平均），便于人工判断是否可合并
    const char *env_debug_dist = std::getenv("SHERPA_CLUSTER_DEBUG_DIST");
    // 默认开启，如需关闭显式设为 "false"/"0"
    bool debug_dist = true;
    if (env_debug_dist != nullptr) {
      std::string v = env_debug_dist;
      for (auto &c : v) c = static_cast<char>(::tolower(c));
      debug_dist = !(v == "0" || v == "false");
    }

    auto PrintClusterDistances = [&](const char *tag) {
      // 计算当前簇质心
      std::unordered_map<int32_t, Eigen::VectorXd> centroids;
      std::unordered_map<int32_t, int32_t> counts;
      for (int32_t lbl : labels) {
        centroids.try_emplace(lbl, Eigen::VectorXd::Zero(num_cols));
        counts[lbl] += 1;
      }
      for (int32_t i = 0; i < num_rows; ++i) {
        centroids[labels[i]] += m.row(i).cast<double>();
      }
      for (auto &kv : centroids) {
        kv.second /= static_cast<double>(counts[kv.first]);
      }

      printf("[cluster distances] %s\n", tag);
      printf("  k=%zu\n", centroids.size());
      // 按簇大小降序打印，便于观察大小与距离的关系
      std::vector<std::pair<int32_t, int32_t>> size_list;
      size_list.reserve(counts.size());
      for (auto &kv : counts) {
        size_list.push_back({kv.first, kv.second});
      }
      std::sort(size_list.begin(), size_list.end(),
                [](const auto &a, const auto &b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
                });

      for (auto &p : size_list) {
        int32_t id = p.first;
        int32_t sz = p.second;
        double nearest = std::numeric_limits<double>::infinity();
        int32_t nearest_id = -1;
        double sum = 0.0;
        int32_t cnt = 0;
        for (auto &kv2 : centroids) {
          if (kv2.first == id) continue;
          // average 路径使用 1-cosine（与主流程一致）
          double n1 = centroids[id].norm();
          double n2 = kv2.second.norm();
          double cos = 0.0;
          if (n1 > 1e-12 && n2 > 1e-12) {
            cos = centroids[id].dot(kv2.second) / (n1 * n2);
            cos = std::clamp(cos, -1.0, 1.0);
          }
          double d = 1.0 - cos;
          sum += d;
          cnt += 1;
          if (d < nearest) {
            nearest = d;
            nearest_id = kv2.first;
          }
        }
        double avg = (cnt > 0) ? (sum / cnt) : 0.0;
        printf("    cluster %d (size=%d): nearest=%.4f (to %d), avg=%.4f\n",
               id, sz, nearest, nearest_id, avg);
      }
    };

    // 若当前簇数已<=1，打印后直接返回，避免二次合并再进入异常路径
    {
      std::unordered_map<int32_t, int32_t> uniq;
      for (int32_t lbl : labels) uniq[lbl] += 1;
      if (uniq.size() <= 1) {
        if (debug_dist && !labels.empty()) {
          PrintClusterDistances("k<=1 skip second-merge");
        } else if (debug_dist) {
          printf("[cluster distances] debug enabled but no labels to report\n");
        }
        return labels;
      }
    }

    if (debug_dist && !labels.empty()) {
      PrintClusterDistances("before second-merge");
    } else if (debug_dist) {
      printf("[cluster distances] debug enabled but no labels to report\n");
    }

    // 可选的二次合并：按簇质心距离继续合并，进一步减少碎片化。
    double merge_threshold = 0.4;  // 默认 0.4，可通过环境变量覆盖
    if (const char *env_merge = std::getenv("SHERPA_CLUSTER_MERGE_THRESHOLD")) {
      try {
        merge_threshold = std::stod(env_merge);
      } catch (...) {
      }
    }

    if (merge_threshold > 0 && !labels.empty()) {
      bool merged = true;
      while (merged) {
        merged = false;

        // 计算当前簇质心
        std::unordered_map<int32_t, Eigen::VectorXd> centroids;
        std::unordered_map<int32_t, int32_t> counts;
        for (int32_t lbl : labels) {
          centroids.try_emplace(lbl, Eigen::VectorXd::Zero(num_cols));
          counts[lbl] += 1;
        }
        if (centroids.size() < 2) {
          break;  // 不需要继续合并
        }
        for (int32_t i = 0; i < num_rows; ++i) {
          centroids[labels[i]] += m.row(i).cast<double>();
        }
        for (auto &kv : centroids) {
          kv.second /= static_cast<double>(counts[kv.first]);
        }

        // 找距离最近的簇对，若 1-cos 距离低于阈值则合并
        std::vector<int32_t> ids;
        ids.reserve(centroids.size());
        for (auto &kv : centroids) ids.push_back(kv.first);

        double best_dist = std::numeric_limits<double>::infinity();
        int32_t a = -1, b = -1;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
          for (size_t j = i + 1; j < ids.size(); ++j) {
            double n1 = centroids[ids[i]].norm();
            double n2 = centroids[ids[j]].norm();
            double cos = 0.0;
            if (n1 > 1e-12 && n2 > 1e-12) {
              cos = centroids[ids[i]].dot(centroids[ids[j]]) / (n1 * n2);
              cos = std::clamp(cos, -1.0, 1.0);
            }
            double d = 1.0 - cos;
            if (d < best_dist) {
              best_dist = d;
              a = ids[i];
              b = ids[j];
            }
          }
        }

        if (best_dist > 0 && best_dist < merge_threshold && a != -1 && b != -1) {
          for (int32_t &lbl : labels) {
            if (lbl == b) lbl = a;
          }
          merged = true;
        }
      }

      // 重编号为 0..K-1
      std::unordered_map<int32_t, int32_t> remap;
      int32_t idx = 0;
      for (int32_t lbl : labels) {
        if (remap.find(lbl) == remap.end()) remap[lbl] = idx++;
      }
      for (int32_t &lbl : labels) lbl = remap[lbl];

      if (debug_dist && !labels.empty()) {
        PrintClusterDistances("after second-merge");
      }
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
