#ifndef SHERPA_ONNX_CSRC_QWEN3_REPETITION_H_
#define SHERPA_ONNX_CSRC_QWEN3_REPETITION_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sherpa_onnx {
struct Qwen3PeriodicRepetition {
  int32_t period = 0;
  int32_t span = 0;
};

// 保守地检测完全相同的尾部周期；既要求 8 次重复，也要求至少 64 个 token。
// 这只是生成退化保护，不能用于断言音频中没有真实重复。
inline Qwen3PeriodicRepetition FindQwen3PeriodicRepetition(
    const std::vector<int64_t> &ids) {
  constexpr int32_t kMaxPeriod = 32;
  constexpr int32_t kMinRepeats = 8;
  constexpr int32_t kMinSpan = 64;
  for (int32_t period = 1; period <= kMaxPeriod; ++period) {
    const int32_t repeats = std::max(kMinRepeats,
                                    (kMinSpan + period - 1) / period);
    const int32_t span = repeats * period;
    if (ids.size() < static_cast<size_t>(span)) continue;
    const size_t start = ids.size() - span;
    bool same = true;
    for (size_t i = start + period; i < ids.size(); ++i) {
      if (ids[i] != ids[i - period]) {
        same = false;
        break;
      }
    }
    if (same) return {period, span};
  }
  return {};
}
inline Qwen3PeriodicRepetition CollapseQwen3RepeatedTail(
    std::vector<int64_t> *ids) {
  const int32_t size = static_cast<int32_t>(ids->size());
  for (int32_t period = 1; period <= 128 && period * 3 <= size; ++period) {
    int32_t span = period;
    while (span < size && (*ids)[size - 1 - span] ==
                             (*ids)[size - 1 - span % period]) ++span;
    if (span >= std::max(8, period * 3)) {
      ids->resize(size - span + period);
      return {period, span};
    }
  }
  return {};
}
}  // namespace sherpa_onnx
#endif
