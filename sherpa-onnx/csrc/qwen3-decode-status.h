// Qwen3 解码状态独立于性能 profiling；原始 ID 仅在显式诊断时保存。
#ifndef SHERPA_ONNX_CSRC_QWEN3_DECODE_STATUS_H_
#define SHERPA_ONNX_CSRC_QWEN3_DECODE_STATUS_H_
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sherpa_onnx {
struct Qwen3DecodeStatus {
  std::string reason = "inference_error";
  bool input_truncated = false;
  bool first_eos_overridden = false;
  bool diagnostics = false;
  int32_t original_audio_tokens = 0;
  int32_t audio_tokens = 0;
  int32_t original_context_tokens = 0;
  int32_t context_tokens = 0;
  int32_t max_total_len = 0;
  int32_t max_new_tokens = 0;
  int32_t generated_tokens = 0;
  int32_t retained_tokens = 0;
  int32_t repetition_window = 0;
  int32_t repetition_period = 0;
  bool repetition_processed = false;
  int32_t compression_mode = 2;
  double compression_threshold = 2.4;
  double compression_ratio = 0;
  double compression_max_ratio = 0;
  double compression_check_ms = 0;
  int32_t compression_checks = 0;
  int32_t compression_hits = 0;
  int64_t eos_id = -1;
  int64_t first_token_id = -1;
  int64_t replacement_token_id = -1;
  int64_t stop_token_id = -1;
  double first_eos_logit = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::pair<int64_t, double>> first_candidates;
  std::vector<int64_t> prompt_ids;
  std::vector<int64_t> generated_ids;

  bool Complete() const {
    return !input_truncated &&
           (reason == "eos" || reason == "no_input" ||
            reason == "no_audio_features");
  }

  std::string AsJson() const {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(9);
    os << "{\"schema_version\":1,\"termination_reason\":\"" << reason
       << "\",\"complete\":" << (Complete() ? "true" : "false")
       << ",\"input_truncated\":" << (input_truncated ? "true" : "false")
       << ",\"first_eos_overridden\":" << (first_eos_overridden ? "true" : "false")
       << ",\"original_audio_tokens\":" << original_audio_tokens
       << ",\"audio_tokens\":" << audio_tokens
       << ",\"original_context_tokens\":" << original_context_tokens
       << ",\"context_tokens\":" << context_tokens
       << ",\"max_total_len\":" << max_total_len
       << ",\"max_new_tokens\":" << max_new_tokens
       << ",\"generated_tokens\":" << generated_tokens
       << ",\"retained_tokens\":" << retained_tokens
       << ",\"removed_tokens\":" << generated_tokens - retained_tokens
       << ",\"repetition_window\":" << repetition_window
       << ",\"repetition_period\":" << repetition_period
       << ",\"repetition_processed\":" << (repetition_processed ? "true" : "false")
       << ",\"compression_mode\":" << compression_mode
       << ",\"compression_threshold\":" << compression_threshold
       << ",\"compression_ratio\":" << compression_ratio
       << ",\"compression_max_ratio\":" << compression_max_ratio
       << ",\"compression_check_ms\":" << compression_check_ms
       << ",\"compression_checks\":" << compression_checks
       << ",\"compression_hits\":" << compression_hits;
    if (diagnostics) {
      os << ",\"diagnostics\":{\"eos_id\":" << eos_id
         << ",\"first_token_id\":" << first_token_id
         << ",\"replacement_token_id\":" << replacement_token_id
         << ",\"stop_token_id\":" << stop_token_id
         << ",\"first_eos_logit\":";
      if (std::isfinite(first_eos_logit)) {
        os << first_eos_logit;
      } else {
        os << "null";
      }
      os << ",\"first_candidates\":[";
      const char *sep = "";
      for (const auto &candidate : first_candidates) {
        os << sep << "{\"id\":" << candidate.first << ",\"logit\":";
        if (std::isfinite(candidate.second)) {
          os << candidate.second;
        } else {
          os << "null";
        }
        os << "}";
        sep = ",";
      }
      os << "]";
      auto append_ids = [&os](const char *key, const std::vector<int64_t> &ids) {
        os << ",\"" << key << "\":[";
        const char *sep = "";
        for (int64_t id : ids) {
          os << sep << id;
          sep = ",";
        }
        os << "]";
      };
      append_ids("prompt_ids", prompt_ids);
      append_ids("generated_ids", generated_ids);
      os << "}";
    }
    os << "}";
    return os.str();
  }
};
}  // namespace sherpa_onnx
#endif
