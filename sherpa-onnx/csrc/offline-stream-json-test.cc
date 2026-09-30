#include "sherpa-onnx/csrc/offline-stream.h"

#include <string>
#include "sherpa-onnx/csrc/qwen3-decode-status.h"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace sherpa_onnx {
TEST(OfflineResultJson, NewlinePreservesTextAndProfile) {
  OfflineRecognitionResult result;
  result.text = "system\n";
  result.profile_json = R"({"total_ms":304.2,"allocator_stats":{"after":{"cpu":{"decoder":{"available":true}}}}})";
  const auto json = nlohmann::json::parse(result.AsJsonString());
  EXPECT_EQ(json.at("text").get<std::string>(), result.text);
  EXPECT_EQ(json.at("qwen_profile"), nlohmann::json::parse(result.profile_json));
}

TEST(OfflineResultJson, AllControlCharactersRoundTripInEveryStringField) {
  std::string text = "中文\"\\";
  for (int i = 0; i < 32; ++i) text.push_back(static_cast<char>(i));
  OfflineRecognitionResult result;
  result.text = result.lang = result.emotion = result.event = text;
  result.tokens = {text, "", std::string(1, static_cast<char>(0x80))};
  result.segment_timestamps = {1.25};
  result.segment_durations = {2.5};
  result.segment_texts = {text};
  const auto json = nlohmann::json::parse(result.AsJsonString());
  for (const char *key : {"text", "lang", "emotion", "event"}) {
    EXPECT_EQ(json.at(key).get<std::string>(), text);
  }
  EXPECT_EQ(json.at("tokens").at(0).get<std::string>(), text);
  EXPECT_EQ(json.at("tokens").at(1).get<std::string>(), "");
  EXPECT_EQ(json.at("tokens").at(2).get<std::string>(), "<0x80>");
  EXPECT_EQ(json.at("segment_texts").at(0).get<std::string>(), text);
  EXPECT_DOUBLE_EQ(json.at("segment_timestamps").at(0).get<double>(), 1.25);
  EXPECT_DOUBLE_EQ(json.at("segment_durations").at(0).get<double>(), 2.5);
  EXPECT_EQ(json.count("qwen_profile"), 0u);
}
// 状态不依赖 profile，裁剪不能伪装成完整成功。
TEST(QwenDecodeStatus, CompletionAndRawCounts) {
  Qwen3DecodeStatus status;
  status.reason = "eos";
  EXPECT_TRUE(status.Complete());
  status.input_truncated = true;
  EXPECT_FALSE(status.Complete());
  status.input_truncated = false;
  status.reason = "repetition_guard";
  status.generated_tokens = 90;
  status.retained_tokens = 26;
  OfflineRecognitionResult result;
  result.text = "正常前文";
  result.qwen_decode_json = status.AsJson();
  auto json = nlohmann::json::parse(result.AsJsonString());
  EXPECT_FALSE(json.at("qwen_decode").at("complete").get<bool>());
  EXPECT_EQ(json.at("qwen_decode").at("removed_tokens"), 64);
  EXPECT_FALSE(json.contains("qwen_profile"));
  EXPECT_FALSE(json.at("qwen_decode").contains("diagnostics"));
}

TEST(QwenDecodeStatus, DiagnosticsAndNonfiniteLogits) {
  Qwen3DecodeStatus status;
  status.diagnostics = true;
  status.first_eos_overridden = true;
  status.first_token_id = 151645;
  status.replacement_token_id = 9125;
  status.generated_ids = {9125, 198};
  status.prompt_ids = {151644, 8948};
  status.first_candidates = {{151645, 10.0}, {9125, 9.5}};
  auto json = nlohmann::json::parse(status.AsJson());
  EXPECT_TRUE(json.at("first_eos_overridden").get<bool>());
  EXPECT_TRUE(json.at("diagnostics").at("first_eos_logit").is_null());
  EXPECT_EQ(json.at("diagnostics").at("generated_ids").size(), 2u);
  EXPECT_EQ(json.at("diagnostics").at("first_token_id"), 151645);
  for (const char *reason : {"output_limit", "context_limit", "invalid_logits",
                            "unexpected_stop_token", "inference_error"}) {
    status.reason = reason;
    EXPECT_FALSE(status.Complete());
  }
}
}  // namespace sherpa_onnx

namespace sherpa_onnx {
TEST(QwenDecodeStatus, PeriodicGuardRemainsIncomplete) {
  Qwen3DecodeStatus status;
  status.reason = "repetition_guard";
  status.generated_tokens = 94;
  status.retained_tokens = 24;
  status.repetition_period = 7;
  status.repetition_window = 70;
  auto value = nlohmann::json::parse(status.AsJson());
  EXPECT_FALSE(value.at("complete").get<bool>());
  EXPECT_EQ(value.at("repetition_period"), 7);
  EXPECT_EQ(value.at("repetition_window"), 70);
  EXPECT_EQ(value.at("removed_tokens"), 70);
}
}  // namespace sherpa_onnx
