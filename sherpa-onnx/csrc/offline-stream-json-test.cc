#include "sherpa-onnx/csrc/offline-stream.h"

#include <string>

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
}  // namespace sherpa_onnx
