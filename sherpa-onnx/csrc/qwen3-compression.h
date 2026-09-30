#ifndef SHERPA_ONNX_CSRC_QWEN3_COMPRESSION_H_
#define SHERPA_ONNX_CSRC_QWEN3_COMPRESSION_H_
#include <cstdint>
#include <string>
namespace sherpa_onnx {
struct Qwen3CompressionConfig {
  int32_t mode = 2;  // 0 关闭，1 仅记录，2 提前停止。
  float threshold = 2.4f;
  int32_t min_tokens = 64;
  int32_t window_tokens = 128;
  int32_t interval = 16;
  int32_t consecutive = 2;
  bool Valid() const;
};
// 与 Whisper Python 的 UTF-8 / zlib.compress(level=6) 比值一致。
double Qwen3TextCompressionRatio(const std::string &text);
}  // namespace sherpa_onnx
#endif
