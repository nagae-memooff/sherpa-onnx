#include "sherpa-onnx/csrc/qwen3-compression.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#define Z_PREFIX
#include "zlib.h"
namespace sherpa_onnx {
bool Qwen3CompressionConfig::Valid() const {
  return mode >= 0 && mode <= 2 && std::isfinite(threshold) && threshold > 1 &&
      min_tokens > 0 && window_tokens >= 16 && window_tokens <= 4096 &&
      interval > 0 && consecutive > 0 && consecutive <= 32;
}
double Qwen3TextCompressionRatio(const std::string &text) {
  std::vector<unsigned char> compressed(compressBound(text.size()));
  uLongf size = compressed.size();
  if (compress2(compressed.data(), &size,
                reinterpret_cast<const Bytef *>(text.data()), text.size(), 6) != Z_OK)
    throw std::runtime_error("Qwen text compression failed");
  return static_cast<double>(text.size()) / size;
}
}  // namespace sherpa_onnx
