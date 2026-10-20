// 模型加载失败的进程级回归测试。fixture 由同目录 Python 脚本生成。
#include "sherpa-onnx/c-api/c-api.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const std::string mode = argv[1], dir = argv[2];
  const int repeats = std::atoi(argv[3]);
  const std::string bad = dir + "/corrupt.onnx", tokens = dir + "/tokens.txt";
  const std::string conv = dir + "/conv.onnx", enc = dir + "/encoder.onnx";
  const std::string dec = dir + "/decoder.onnx", tokenizer = dir + "/tokenizer";
  const std::string bad_tokenizer = dir + "/bad-tokenizer";
  const std::string wrong = dir + "/wrong.onnx", vad_path = dir + "/vad.onnx";
  SherpaOnnxOfflineRecognizerConfig q{};
  q.model_config.qwen3_asr.conv_frontend = conv.c_str();
  q.model_config.qwen3_asr.encoder = enc.c_str();
  q.model_config.qwen3_asr.decoder = dec.c_str();
  q.model_config.qwen3_asr.tokenizer = tokenizer.c_str();
  q.model_config.num_threads = 1;
  q.model_config.provider = "cpu";
  q.model_config.qwen3_asr.max_total_len = 32;
  q.model_config.qwen3_asr.max_new_tokens = 4;
  for (int i = 0; i < repeats; ++i) {
    bool failed = false;
    if (mode == "online") {
      SherpaOnnxOnlineRecognizerConfig c{};
      c.model_config.transducer.encoder = bad.c_str();
      c.model_config.transducer.decoder = bad.c_str();
      c.model_config.transducer.joiner = bad.c_str();
      c.model_config.tokens = tokens.c_str();
      c.model_config.num_threads = 1;
      const auto* p = SherpaOnnxCreateOnlineRecognizer(&c);
      failed = p == nullptr;
      SherpaOnnxDestroyOnlineRecognizer(p);
    } else if (mode == "qwen" || mode == "shared" || mode == "structure") {
      auto c = q;
      c.model_config.qwen3_asr.conv_frontend = mode == "structure" ? wrong.c_str() : bad.c_str();
      if (mode == "qwen") {
        const auto* p = SherpaOnnxCreateOfflineRecognizer(&c);
        failed = p == nullptr;
        SherpaOnnxDestroyOfflineRecognizer(p);
      } else {
        const auto* p = SherpaOnnxCreateOfflineQwen3ASRSharedModel(&c);
        failed = p == nullptr;
        SherpaOnnxDestroyOfflineQwen3ASRSharedModel(p);
      }
    } else if (mode == "tokenizer") {
      auto c = q;
      const auto* model = SherpaOnnxCreateOfflineQwen3ASRSharedModel(&c);
      if (!model) return 3;
      c.model_config.qwen3_asr.tokenizer = bad_tokenizer.c_str();
      const auto* p = SherpaOnnxCreateOfflineRecognizerWithSharedQwen3ASRModel(&c, model);
      failed = p == nullptr;
      SherpaOnnxDestroyOfflineRecognizer(p);
      SherpaOnnxDestroyOfflineQwen3ASRSharedModel(model);
    } else if (mode == "vad") {
      SherpaOnnxVadModelConfig c{};
      c.silero_vad.model = bad.c_str();
      c.sample_rate = 16000;
      c.num_threads = 1;
      const auto* p = SherpaOnnxCreateVoiceActivityDetector(&c, 2);
      failed = p == nullptr;
      SherpaOnnxDestroyVoiceActivityDetector(p);
    } else if (mode == "embedding") {
      SherpaOnnxSpeakerEmbeddingExtractorConfig c{};
      c.model = bad.c_str(); c.num_threads = 1;
      const auto* p = SherpaOnnxCreateSpeakerEmbeddingExtractor(&c);
      failed = p == nullptr;
      SherpaOnnxDestroySpeakerEmbeddingExtractor(p);
    } else if (mode == "punctuation") {
      SherpaOnnxOfflinePunctuationConfig c{};
      c.model.ct_transformer = bad.c_str(); c.model.num_threads = 1;
      const auto* p = SherpaOnnxCreateOfflinePunctuation(&c);
      failed = p == nullptr;
      SherpaOnnxDestroyOfflinePunctuation(p);
    } else return 2;
    if (!failed) return 4;
    // 每次失败后均加载有效的小型 ONNX fixture，验证同一进程可继续构造。
    const auto* model = SherpaOnnxCreateOfflineQwen3ASRSharedModel(&q);
    if (!model) return 5;
    const auto* recognizer = SherpaOnnxCreateOfflineRecognizerWithSharedQwen3ASRModel(&q, model);
    if (!recognizer) return 6;
    SherpaOnnxDestroyOfflineRecognizer(recognizer);
    SherpaOnnxDestroyOfflineQwen3ASRSharedModel(model);
    SherpaOnnxVadModelConfig v{};
    v.silero_vad.model = vad_path.c_str(); v.sample_rate = 16000; v.num_threads = 1;
    const auto* vad = SherpaOnnxCreateVoiceActivityDetector(&v, 2);
    if (!vad) return 7;
    SherpaOnnxDestroyVoiceActivityDetector(vad);
  }
  std::cout << mode << ": " << repeats << " failure/recovery cycles passed\n";
}
