// sherpa-onnx/csrc/ascend/speaker-embedding-extractor-model-ascend.h

#ifndef SHERPA_ONNX_CSRC_ASCEND_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_ASCEND_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model-meta-data.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

// AscendCL backend for the production WeSpeaker ResNet34 model. The model is
// compiled as an OM with input feats:[1,16~1000,80] and output embs:[1,256].
class SpeakerEmbeddingExtractorModelAscend {
 public:
  explicit SpeakerEmbeddingExtractorModelAscend(
      const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractorModelAscend();

  const SpeakerEmbeddingExtractorModelMetaData &GetMetaData() const;

  /**
   * @param x A float32 tensor of shape (1, T, 80), 16 <= T <= 1000.
   * @return A float32 tensor of shape (1, 256).
   */
  Ort::Value Compute(Ort::Value x) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_ASCEND_H_
