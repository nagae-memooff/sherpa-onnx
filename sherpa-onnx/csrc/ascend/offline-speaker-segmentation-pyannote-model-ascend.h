// sherpa-onnx/csrc/ascend/offline-speaker-segmentation-pyannote-model-ascend.h

#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_ASCEND_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-meta-data.h"

namespace sherpa_onnx {

// AscendCL backend for the production pyannote segmentation 3.0 model.
// The first implementation intentionally supports only a static batch-1 OM.
class OfflineSpeakerSegmentationPyannoteModelAscend {
 public:
  explicit OfflineSpeakerSegmentationPyannoteModelAscend(
      const OfflineSpeakerSegmentationModelConfig &config);

  ~OfflineSpeakerSegmentationPyannoteModelAscend();

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const;

  /**
   * @param x A float32 tensor of shape (1, 1, 160000).
   * @return A float32 tensor of shape (1, 589, 7).
   */
  Ort::Value Forward(Ort::Value x) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_ASCEND_H_
