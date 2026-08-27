# File description

Please download test wave files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models

## 0-four-speakers-zh.wav

It is recorded by @csukuangfj

## 1-two-speakers-en.wav

This file is from
https://github.com/pengzhendong/pyannote-onnx/blob/master/data/test_16k.wav
and it contains speeches from two speakers.

Note that we have renamed it from `test_16k.wav` to `1-two-speakers-en.wav`


## 2-two-speakers-en.wav
This file is from
https://huggingface.co/spaces/Xenova/whisper-speaker-diarization

Note that the original file is `./fcf059e3-689f-47ec-a000-bdace87f0113.mp4`.
We use the following commands to convert it to `2-two-speakers-en.wav`.

```bash
ffmpeg -i ./fcf059e3-689f-47ec-a000-bdace87f0113.mp4 -ac 1 -ar 16000 ./2-two-speakers-en.wav
```

## 3-two-speakers-en.wav

This file is from
https://aws.amazon.com/blogs/machine-learning/deploy-a-hugging-face-pyannote-speaker-diarization-model-on-amazon-sagemaker-as-an-asynchronous-endpoint/

Note that the original file is `ML16091-Audio.mp3`. We use the following
commands to convert it to `3-two-speakers-en.wav`


```bash
sox ML16091-Audio.mp3 -r 16k 3-two-speakers-en.wav
```

## Ascend 310P static-batch model

CANN 9.1 requires a static graph and 4-D inputs for InstanceNormalization on
Ascend 310P. Prepare the existing Pyannote segmentation 3.0 ONNX model without
overwriting it:

```bash
python3 prepare-ascend-310p-b1.py \
  sherpa-onnx-pyannote-segmentation-3-0.onnx \
  sherpa-onnx-pyannote-segmentation-3-0_atc_b4.onnx \
  --batch-size 4
```

The script supports static batch sizes from 1 through 16. It fixes the input to
`[B,1,160000]`, replaces the zero LSTM initial states with `[2,B,128]`
constants, and wraps each 3-D InstanceNormalization input as 4-D. For the
production model, ONNX Runtime produces exactly the same output before and
after this preprocessing.

Compile the prepared model for Atlas 300I Duo / Ascend 310P3:

```bash
atc \
  --framework=5 \
  --model=sherpa-onnx-pyannote-segmentation-3-0_atc_b4.onnx \
  --output=sherpa-onnx-pyannote-segmentation-3-0_b4 \
  --input_format=ND \
  --input_shape="x:4,1,160000" \
  --soc_version=Ascend310P3
```

The generated OM accepts float32 `[B,1,160000]` and returns float32
`[B,589,7]`. The rewritten production model compiles with the default CANN 9.1
fusion rules. `ascend-310p-fusion-all-off.json` is retained as a conservative
fallback for compiler diagnostics.

After building sherpa-onnx with `SHERPA_ONNX_ENABLE_ASCEND_NPU=ON`, compare
the CPU ONNX output and Ascend OM output on the same 10-second audio window:

```bash
sherpa-onnx-speaker-segmentation-ascend-compare \
  sherpa-onnx-pyannote-segmentation-3-0.onnx \
  sherpa-onnx-pyannote-segmentation-3-0_b4.om \
  3-two-speakers-en.wav \
  5 \
  4
```

The last argument is the batch size. At runtime, set
`segmentation_batch_size` no higher than the static batch of the OM. A short
final batch repeats its last real item internally and returns only real outputs.
