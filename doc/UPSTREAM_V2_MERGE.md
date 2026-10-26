# v2.0.0 私有分支合并记录

## 基线与范围

私有基线为 `8f8005013f7d8ae32b8b8123ab6afb277984e553`，已包含官方
v1.13.8。此次合入官方 v2.0.0（`8abb7c8f87eae85b2e75f9a4115236c978980add`），
不包含该 tag 之后的 master 提交。Scribe 必须配套使用本 fork 的头文件、库和源码重新编译。

## 保留的私有行为

- Qwen3 共享模型 C API、ORT Session 共享与每路解码状态隔离。
- Qwen3 int8 终止 token、centered STFT reflection、周期循环检测、compression guard、
  folded text、解码终止原因与可选 token/allocator/profile 诊断。
- diarization 停止状态、分阶段计时、embedding workers/batch、segmentation batch、提前释放。
- Ascend 分段与 embedding、设备号、资源生命周期，以及私有聚类、小簇合并和阈值。
- 运行日志 C API、模型加载异常隔离、RAII 释放及依赖下载缓存。

CUDA 暂不采用上游 `2e3968f5`（#3985）的配置文件 provider 参数透传。
`session.cc`、`session.h` 保持私有基线内容，继续使用 `SessionOptionsConfig`、
`OrtCUDAProviderOptions` 和既有参数优先级。`disable_mem_pattern`、
`cuda_arena_extend_strategy`、`cuda_gpu_mem_limit` 的 Scribe 映射不变。
既有 `SessionConfig.*` 和显式 `EnableMemPattern` 支持保留。
对应的上游新 helper 测试不引入；原有 Session 测试继续执行。

这是有意暂缓的功能。合并 v2.0.0 后，未来合并上游不会自动重新应用该祖先提交；
CUDA 产品启动时应重新评估参数来源、优先级、失败语义和真机验证。

流式识别继续保留句首标点，不引入 `OnlineRecognizer::GetResult` 新增的
`RemoveLeadingPunct` 调用；公共工具函数仍保留。

## 冲突处理与依赖

C API 保留私有异常边界和 unique_ptr 生命周期实现，同时接入 CTC FST 配置等上游新增功能，
不叠加上游 raw pointer 的 delete 清理代码。offline-stream 删除私有同名 QuoteJsonString，
统一使用上游公共 helper，保留全部私有结果字段。

接受上游删除 cargs、espeak-ng-for-piper、piper-phonemize 构建片段；Scribe 继续关闭 TTS。
新增 utf8proc 2.12.0 使用私有 FetchContent 下载缓存，kaldi-native-fbank 更新为 1.22.4。
即使关闭 TTS，核心仍依赖 utf8proc；Apple 合并静态库时必须加入 libutf8proc.a 及许可证。

上游 CTC FST 配置追加到结构体，VAD/circular buffer 样本位置扩展到 int64_t。
不要混用旧头文件或旧二进制。Scribe 当前 VAD 使用 whisper.cpp，本次没有迁移 VAD 实现。

## 本机验证（2026-10-26）

Linux x86_64，Release，共享库，CPU，TTS OFF，diarization ON，tests ON。
独立构建和安装成功，执行 `ctest --test-dir tmp/v2-merge/build --output-on-failure`：
31 个测试目标通过，包括 Qwen、JSON、CTC、UTF-8、聚类、Session 和模型加载异常隔离。
text2token 的 4 个外部数据用例因未安装 sherpa-test-data 在测试内部提前返回，
不能将它们算作真实数据验证。

对私有差异逐文件核对：97 个仅私有分支修改的文件与基线一致；双方共同修改的 20 个文件
另行审查。原头文件的 182 个非 OHOS C API 函数均存在于 Linux 动态库导出表；
Scribe src 中调用的 47 个 sherpa 函数均有导出。OHOS 专用符号不属于 Linux 构建。
测试临时文件放在构建目录内，避免使用系统临时目录。

本次修改文件的 diff whitespace 检查通过；完整上游差异中包含原有 CRLF/尾空格告警，
保留上游格式，不做无关重排。Scribe SDK 与真实音频验证记录由 Scribe 仓库维护。
Apple、Windows、Android、Ascend 真机验证不能由 Linux CPU 测试代替；CUDA 真机验证暂缓。

上游 version.cc 自带的 Git SHA 字符串不代表本 fork 合并提交。产物来源须以实际 Git SHA、
Scribe 子模块引用和构建记录为准。
