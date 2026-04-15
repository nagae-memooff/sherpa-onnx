# 执行计划：为 libscribe 增加平滑“停止转录”能力

## 1. 重构目的

本次重构的目标，是让当前 libscribe 交付产物（so / xcframework / 桌面端动态库）具备平滑停止转录能力。

目标边界如下：

- 停止语义是 stop，不是 pause
- 停止后不要求恢复
- 不要求“立刻中断”，接受 chunk 级 / 阶段边界级 延后停止
- 优先保证：
    - 不崩溃
    - 不出现 use-after-free
    - 不留下不可控后台线程
    - 返回明确的“已停止”状态
- sensevoice 路径不纳入本次范围
- 移动端目前不启用 sherpa-onnx 路径，因此 sherpa-onnx 停止能力主要服务于桌面端 speaker_post 场景

最终效果应为：

- 未开启说话人识别时，whisper 路径支持安全停止
- 开启说话人识别时，whisper 与 sherpa-onnx 两条路径都支持协作式停止
- 外部调用方可通过新增 stop API 发起停止请求，scribe_run() 最终返回“已停止”错误码

———

## 2. whisper.cpp 路径已经支持，当前工程的改造方式

### 2.1 改造目标

利用 whisper.cpp 已提供的 abort_callback，将外部的停止请求传入 whisper_full() 执行过程，实现 whisper 主链路的安全停止。

### 2.2 需要修改的模块

建议修改这些位置：

- include/scribe_api.h
- src/c-api.cpp
- include/scribe_core.h
- src/scribe_core.cpp
- include/transcribe.h
- src/transcribe.cpp

### 2.3 对外 C API 改造

在 include/scribe_api.h 中新增：

- 错误码：SCRIBE_ERR_STOPPED
- 函数：int scribe_stop(void* handle);

行为定义：

- scribe_stop(handle) 为线程安全函数，可由其他线程调用
- 该函数只设置“停止请求”，不做资源释放
- scribe_run() 收到停止请求后，返回 SCRIBE_ERR_STOPPED
- scribe_finalize(handle) 仍要求在 scribe_run() 返回后调用，不允许与执行中的 scribe_run() 并发销毁同一 handle

### 2.4 ScribeHandle 增加运行期状态

在 src/c-api.cpp 中扩展 ScribeHandle，至少增加：

- std::atomic<bool> stop_requested
- std::atomic<bool> running

必要时增加：

- 互斥锁或轻量状态保护
- 本次 run 的运行代次标识，避免 stop 状态污染下一次 run

### 2.5 scribe_run() / scribe_stop() 行为调整

在 src/c-api.cpp 中：

- scribe_init() 初始化运行状态
- scribe_run() 启动前清空 stop_requested
- scribe_stop() 只将 stop_requested = true
- scribe_run() 执行完成后重置 running

### 2.6 将停止信号传入 scribe_core

在 include/scribe_core.h / src/scribe_core.cpp 中，为核心执行函数增加“停止查询”能力，建议形式如下：

- 增加一个 stop 查询回调
- 或直接传入原子标志引用/指针

建议统一抽象成：

- StopToken
- 或 std::function<bool()> should_stop

这样后续 whisper 与 sherpa 路径都能共用同一套停止判定接口。

### 2.7 whisper_full() 接入 abort_callback

在 src/transcribe.cpp 的 whisper_full_with_sentence_ts() 中：

- 为 whisper_full_params 设置：
    - abort_callback
    - abort_callback_user_data

callback 内逻辑：

- 读取 stop token
- 若外部请求停止，返回 true / 中止信号
- 否则继续执行

同时需要统一处理 whisper_full() 被 abort 后的返回值：

- 不把它当成普通 runtime failure
- 转换为“本次被用户停止”的状态，最终上抛为 SCRIBE_ERR_STOPPED

### 2.8 scribe_core 中 stop 后的行为

在 src/scribe_core.cpp 中：

- 在 ASR 前检查一次 stop
- 在 ASR 完成后检查一次 stop
- stop 后不再进入：
    - JSON 保存
    - diarization 合并
    - speaker 平滑
    - 输出写文件
    - 控制台后处理展示
- 统一走“停止返回”路径

———

## 3. 启用说话人识别时，sherpa-onnx 当前不具备该能力，需要改造

### 3.1 当前问题

当前 scribe 的说话人识别链路依赖 lib/sherpa-onnx 中的 offline speaker diarization 实现。

虽然它的 Process(callback) 接口签名中 callback 返回 int32_t，但当前 pyannote 实现中：

- callback 返回值基本未被消费
- 只当作进度通知使用
- 没有形成真正的中断控制流

因此当前现状是：

- 可以“知道进度”
- 不能“通过 callback 安全停止”

### 3.2 当前 sherpa-onnx 的停止粒度限制

即使改造 sherpa-onnx，停止也很难做到“即时”：

- segmentation model 的单次 Ort::Session::Run() 是阻塞调用
- embedding model 的单次 Ort::Session::Run() 也是阻塞调用
- 所以可实现的合理停止粒度是：
    - segmentation chunk 之间
    - embedding 段之间
    - clustering 前后阶段边界

这符合本次目标：接受 chunk 级延后停止。

———

## 4. sherpa-onnx 的改造详细计划

## 4.1 改造目标

在 lib/sherpa-onnx 中为 offline speaker diarization 增加 cooperative cancel 能力，使其能够在 chunk 边界 / embedding 边界安全退出，并将“已停止”状态向上返回给 scribe。

## 4.2 改造范围

重点修改目录：

- lib/sherpa-onnx/sherpa-onnx/csrc/offline-speaker-diarization.h
- lib/sherpa-onnx/sherpa-onnx/csrc/offline-speaker-diarization-impl.h
- lib/sherpa-onnx/sherpa-onnx/csrc/offline-speaker-diarization.cc
- lib/sherpa-onnx/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
- lib/sherpa-onnx/sherpa-onnx/c-api/c-api.h
- lib/sherpa-onnx/sherpa-onnx/c-api/c-api.cc

如需把“取消”暴露到 C API，还需要同步 JNI / 其他绑定层，但本次 scribe 集成只要本地 C API 行为正确即可。

## 4.3 明确 callback 取消语义

在 sherpa-onnx 内部统一约定：

- callback(...) == 0：继续执行
- callback(...) != 0：请求停止

这个语义需要体现在：

- offline-speaker-diarization.h
- C API 头文件注释
- 相关 JNI / wrapper 注释

## 4.4 为 diarization 结果增加“停止状态”

当前 Process() 直接返回 OfflineSpeakerDiarizationResult，没有显式状态位。

建议二选一：

### 方案 A：新增状态返回结构

新增一个带状态的返回结构，例如：

- OfflineSpeakerDiarizationProcessResult
- 包含：
    - OfflineSpeakerDiarizationResult result
    - bool stopped

优点是语义清晰。
缺点是改动面更大。

### 方案 B：维持原结果结构，附加 stop 输出参数

例如在内部实现层增加：

- bool* stopped

优点是侵入较小。
建议本次优先用这一类方案，降低 fork 成本。

推荐方案：

- Process(...) 内部新增 bool* stopped
- C API 层根据 stopped 返回空结果或特定状态码

## 4.5 改造 OfflineSpeakerDiarizationPyannoteImpl::Process()

当前函数需要变成“阶段化可中止”流程：

1. 进入前先检查 stop
2. RunSpeakerSegmentationModel() 结束后检查 stop
3. ComputeEmbeddings() 结束后检查 stop
4. clustering 前检查 stop
5. clustering 后检查 stop
6. ComputeResult() 前检查 stop

一旦 stop：

- 不再继续后续阶段
- 返回“已停止”状态
- 不产出不完整的正常 diarization 结果

## 4.6 改造 RunSpeakerSegmentationModel()

当前它内部按 chunk 依次执行 ProcessChunk()。

需要改造为：

- 增加 callback / stop 检查参数
- 每完成一个 chunk 后调用 callback
- 若 callback 返回非 0，立即停止后续 chunk 处理
- 将“stopped”状态上报给外层

注意：

- 单个 ProcessChunk() 内部的 segmentation_model_.Forward() 仍不可打断
- 所以 stop 发生时，最多等当前 chunk 推理结束

这是可接受的。

## 4.7 改造 ComputeEmbeddings()

当前它已经有 callback 参数，但返回值未被消费。

需要修改为：

- 每完成一个 embedding 计算后调用 callback
- 若 callback 返回非 0：
    - 停止后续 (chunk, speaker) embedding 处理
    - 设置 stopped = true
    - 直接返回当前中止状态，而不是继续聚类

这部分是当前最直接、收益最大的改造点。

## 4.8 clustering 阶段的处理策略

FastClustering::Cluster() 当前没有 callback。

建议本次不深入改 clustering 内部，而是采用：

- clustering 前检查 stop
- 如已 stop，则不进入 clustering
- clustering 执行中不支持中断
- clustering 完成后再检查一次 stop

理由：

- clustering 通常不是最长耗时段
- 先把 chunk / embedding 级停止打通即可
- 这样可以控制 sherpa-onnx fork 改动范围

## 4.9 sherpa-onnx C API 层配套修改

在 c-api 中应把“停止”与“失败”区分开，建议：

- diarization process callback 返回非 0 时，底层过程返回“stopped”
- C API 层新增可识别的错误码或状态码

如果不想大幅改现有 sherpa C API，可退一步：

- scribe 内部直接用 csrc/c-api 的本地行为判断 stop
- scribe 自己映射成 SCRIBE_ERR_STOPPED

但从长期维护看，建议 sherpa-onnx 自己也有明确 stopped 语义。

## 4.10 sherpa-onnx 测试计划

建议至少新增这些测试：

- segmentation chunk 跑一半时 callback 返回停止，请求应安全结束
- embedding 循环中 callback 返回停止，应安全结束
- 停止后无崩溃、无悬挂线程
- 正常不停止时，结果与现有逻辑一致
- callback 始终返回 0 时，行为完全兼容旧版本

———

## 5. sherpa-onnx 改造完毕后，本工程如何集成其停止能力

## 5.1 总体思路

本工程应使用同一个 stop token同时驱动：

- whisper.cpp 路径
- sherpa-onnx diarization 路径

这样外部调用一次 scribe_stop(handle)，即可同时作用于两条并行链路。

## 5.2 scribe_core 并行链路改造

当前 src/scribe_core.cpp 中：

- whisper 主线程执行
- diarization 通过 std::async 后台执行
- 最终等待双方结束并合并结果

改造后应变成：

### 后台 diarization 任务

后台任务在调用 run_diarization() 时：

- 传入 stop callback / stop token
- sherpa-onnx 内部按新语义支持停止
- 若被停止，则返回“已停止”状态，而不是一般失败

### 主线程 whisper 任务

主线程在 whisper_full_with_sentence_ts() 中：

- 使用 abort_callback
- 一旦 stop，尽快返回 stopped 状态

### 汇总阶段

在 scribe_core.cpp 中统一处理四种结果组合：

- whisper OK + diar OK
    - 正常合并 speaker 与 transcript
- whisper STOPPED
    - 直接返回 SCRIBE_ERR_STOPPED
    - 不再做合并输出
- diar STOPPED
    - 如果本次 stop 是外部请求触发，也整体返回 SCRIBE_ERR_STOPPED
- whisper OK + diar FAIL
    - 维持现有错误策略或按产品策略决定是否降级

本次建议策略简单化：

- 只要 stop token 被置位且任一路径进入 stopped，整个 scribe_run() 返回 SCRIBE_ERR_STOPPED

## 5.3 run_diarization() 封装改造

在当前工程中需要改：

- include/diarization.h
- src/diarization.cpp

让 run_diarization() 支持接收 stop 回调或 stop token，并把它传入：

- SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(...)

同时本地 callback 需要从“只更新 spinner”改为：

- 先更新 spinner
- 再检查 stop token
- stop 时返回非 0

这样可以直接利用改造后的 sherpa-onnx 中断语义。

## 5.4 停止后的返回和资源收尾

本工程中需要统一保证：

- stop 后仍然等待后台 future 正常结束
- 不允许因为 stop 而跳过后台任务收口
- scribe_run_pipeline() 在返回前必须完成：
    - 线程收尾
    - audio 释放
    - whisper context 释放
    - diarization 资源释放
    - spinner 停止

也就是说：

- stop 不等于“立刻返回”
- stop 等于“停止继续计算，并安全收尾后返回”

这是这次设计的核心原则。

## 5.5 本工程测试计划

建议新增这些测试场景：

### 仅 whisper

- 不启用 speaker_post 时，stop 能在 whisper 中安全返回
- 返回码为 SCRIBE_ERR_STOPPED
- 不写出半成品 JSON

### whisper + diarization

- 启用 speaker_post 时，stop 后 whisper 与 diarization 都能最终收敛退出
- 不死锁
- 不崩溃
- 不发生 finalize 提前释放导致的悬挂访问

### 多平台验证

- Android / iOS：确认未启用 sherpa 路径时 stop 仅影响 whisper
- macOS / Windows：确认启用 sherpa 时 stop 可正常工作

## 5.6 对集成方的影响

调用接口层面：

- 新增 scribe_stop(void* handle)

调用模式层面：

- 需要持有 handle
- scribe_run() 需在工作线程执行
- UI/控制线程可调用 scribe_stop(handle)
- scribe_run() 返回后，再 scribe_free_response() / scribe_finalize()

不需要改动：

- scribe_config 结构体字段
- scribe_request 结构体字段
- 模型路径传参方式

———

## 6. 推荐落地顺序

建议按以下顺序实施，降低风险：

1. 先改本工程 whisper stop
2. 再改本工程 handle 生命周期与 scribe_stop() API
3. 再改 lib/sherpa-onnx 的 diarization cooperative cancel
4. 最后把 sherpa-onnx 停止能力接回 scribe_core
5. 补桌面端 Demo 或最小验证代码，验证 stop 时序

这样做的好处是：

- 第一阶段就能让移动端路径先获得 stop 能力
- sherpa-onnx 改造可以独立验证
- 最终集成时问题边界更清晰

———

## 7. 最终预期结果

改造完成后，系统应具备以下能力：

- libscribe 提供统一 stop API
- 未启用说话人识别时，whisper 可安全停止
- 启用说话人识别时，whisper 与 sherpa-onnx 可协作停止
- 停止是“平滑收尾”，不是强杀线程
- 停止粒度接受 chunk 级延迟
- 对桌面端集成方可用，对移动端无额外负担
