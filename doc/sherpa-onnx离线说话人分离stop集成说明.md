# sherpa-onnx 离线说话人分离 stop 集成说明

本文说明 `sherpa-onnx` 当前工程中，`offline speaker diarization` 新增的 stop 能力应如何由集成方调用。

适用范围：
- `sherpa-onnx/csrc/offline-speaker-diarization.h`
- `sherpa-onnx/c-api/c-api.h`

不包含：
- Java 路径
- Python 路径
- 上层 `libscribe` 的业务编排

## 1. 新增语义

这次改造后，`offline speaker diarization` 的停止语义是：

- 通过 `Process(..., callback, ...)` 或 `ProcessWithCallback(..., callback, ...)` 的 callback 发起停止
- `callback(...) == 0` 表示继续执行
- `callback(...) != 0` 表示请求停止
- stop 为 cooperative cancel，不是强杀线程
- 停止粒度是阶段边界级：
  - segmentation 当前 chunk 完成后可停
  - embedding 当前段完成后可停
  - clustering 只在进入前/结束后检查

停止后：
- 不再继续后续阶段
- 返回结果对象
- 结果对象可被识别为 `stopped`
- 不应把 stopped 当成普通失败，也不应当成正常空结果

## 2. C++ API 调用方式

头文件入口：
- `sherpa-onnx/csrc/offline-speaker-diarization.h`
- `sherpa-onnx/csrc/offline-speaker-diarization-result.h`

### 2.1 基本调用

```cpp
#include "sherpa-onnx/csrc/offline-speaker-diarization.h"

std::atomic<bool> stop_requested = false;

auto callback = [&stop_requested](int32_t processed,
                                  int32_t total,
                                  void * /*arg*/) -> int32_t {
  return stop_requested.load() ? 1 : 0;
};

sherpa_onnx::OfflineSpeakerDiarization sd(config);

auto result = sd.Process(samples.data(), samples.size(), callback, nullptr);

if (result.IsStopped()) {
  // 用户主动停止
  return;
}

auto segments = result.SortByStartTime();
```

### 2.2 集成建议

- `Process()` 应放在工作线程执行
- UI 线程或控制线程只负责修改外部 stop 标志
- callback 内不要做阻塞操作
- callback 内优先只做两件事：
  - 更新进度
  - 判断是否 stop

推荐写法：

```cpp
struct ProgressContext {
  std::atomic<bool> *stop_requested;
  std::atomic<int32_t> processed{0};
  std::atomic<int32_t> total{0};
};

auto callback = [](int32_t processed, int32_t total, void *arg) -> int32_t {
  auto *ctx = static_cast<ProgressContext *>(arg);
  ctx->processed = processed;
  ctx->total = total;
  return ctx->stop_requested->load() ? 1 : 0;
};
```

## 3. C API 调用方式

头文件入口：
- `sherpa-onnx/c-api/c-api.h`

相关接口：
- `SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback()`
- `SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg()`
- `SherpaOnnxOfflineSpeakerDiarizationResultIsStopped()`

### 3.1 基本调用

```c
static int32_t MyProgressCallback(int32_t processed,
                                  int32_t total,
                                  void *arg) {
  const volatile int32_t *stop_requested = (const volatile int32_t *)arg;
  return *stop_requested ? 1 : 0;
}

volatile int32_t stop_requested = 0;

const SherpaOnnxOfflineSpeakerDiarizationResult *result =
    SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
        sd, samples, n, MyProgressCallback, (void *)&stop_requested);

if (SherpaOnnxOfflineSpeakerDiarizationResultIsStopped(result)) {
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
  return;
}

int32_t num_segments =
    SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);

const SherpaOnnxOfflineSpeakerDiarizationSegment *segments =
    SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

/* 使用完成后释放 */
SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
```

### 3.2 正确判定 stopped

集成方必须优先判断：

```c
SherpaOnnxOfflineSpeakerDiarizationResultIsStopped(result)
```

不要使用下面这些方式替代 stopped 判定：

- `num_segments == 0`
- `segments == NULL`
- “没有分离结果就认为是 stop”

原因是：
- 正常输入下本来就可能没有可用 speaker 结果
- stopped 和 empty result 现在是两种不同语义

## 4. 推荐调用时序

推荐时序如下：

1. 主线程或工作线程调用 `ProcessWithCallback()`
2. 控制线程在需要停止时设置外部 stop 标志
3. callback 读取 stop 标志并返回非 0
4. diarization 在最近的安全边界停止
5. `ProcessWithCallback()` 正常返回
6. 集成方调用 `ResultIsStopped()` 判断是否为用户停止
7. 无论 stopped 与否，都按正常路径释放 result / segment 资源

## 5. 行为边界

集成方需要明确以下几点：

- stop 不保证“立刻返回”
- stop 不会打断单次 ONNX `Run()`
- 最长可能等待当前 chunk 或当前 embedding 计算结束
- clustering 一旦开始，本次版本不会在其中途打断

这属于设计预期，不是异常行为。

## 6. 返回结果处理建议

建议上层统一按以下逻辑处理：

- `result == NULL`
  说明创建或调用失败，按错误处理
- `ResultIsStopped(result) != 0`
  说明是用户主动停止，按 stopped 处理
- 否则
  按正常 diarization 结果处理

推荐不要把 stopped 记录为 runtime failure。

## 7. 对上层 stop 设计的建议

如果上层还有 ASR、写文件、后处理等并行链路，建议：

- 整个任务共享一个 stop token
- diarization callback 只读取该 token
- 其他链路也读取同一个 token
- 最终由上层统一把 stopped 映射成自己的业务状态码

这样可以保证：
- stop 入口只有一个
- 多条链路语义一致
- 资源收尾路径更容易统一
