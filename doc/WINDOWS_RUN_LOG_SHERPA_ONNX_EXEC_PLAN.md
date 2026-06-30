# Windows run_log_file_path 改造执行计划（sherpa-onnx fork）

## 背景

scribe 本工程在 Windows GUI 程序中使用 `run_log_file_path` 时，不能依赖 stdout/stderr 的 `_dup2` 重定向。我们已经决定：Windows 上改为显式文件 sink，只捕获本工程和我们 fork 维护的 `sherpa-onnx` 中显式走日志入口的输出；其他第三方库直接写 stdout/stderr 的内容不强制捕获。

sherpa-onnx 是本工程的子模块和 fork 维护版本。当前本工程主要通过 sherpa-onnx C API 使用说话人相关能力：

- `SherpaOnnxOfflineSpeakerDiarizationConfig` / `SherpaOnnxCreateOfflineSpeakerDiarization()`
- `SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback()`
- `SherpaOnnxSpeakerEmbeddingExtractorConfig` / `SherpaOnnxCreateSpeakerEmbeddingExtractor()`
- clustering 路径中的 `fast-clustering.cc`

静态代码分析显示，sherpa-onnx 已经有自己的日志宏体系：

- `sherpa-onnx/csrc/log.h`
- `SHERPA_ONNX_LOGE(...)`
- `SHERPA_ONNX_LOG(...)`

同时，我们 fork 中 `sherpa-onnx/csrc/fast-clustering.cc` 有直接 `printf()` 调试输出，例如：

- `use pyannote like.`
- `not use pyannote like.`
- `[cluster distances] ...`

这些输出是 scribe SDK 用户关心的运行诊断信息，Windows GUI 下应能写入 scribe 传入的同一个 `run_log_file_path`。

## 目标

1. sherpa-onnx 新增独立的运行日志文件 sink，不依赖 stdout/stderr 重定向。
2. 提供稳定 C API，让 scribe 本工程传入同一个 `run_log_file_path`。
3. 未设置 run log 文件路径时，sherpa-onnx 保持现有行为：日志宏和 printf 仍按原先方式输出到 stderr/stdout。
4. 设置 run log 文件路径时，sherpa-onnx 中我们控制的日志入口和已知直接输出点追加写入该文件。
5. 文件路径处理遵循同一个协议：
   - 调用方原则上负责创建文件和权限。
   - sherpa-onnx 收到非空路径时必须尝试 append 打开。
   - 文件不存在则尝试创建。
   - 打开/创建失败返回错误，不静默降级。
6. 多线程写入必须安全。

## 与 scribe 的共享协议

### API 协议

建议新增 C API：

```c
SHERPA_ONNX_API int32_t SherpaOnnxSetRunLogFilePath(const char *path);
SHERPA_ONNX_API void SherpaOnnxClearRunLogFilePath(void);
```

语义：

- `path == nullptr` 或 `path[0] == '\0'`：关闭文件 sink，等价于 `SherpaOnnxClearRunLogFilePath()`，返回 `1`。
- 非空路径：
  - 复制路径到 sherpa-onnx 内部全局状态。
  - 以 append 模式打开文件。
  - 文件不存在时尝试创建。
  - 打开失败返回 `0`。
  - 打开成功返回 `1`。
- 该设置是进程级全局设置，对后续 sherpa-onnx 日志输出生效。
- 调用方只需保证 `path` 指针在函数调用期间有效；sherpa-onnx 必须复制路径。

### 错误暴露

第一阶段 API 只返回 `0/1`。如果希望让 scribe 拿到更详细错误，可追加第二个 API：

```c
SHERPA_ONNX_API const char *SherpaOnnxGetLastRunLogError(void);
```

约定：

- 返回内部线程安全或进程级保存的最近错误字符串。
- 字符串由 sherpa-onnx 持有，调用方不释放。
- 如果不实现该 API，scribe 侧只能报通用错误：`Failed to configure sherpa-onnx run log file: <path>`。

### 版本配套与 ABI 注意

本次改造不考虑新版 scribe 搭配旧版 sherpa-onnx，也不考虑旧版 scribe 搭配新版 sherpa-onnx。版本必须成对升级：

1. 现有 scribe 项目继续引用现有 sherpa-onnx 库。
2. 新版 scribe 必须引用已经实现 run log C API 的新版 sherpa-onnx 库。
3. sherpa-onnx 不需要为旧版 scribe 提供兼容垫片。
4. 如果头文件和 DLL / dylib / so 不匹配，视为集成错误，由发布和集成流程保证一致。

即使不做新旧混用兼容，也不建议把字段直接追加到 `SherpaOnnxOfflineSpeakerDiarizationConfig` 或 `SherpaOnnxSpeakerEmbeddingExtractorConfig` 里作为第一方案，原因：

1. sherpa-onnx C API config 结构体被多个语言绑定和示例使用，追加字段会扩大调用方重新生成绑定的范围。
2. scribe 当前已经遇到过头文件和 DLL 不匹配导致字段错位的问题，减少 config 结构体变更仍然更稳妥。
3. 全局 setter 协议可以避免改动多个 config，且更容易在 scribe 运行开始阶段统一同步。

如果未来确实需要每实例日志路径，再另行设计带 `size` 字段的新 config v2，不在本计划范围内。

## sherpa-onnx 改造方案

### 1. 新增 run log sink 模块

建议新增文件：

- `sherpa-onnx/csrc/run-log.h`
- `sherpa-onnx/csrc/run-log.cc`

建议内部接口：

```cpp
namespace sherpa_onnx {

bool SetRunLogFilePath(const char *path);
void ClearRunLogFilePath();
bool IsRunLogFileEnabled();
std::string GetLastRunLogError();

void WriteRunLog(const char *text);
void WriteRunLogLine(const char *text);
void WriteRunLogLine(const std::string &text);

} // namespace sherpa_onnx
```

行为：

- 未启用文件 sink 时，`WriteRunLog*` 不直接输出；由调用点决定是否继续走原 stdout/stderr。
- 启用文件 sink 时，`WriteRunLog*` 追加写文件并 flush 或按策略缓冲。
- 写入加 `std::mutex`。
- 使用 append 模式，不截断已有文件。

### 2. Windows 和非 Windows 实现

虽然该 sink 可以全平台可用，但本计划重点是 Windows。实现建议：

#### Windows

- 避免依赖 stdout/stderr。
- 优先支持 Unicode 路径：
  - 可使用 `std::filesystem::path` + `std::ofstream(path, std::ios::app)`；
  - 或使用 `_wfopen`。
- 打开失败时保存错误到 `last_error`。

#### 非 Windows

- 可复用同一个 `std::ofstream` append 实现。
- 未设置路径时完全保持原行为。
- 设置路径时也允许 sherpa-onnx 显式日志入口写文件，这不影响 scribe 非 Windows 的 stdout/stderr 重定向。

### 3. 暴露 C API

在以下文件加入声明和实现：

- `sherpa-onnx/c-api/c-api.h`
- `sherpa-onnx/c-api/c-api.cc`
- 如有导出符号列表，也需要同步：
  - `sherpa-onnx/c-api/sherpa-onnx-symbols-c.lds`
  - `sherpa-onnx/c-api/sherpa-onnx-symbols-c.exp`

建议实现：

```cpp
int32_t SherpaOnnxSetRunLogFilePath(const char *path) {
  return sherpa_onnx::SetRunLogFilePath(path) ? 1 : 0;
}

void SherpaOnnxClearRunLogFilePath(void) {
  sherpa_onnx::ClearRunLogFilePath();
}
```

如实现错误查询：

```cpp
const char *SherpaOnnxGetLastRunLogError(void) {
  static thread_local std::string s;
  s = sherpa_onnx::GetLastRunLogError();
  return s.c_str();
}
```

是否用 `thread_local` 或全局字符串由实现决定，但必须保证返回指针在下一次调用前有效。

### 4. 接入 sherpa-onnx 日志宏

当前 `sherpa-onnx/csrc/log.h` 的 `Logger` 构造/析构中直接 `fprintf(stderr, ...)`。建议将格式化输出集中到一个内部函数：

```cpp
void SherpaOnnxLogWriteStderr(const std::string &text);
```

语义：

- 未启用 run log 文件：写 `stderr`，保持现有行为。
- 启用 run log 文件：写 run log 文件；是否同时写 stderr 可由条件决定。为了避免 Windows GUI 问题，Windows 下建议只写文件；非 Windows 可继续写 stderr。

更直接的方式是在 `log.h` 中引入 `run-log.h` 并替换 `fprintf(stderr, ...)`。注意 `log.h` 是头文件，避免引入过重依赖或静态初始化问题。可以只声明轻量函数：

```cpp
void WriteLogToStderrOrRunLog(const char *s);
```

### 5. 替换 fork 中已知直接 printf 输出

重点文件：

- `sherpa-onnx/csrc/fast-clustering.cc`

当前直接 `printf()` 输出 cluster 调试信息。建议改成：

```cpp
SHERPA_ONNX_RUN_LOG_PRINTF("use pyannote like.\n");
```

或：

```cpp
sherpa_onnx::WriteRunLogOrStdout("use pyannote like.\n");
```

语义：

- 未启用 run log 文件：仍写 stdout，保持现有行为。
- 启用 run log 文件：写文件，Windows 下不依赖 stdout。

建议新增 printf 风格工具，便于少量替换：

```cpp
void WriteRunLogOrStdoutPrintf(const char *fmt, ...);
void WriteRunLogOrStderrPrintf(const char *fmt, ...);
```

实现注意：

- 使用 `va_list` + `vsnprintf` 先格式化到 `std::string`。
- 长日志需要处理 buffer 扩容。
- 写文件时加锁。

### 6. 接入范围

第一阶段范围只覆盖 scribe 实际依赖路径：

1. `sherpa-onnx/csrc/log.h` 的 `SHERPA_ONNX_LOG*` 输出。
2. `sherpa-onnx/csrc/fast-clustering.cc` 中我们 fork 增加的 `printf()`。
3. speaker diarization / embedding extractor 创建和运行过程中直接使用 `SHERPA_ONNX_LOGE` 的错误信息。

暂不要求覆盖：

- sherpa-onnx 示例程序 `cxx-api-examples/`
- JNI 示例输出
- wasm 示例输出
- 其他第三方依赖直接 stdout/stderr

### 7. 与 scribe 的调用顺序

scribe 本工程应在每次 `scribe_run_pipeline()` 中、任何 sherpa-onnx 对象创建前同步路径：

1. `run_log_file_path` 非空：
   - scribe 先完成自己的文件可写校验。
   - scribe 调用 `SherpaOnnxSetRunLogFilePath(path)`。
   - 返回 `0` 则 scribe 返回 `SCRIBE_ERR_RUNTIME`。
2. `run_log_file_path` 为空：
   - scribe 调用 `SherpaOnnxClearRunLogFilePath()`，避免同进程上一次调用留下全局路径。

sherpa-onnx 不需要知道 scribe 的 handle 或请求生命周期。

## 文件存在和权限策略

路径非空时，sherpa-onnx 的 `SetRunLogFilePath()` 必须执行以下流程：

1. 尝试 append 打开路径。
2. 如果文件不存在：
   - append 打开通常会创建文件。
   - 创建失败则保存错误并返回 `false`。
3. 如果文件存在但不可写：
   - 打开失败，保存错误并返回 `false`。
4. 打开成功：
   - 写入一个空串或 flush，确保权限真实可用。
   - 保存文件句柄/stream。
   - 返回 `true`。

不做目录自动创建。父目录不存在视为失败。

## 测试计划

### C API 单元测试或小程序

1. `SherpaOnnxSetRunLogFilePath(nullptr)` 返回成功，日志仍走原 stderr/stdout。
2. `SherpaOnnxSetRunLogFilePath("")` 返回成功，日志仍走原 stderr/stdout。
3. 传入已存在可写文件，返回成功，`SHERPA_ONNX_LOGE` 输出写入文件。
4. 传入不存在但父目录可写文件，返回成功并创建文件。
5. 传入父目录不存在路径，返回失败。
6. 传入不可写路径，返回失败。
7. 多线程并发写入不崩溃、不产生明显交错半行。

### 与 scribe 联调

1. Windows GUI 程序传入 `run_log_file_path`，scribe 不再因 stdout/stderr 重定向失败返回 `-5`。
2. 启用 speaker diarization，确认 sherpa-onnx 的日志宏输出进入同一文件。
3. 触发 `SHERPA_CLUSTER_DEBUG_DIST`，确认 `fast-clustering.cc` 的 cluster distance 输出进入同一文件。
4. 不传 `run_log_file_path`，确认原有 stdout/stderr 行为保持。

## 风险与边界

1. 全局 sink 会影响同进程所有 sherpa-onnx 调用。scribe 每次运行开始显式 set/clear 可降低状态残留风险。
2. 如果多个上层组件同时使用 sherpa-onnx 并设置不同日志文件，最后一次 set 生效。第一阶段接受该限制。
3. `log.h` 是广泛包含的头文件，改造时要避免引入复杂依赖造成编译膨胀或循环 include。
4. Windows Unicode 路径需要特别验证。
5. 不承诺捕获未接入 run-log 工具的第三方库直接 stdout/stderr。

## 交付物

1. `run-log.h` / `run-log.cc`。
2. `SherpaOnnxSetRunLogFilePath()` / `SherpaOnnxClearRunLogFilePath()` C API。
3. 可选 `SherpaOnnxGetLastRunLogError()` C API。
4. `SHERPA_ONNX_LOG*` 输出接入 run log sink。
5. `fast-clustering.cc` 直接 `printf()` 替换为公共输出入口。
6. Windows 和非 Windows 基础验证记录。
