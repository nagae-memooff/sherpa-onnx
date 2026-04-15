# Repository Guidelines

## Project Structure & Module Organization
- `csrc/`: 核心 C++ 实现与可执行示例，测试源码多为 `*-test.cc`。
- `c-api/`: C API 包装层，面向外部 C 接口集成。
- `python/`: Python 绑定，`csrc/` 为扩展实现，`tests/` 为 Python 测试。
- `jni/`、`java-api/`、`kotlin-api/`: JVM 相关绑定与接口，`java-api/Makefile` 负责打包 Jar。
- `pascal-api/`: Pascal 绑定与相关资源。
- 顶层 `CMakeLists.txt` 仅在开启 `SHERPA_ONNX_ENABLE_*` 选项时加入对应子模块。

## Build, Test, and Development Commands
- `cmake -S . -B build -DSHERPA_ONNX_ENABLE_TESTS=ON -DSHERPA_ONNX_ENABLE_PYTHON=ON`：配置工程并开启测试与 Python 模块。
- `cmake --build build`：构建全部目标。
- `ctest --test-dir build`：运行 C++/Python 测试（需已启用测试）。
- `ctest --test-dir build -R test_offline_recognizer_py`：运行单个 Python 测试。
- `make -C java-api`：生成 `build/sherpa-onnx.jar`。
- `make -C java-api native`：打包包含 `resources/` 的 native Jar。

## Coding Style & Naming Conventions
- C++ 文件使用 `*.cc`/`*.h`，文件名多为中划线风格，如 `offline-recognizer-impl.cc`。
- C++ 测试命名为 `*-test.cc`；Python 测试命名为 `test_*.py`。
- Java 包路径固定为 `com.k2fsa.sherpa.onnx`，类名为 `UpperCamelCase`。
- 缩进与空格请保持与周围代码一致（C++ 多为 2 空格，Python/Java 按语言习惯 4 空格）。
- 未见统一格式化工具配置，请避免大规模重排。

## Testing Guidelines
- C++ 测试使用 GoogleTest，通过 CTest 注册（需 `SHERPA_ONNX_ENABLE_TESTS=ON`）。
- Python 测试使用 `unittest`，同样由 CTest 调度。
- 部分测试依赖外部模型或数据目录（如 `/tmp/sherpa-test-data` 或模型包），缺失时会自动跳过。

## Commit & Pull Request Guidelines
- 提交历史中既有 `feat:`/`fix(scope):`，也有 `Add/Refactor/Fix ...` 命令式标题；请保持简短、可读，并在需要时标明范围。
- PR 请包含变更摘要、测试命令与结果、相关配置开关（如 `SHERPA_ONNX_ENABLE_TTS`）、关联 Issue；涉及 API/模型兼容性变更需明确说明。

## Configuration Tips
- 常用开关：`SHERPA_ONNX_ENABLE_PYTHON`、`SHERPA_ONNX_ENABLE_JNI`、`SHERPA_ONNX_ENABLE_C_API`、`SHERPA_ONNX_ENABLE_TESTS`、`SHERPA_ONNX_ENABLE_TTS`、`SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION`。
