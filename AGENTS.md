# Repository Guidelines

## Project Structure & Module Organization
仓库以多语言 SDK 和示例为主。核心实现位于 `sherpa-onnx/`：`csrc/` 是 C++ 主体，`c-api/`、`python/`、`jni/`、`rust/` 等提供绑定。仓库根目录下的 `c-api-examples/`、`go-api-examples/`、`dart-api-examples/`、`rust-api-examples/`、`nodejs-examples/`、`wasm/`、`android/`、`flutter/`、`harmony-os/` 用于平台示例与集成验证。脚本和模型转换工具集中在 `scripts/`，交叉编译配置在 `toolchains/`。

## Build, Test, and Development Commands
- `cmake -S . -B build -DSHERPA_ONNX_ENABLE_TESTS=ON`：配置默认 C++ 构建。
- `cmake -S . -B build -DSHERPA_ONNX_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON`：启用 Python 绑定。
- `cmake --build build -j`：编译当前配置。
- `ctest --test-dir build --output-on-failure`：运行已注册测试。
- `ctest --test-dir build -R test_offline_recognizer_py`：按名称运行单个 Python 测试。
- `make -C c-api-examples`、`cargo build --manifest-path rust-api-examples/Cargo.toml`：构建对应语言示例。

## Coding Style & Naming Conventions
C++/Java 代码遵循 `.clang-format`，基于 Google 风格；Java 使用 4 空格缩进，C++ 保持现有文件风格。Python 遵循 `.flake8`，最大行长 120。新增 C++ 文件优先使用 kebab-case，如 `offline-recognizer.cc`；Python 测试使用 `test_*.py`；C++ 测试使用 `*-test.cc`。避免无关的大规模格式化。

## Testing Guidelines
根工程通过 CTest 调度测试；`sherpa-onnx/python/tests/` 中的 `unittest` 脚本会注册为 `*_py` 测试。提交前至少运行受影响模块的最小测试集；修改公共 API、解码流程或绑定层时，补充一个可复现问题的测试或示例命令。若测试依赖外部模型，请在 PR 描述中写明模型来源与路径约定。

## Commit & Pull Request Guidelines
历史提交以简短命令式标题为主，例如 `Fix ...`、`Update ...`、`Release v1.12.38`、`CI: ...`。延续这种风格，首行概括单一变更；必要时加范围，如 `python: fix recognizer config`。PR 应包含：变更摘要、影响目录、验证命令、关联 issue；涉及 UI、移动端或 WebAssembly 示例时，附截图或运行结果。

## Configuration Tips
常用开关包括 `SHERPA_ONNX_ENABLE_TESTS`、`SHERPA_ONNX_ENABLE_PYTHON`、`SHERPA_ONNX_ENABLE_JNI`、`SHERPA_ONNX_ENABLE_C_API`、`SHERPA_ONNX_ENABLE_WASM`。修改版本号时，留意根目录 `CMakeLists.txt`、`CHANGELOG.md` 与 `new-release.sh` 的同步更新。
