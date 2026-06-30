// sherpa-onnx/csrc/run-log.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/run-log.h"

#include <errno.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#include <cstring>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace sherpa_onnx {
namespace {

struct RunLogState {
  std::mutex mutex;
  FILE *file = nullptr;
  std::string path;
  std::string last_error;
};

RunLogState &GetRunLogState() {
  static RunLogState state;
  return state;
}

std::string GetErrnoMessage(const char *path) {
  std::string msg = "Failed to open run log file";
  if (path != nullptr && path[0] != '\0') {
    msg += ": ";
    msg += path;
  }
  if (errno != 0) {
    msg += ": ";
    msg += std::strerror(errno);
  }
  return msg;
}

#if defined(_WIN32)
std::wstring Utf8ToWide(const char *s) {
  if (s == nullptr) return {};

  int32_t size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1,
                                     nullptr, 0);
  if (size <= 0) {
    size = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (size <= 0) return {};

    std::wstring ans(size, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s, -1, &ans[0], size);
    if (!ans.empty() && ans.back() == L'\0') ans.pop_back();
    return ans;
  }

  std::wstring ans(size, L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, &ans[0], size);
  if (!ans.empty() && ans.back() == L'\0') ans.pop_back();
  return ans;
}

FILE *OpenRunLogFile(const char *path, std::string *error) {
  std::wstring wide_path = Utf8ToWide(path);
  if (wide_path.empty()) {
    if (error != nullptr) {
      *error = "Failed to convert run log path to UTF-16: ";
      *error += path == nullptr ? "" : path;
    }
    return nullptr;
  }

  FILE *file = _wfopen(wide_path.c_str(), L"ab");
  if (file == nullptr && error != nullptr) {
    *error = GetErrnoMessage(path);
  }
  return file;
}
#else
FILE *OpenRunLogFile(const char *path, std::string *error) {
  FILE *file = fopen(path, "ab");
  if (file == nullptr && error != nullptr) {
    *error = GetErrnoMessage(path);
  }
  return file;
}
#endif

std::string VFormat(const char *fmt, va_list args) {
  if (fmt == nullptr) return "(null)";

  char buf[4096];
  va_list tmp;
  va_copy(tmp, args);
  int32_t n = vsnprintf(buf, sizeof(buf), fmt, tmp);
  va_end(tmp);

  if (n < 0) {
    return fmt;
  }

  if (static_cast<size_t>(n) < sizeof(buf)) {
    return std::string(buf, n);
  }

  std::vector<char> large(static_cast<size_t>(n) + 1);
  va_copy(tmp, args);
  n = vsnprintf(large.data(), large.size(), fmt, tmp);
  va_end(tmp);

  if (n < 0) {
    return fmt;
  }

  return std::string(large.data(), static_cast<size_t>(n));
}

void WriteToRunLogLocked(RunLogState *state, const char *text, size_t size) {
  if (state == nullptr || state->file == nullptr || text == nullptr ||
      size == 0) {
    return;
  }

  size_t n = fwrite(text, 1, size, state->file);
  if (n != size) {
    state->last_error = "Failed to write run log file";
    if (!state->path.empty()) {
      state->last_error += ": ";
      state->last_error += state->path;
    }
  }
  fflush(state->file);
}

}  // namespace

bool SetRunLogFilePath(const char *path) {
  if (path == nullptr || path[0] == '\0') {
    ClearRunLogFilePath();
    return true;
  }

  std::string error;
  FILE *file = OpenRunLogFile(path, &error);
  RunLogState &state = GetRunLogState();

  std::lock_guard<std::mutex> lock(state.mutex);
  if (file == nullptr) {
    state.last_error = error;
    return false;
  }

  if (state.file != nullptr) {
    fclose(state.file);
  }

  state.file = file;
  state.path = path;
  state.last_error.clear();
  return true;
}

void ClearRunLogFilePath() {
  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.file != nullptr) {
    fclose(state.file);
    state.file = nullptr;
  }
  state.path.clear();
  state.last_error.clear();
}

bool IsRunLogFileEnabled() {
  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.file != nullptr;
}

std::string GetLastRunLogError() {
  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.last_error;
}

void WriteRunLog(const char *text) {
  if (text == nullptr) return;

  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  WriteToRunLogLocked(&state, text, std::strlen(text));
}

void WriteRunLogLine(const char *text) {
  if (text == nullptr) text = "";

  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  WriteToRunLogLocked(&state, text, std::strlen(text));
  WriteToRunLogLocked(&state, "\n", 1);
}

void WriteRunLogLine(const std::string &text) {
  WriteRunLogLine(text.c_str());
}

void WriteRunLogOrStdout(const char *text) {
  if (text == nullptr) return;

  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.file != nullptr) {
    WriteToRunLogLocked(&state, text, std::strlen(text));
  } else {
    fputs(text, stdout);
  }
}

void WriteRunLogOrStderr(const char *text) {
  if (text == nullptr) return;

  RunLogState &state = GetRunLogState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.file != nullptr) {
    WriteToRunLogLocked(&state, text, std::strlen(text));
  } else {
    fputs(text, stderr);
  }
}

void WriteRunLogOrStdoutPrintf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::string s = VFormat(fmt, args);
  va_end(args);
  WriteRunLogOrStdout(s.c_str());
}

void WriteRunLogOrStderrPrintf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::string s = VFormat(fmt, args);
  va_end(args);
  WriteRunLogOrStderr(s.c_str());
}

}  // namespace sherpa_onnx
