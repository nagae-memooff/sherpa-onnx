// sherpa-onnx/csrc/run-log.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RUN_LOG_H_
#define SHERPA_ONNX_CSRC_RUN_LOG_H_

#include <string>

namespace sherpa_onnx {

bool SetRunLogFilePath(const char *path);
void ClearRunLogFilePath();
bool IsRunLogFileEnabled();
std::string GetLastRunLogError();

void WriteRunLog(const char *text);
void WriteRunLogLine(const char *text);
void WriteRunLogLine(const std::string &text);

void WriteRunLogOrStdout(const char *text);
void WriteRunLogOrStderr(const char *text);
void WriteRunLogOrStdoutPrintf(const char *fmt, ...);
void WriteRunLogOrStderrPrintf(const char *fmt, ...);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RUN_LOG_H_
