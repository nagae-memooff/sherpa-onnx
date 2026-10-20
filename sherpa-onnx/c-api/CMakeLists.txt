include_directories(${PROJECT_SOURCE_DIR})
add_library(sherpa-onnx-c-api c-api.cc)
target_link_libraries(sherpa-onnx-c-api sherpa-onnx-core)
target_include_directories(sherpa-onnx-c-api PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_MAIN_LIB=1)
endif()

add_library(sherpa-onnx-cxx-api cxx-api.cc)
target_link_libraries(sherpa-onnx-cxx-api sherpa-onnx-c-api)
target_include_directories(sherpa-onnx-cxx-api PUBLIC ${PROJECT_SOURCE_DIR})

if(ANDROID OR (UNIX AND NOT APPLE))
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.lds"
  )
elseif(APPLE)
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.exp"
  )
endif()

install(
  TARGETS
    sherpa-onnx-c-api
    sherpa-onnx-cxx-api
  DESTINATION
    lib
)

install(
  FILES
    c-api.h
    cxx-api.h
  DESTINATION
    include/sherpa-onnx/c-api
)

# 每个错误场景在独立进程运行，避免回归时终止整个测试调度器。
if(SHERPA_ONNX_ENABLE_TESTS)
  add_executable(sherpa-onnx-model-loading-exception-test
    model-loading-exception-test.cc)
  target_link_libraries(sherpa-onnx-model-loading-exception-test
    PRIVATE sherpa-onnx-c-api)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME sherpa-onnx-model-loading-exception-test
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_model_loading_exceptions.py
        $<TARGET_FILE:sherpa-onnx-model-loading-exception-test>
        ${CMAKE_CURRENT_BINARY_DIR}/model-loading-exception-fixtures)
    set_tests_properties(sherpa-onnx-model-loading-exception-test
      PROPERTIES TIMEOUT 180)
  endif()
endif()
