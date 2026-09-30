# 固定 zlib 算法与版本，静态链接并隐藏符号，不增加 SDK 动态库依赖。
include(FetchContent)
FetchContent_Declare(qwen_zlib
  URL https://zlib.net/fossils/zlib-1.3.1.tar.gz
  URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23)
FetchContent_GetProperties(qwen_zlib)
if(NOT qwen_zlib_POPULATED)
  FetchContent_Populate(qwen_zlib)
endif()
add_library(sherpa-qwen-zlib STATIC
  ${qwen_zlib_SOURCE_DIR}/adler32.c ${qwen_zlib_SOURCE_DIR}/crc32.c
  ${qwen_zlib_SOURCE_DIR}/compress.c ${qwen_zlib_SOURCE_DIR}/deflate.c
  ${qwen_zlib_SOURCE_DIR}/trees.c ${qwen_zlib_SOURCE_DIR}/zutil.c)
target_include_directories(sherpa-qwen-zlib PUBLIC ${qwen_zlib_SOURCE_DIR})
target_compile_definitions(sherpa-qwen-zlib PRIVATE Z_PREFIX)
set_target_properties(sherpa-qwen-zlib PROPERTIES
  POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
install(TARGETS sherpa-qwen-zlib ARCHIVE DESTINATION lib)
