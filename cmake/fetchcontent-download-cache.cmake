include_guard(GLOBAL)

include(FetchContent)

set(_download_cache_default "${CMAKE_SOURCE_DIR}/.cache/downloads")
if(DEFINED ENV{DOWNLOAD_CACHE_DIR})
  set(_download_cache_default "$ENV{DOWNLOAD_CACHE_DIR}")
endif()
set(DOWNLOAD_CACHE_DIR "${_download_cache_default}" CACHE PATH
    "Directory used to cache original files downloaded by FetchContent")
unset(_download_cache_default)

function(sherpa_onnx_fetchcontent_declare content_name)
  if("${DOWNLOAD_CACHE_DIR}" STREQUAL "")
    FetchContent_Declare(${content_name} ${ARGN})
    return()
  endif()

  set(_first_url "")
  set(_reading_urls OFF)
  set(_has_url_hash OFF)
  foreach(_arg IN LISTS ARGN)
    if(_arg STREQUAL "URL")
      set(_reading_urls ON)
    elseif(_arg STREQUAL "URL_HASH")
      set(_reading_urls OFF)
      set(_has_url_hash ON)
    elseif(_reading_urls AND "${_first_url}" STREQUAL "")
      set(_first_url "${_arg}")
    endif()
  endforeach()

  if("${_first_url}" STREQUAL "")
    message(FATAL_ERROR
            "${content_name}: download cache requires a URL declaration")
  endif()
  if(NOT _has_url_hash)
    message(FATAL_ERROR
            "${content_name}: URL_HASH is required when DOWNLOAD_CACHE_DIR is enabled")
  endif()

  get_filename_component(_cache_root "${DOWNLOAD_CACHE_DIR}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  string(SHA256 _url_key "${_first_url}")
  set(_download_dir "${_cache_root}/${_url_key}")

  FetchContent_Declare(
    ${content_name}
    ${ARGN}
    DOWNLOAD_DIR "${_download_dir}"
  )
endfunction()

message(STATUS "FetchContent archive cache: ${DOWNLOAD_CACHE_DIR}")
