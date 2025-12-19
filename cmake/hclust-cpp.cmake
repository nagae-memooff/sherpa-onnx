function(download_hclust_cpp)
  include(FetchContent)

  # The latest commit as of 2024.09.29
  set(hclust_cpp_URL  "https://github.com/csukuangfj/hclust-cpp/archive/refs/tags/2024-09-29.tar.gz")
  set(hclust_cpp_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/hclust-cpp-2024-09-29.tar.gz")
  set(hclust_cpp_HASH "SHA256=a2037bcb260c0e5ec5c2b5344a5af7359a4c5c9c34c92e086adff3ae7f21d700")

  # If you don't have access to the Internet,
  # please pre-download hclust-cpp
  set(possible_file_locations
    $ENV{HOME}/Downloads/hclust-cpp-2024-09-29.tar.gz
    ${CMAKE_SOURCE_DIR}/hclust-cpp-2024-09-29.tar.gz
    ${CMAKE_BINARY_DIR}/hclust-cpp-2024-09-29.tar.gz
    /tmp/hclust-cpp-2024-09-29.tar.gz
    /star-fj/fangjun/download/github/hclust-cpp-2024-09-29.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(hclust_cpp_URL  "${f}")
      file(TO_CMAKE_PATH "${hclust_cpp_URL}" hclust_cpp_URL)
      message(STATUS "Found local downloaded hclust_cpp: ${hclust_cpp_URL}")
      set(hclust_cpp_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(hclust_cpp
    URL
      ${hclust_cpp_URL}
      ${hclust_cpp_URL2}
    URL_HASH          ${hclust_cpp_HASH}
  )

  FetchContent_GetProperties(hclust_cpp)
  if(NOT hclust_cpp_POPULATED)
    message(STATUS "Downloading hclust_cpp from ${hclust_cpp_URL}")
    FetchContent_Populate(hclust_cpp)
  endif()

  message(STATUS "hclust_cpp is downloaded to ${hclust_cpp_SOURCE_DIR}")
  message(STATUS "hclust_cpp's binary dir is ${hclust_cpp_BINARY_DIR}")
  include_directories(${hclust_cpp_SOURCE_DIR})
endfunction()

download_hclust_cpp()
