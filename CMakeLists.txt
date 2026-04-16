cmake_minimum_required(VERSION 3.16)
project(adaptive_btree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(adaptive_btree
  src/adaptive_btree.cpp
  src/inner_node.cpp
  src/leaf_node.cpp
  src/node.cpp
  src/slotted_page.cpp
)

target_include_directories(adaptive_btree PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(adaptive_btree PRIVATE /W4)
else()
  target_compile_options(adaptive_btree PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(benchmark
  benchmark/benchmark.cpp
)

target_link_libraries(benchmark PRIVATE adaptive_btree)
