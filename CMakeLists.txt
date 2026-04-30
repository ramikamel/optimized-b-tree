cmake_minimum_required(VERSION 3.16)
project(adaptive_btree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg)
if(_ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

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
  target_compile_options(adaptive_btree PRIVATE /W4 /O2)
else()
  target_compile_options(adaptive_btree PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    $<$<CONFIG:RelWithDebInfo>:-O3 -DNDEBUG -march=native>
  )
endif()

add_executable(benchmark
  benchmark/benchmark.cpp
)

if(MSVC)
  target_compile_options(benchmark PRIVATE /O2)
else()
  target_compile_options(benchmark PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    $<$<CONFIG:RelWithDebInfo>:-O3 -DNDEBUG -march=native>
  )
endif()

target_link_libraries(benchmark PRIVATE adaptive_btree)
