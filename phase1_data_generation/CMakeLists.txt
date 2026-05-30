cmake_minimum_required(VERSION 3.20)
project(puzzle_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Optimization flags (GCC / Clang / MinGW)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -funroll-loops -Wall -Wextra)
endif()

# Portable std::thread linking (no-op where not needed, e.g. macOS).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenMP is only needed by the (future) PDB builder; optional so a stock dev
# machine without libomp can still build generate_data / verify_state.
find_package(OpenMP QUIET)

set(COMMON_SRCS
    src/puzzle_state.cpp
    src/pdb.cpp
)

# --- Target 1: generate_data — main pipeline ---
add_executable(generate_data ${COMMON_SRCS} src/main.cpp)
target_include_directories(generate_data PRIVATE include)
target_link_libraries(generate_data PRIVATE Threads::Threads)

# --- Target 2: build_pdbs — one-shot additive PDB generator (skeleton) ---
add_executable(build_pdbs ${COMMON_SRCS} src/pdb_builder.cpp)
target_include_directories(build_pdbs PRIVATE include)
target_link_libraries(build_pdbs PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(build_pdbs PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- Target 3: verify_state — correctness tests ---
add_executable(verify_state ${COMMON_SRCS} tests/verify_state.cpp)
target_include_directories(verify_state PRIVATE include)
target_link_libraries(verify_state PRIVATE Threads::Threads)

message(STATUS "")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Build:   cmake -S . -B build && cmake --build build -j")
message(STATUS "Test:    ./build/verify_state")
message(STATUS "Run:     ./build/generate_data --threads 4 --target 1000 --scramble 30 --out-dir ./data")
message(STATUS "")
