cmake_minimum_required(VERSION 3.20)
project(cube_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler flags (GCC / MinGW-w64)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3
        -march=native       # use full i7-11700F ISA (AVX2, etc.)
        -funroll-loops
        -ffast-math
        -Wall -Wextra
    )
endif()

# ---------------------------------------------------------------------------
# OpenMP (parallelises PDB BFS; solver uses std::thread)
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Shared source files used by both executables
# ---------------------------------------------------------------------------
set(COMMON_SRCS
    src/cube_state.cpp
    src/move_tables.cpp
    src/pdb.cpp
)

# ---------------------------------------------------------------------------
# Target 1: generate_data — main data pipeline
# ---------------------------------------------------------------------------
add_executable(generate_data
    ${COMMON_SRCS}
    src/ida_star.cpp
    src/main.cpp
)
target_include_directories(generate_data PRIVATE include)
target_link_libraries(generate_data PRIVATE OpenMP::OpenMP_CXX)

# On Windows (MinGW), link pthreads explicitly for std::thread
if(WIN32 AND MINGW)
    target_link_libraries(generate_data PRIVATE -lpthread)
endif()

# ---------------------------------------------------------------------------
# Target 2: build_pdbs — one-shot PDB generator (run before generate_data)
# ---------------------------------------------------------------------------
add_executable(build_pdbs
    ${COMMON_SRCS}
    src/pdb_builder.cpp
)
target_include_directories(build_pdbs PRIVATE include)
target_link_libraries(build_pdbs PRIVATE OpenMP::OpenMP_CXX)

if(WIN32 AND MINGW)
    target_link_libraries(build_pdbs PRIVATE -lpthread)
endif()

# ---------------------------------------------------------------------------
# Build instructions (printed at configure time as a reminder)
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "To build:")
message(STATUS "  cd build && mingw32-make -j16")
message(STATUS "")
message(STATUS "Step 1 — generate PDBs (run once, ~2-3 hours):")
message(STATUS "  ./build_pdbs.exe --out D:/Search_DB/")
message(STATUS "")
message(STATUS "Step 2 — generate dataset (~2-4 days):")
message(STATUS "  ./generate_data.exe --threads 14 --target 100000000")
message(STATUS "                      --pdb-dir D:/Search_DB/")
message(STATUS "                      --out-dir D:/Search_DB/")
message(STATUS "")
