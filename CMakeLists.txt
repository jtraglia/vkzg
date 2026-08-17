cmake_minimum_required(VERSION 3.20)
project(vkzg LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VKZG_BUILD_TESTS "Build the test executables" ON)
option(VKZG_BUILD_BENCHMARKS "Build the benchmark executables" ON)
option(VKZG_REGENERATE "Re-run the constant and shader generators at build time" ON)

find_package(Python3 COMPONENTS Interpreter)
find_package(Vulkan REQUIRED)
include(GNUInstallDirs)

# ---------------------------------------------------------------- generated
# Both generators are deterministic and their outputs are committed, so a
# checkout without Python still builds; regeneration is just kept honest here.
set(GEN_DEV_CONSTANTS  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/bls12_381_constants.glsl.inc)
set(GEN_SHADER_SOURCE  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader_source.h)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
set(KERNEL_NAMES
    k_blob_to_fr k_ntt_pass k_build_circulant
    k_phase_a_sort k_phase_a k_ladder k_fold_ladder k_phase_b_split
    k_combine_split k_normalize k_compress_proofs k_bucket_reduce
    k_ew_mul k_ew_div k_pow_seq k_vanishing_poly
    k_cells_to_fr k_fr_to_cells)
set(KERNEL_SOURCES "")
foreach(k ${KERNEL_NAMES})
    list(APPEND KERNEL_SOURCES ${SHADER_DIR}/${k}.comp)
endforeach()

# Shared GLSL the kernels #include; not compiled on its own, but the embed
# step must re-run when it changes.
set(SHADER_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layout_defs.h
    ${GEN_DEV_CONSTANTS}
    ${SHADER_DIR}/field.glsl.inc)

if(VKZG_REGENERATE AND Python3_Interpreter_FOUND)
    find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "glslangValidator not found; install glslang (e.g. `dnf install glslang`)")
    endif()

    add_custom_command(
        OUTPUT ${GEN_DEV_CONSTANTS}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
                ${GEN_DEV_CONSTANTS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
        COMMENT "Generating BLS12-381 constants")
    add_custom_command(
        OUTPUT ${GEN_SHADER_SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py
                ${GEN_SHADER_SOURCE} ${GLSLANG_VALIDATOR}
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                ${KERNEL_SOURCES}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py ${SHADER_COMMON} ${KERNEL_SOURCES}
        COMMENT "Compiling and embedding SPIR-V kernels")
    add_custom_target(vkzg_generated DEPENDS ${GEN_SHADER_SOURCE} ${GEN_DEV_CONSTANTS})
endif()

# ------------------------------------------------------- precomputed tables
# The mainnet FK20/position tables are precomputed once (see the "how this
# was generated" note in src/precomputed_tables.cpp) and shipped as a binary
# blob rather than derived from the trusted setup at runtime. It's tried at a
# build-tree path first (so tests/bench work with no install step) and falls
# back to the installed path.
set(TABLES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/precomputed_tables.bin)
set(TABLES_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/precomputed_tables.bin)
set(TABLES_INSTALL_PATH ${CMAKE_INSTALL_FULL_DATADIR}/vkzg/precomputed_tables.bin)
configure_file(${TABLES_SRC} ${TABLES_BUILD_PATH} COPYONLY)
# Generated into the build tree, not committed: it bakes in this machine's
# absolute paths.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/install_paths.h
    "#pragma once\n"
    "#define VKZG_TABLES_PATH_BUILD \"${TABLES_BUILD_PATH}\"\n"
    "#define VKZG_TABLES_PATH_INSTALL \"${TABLES_INSTALL_PATH}\"\n")
install(FILES ${TABLES_SRC} DESTINATION ${CMAKE_INSTALL_DATADIR}/vkzg)

# ------------------------------------------------------------------ library
add_library(vkzg
    src/vkzg.cpp
    src/gpu_topology.cpp
    src/precomputed_tables.cpp
    src/setup_data.cpp)

if(TARGET vkzg_generated)
    add_dependencies(vkzg vkzg_generated)
endif()

target_include_directories(vkzg
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/generated)
# Vulkan structs are brace-initialized with only sType set (rest zeroed by
# C++ value-init), which -Wextra otherwise complains about on every field.
target_compile_options(vkzg PRIVATE -Wall -Wextra -Wno-missing-field-initializers -O3)
target_link_libraries(vkzg PUBLIC Vulkan::Vulkan)
set_target_properties(vkzg PROPERTIES
    PUBLIC_HEADER include/vkzg.h
    VERSION ${PROJECT_VERSION})

# -------------------------------------------------------------------- tests
if(VKZG_BUILD_TESTS)
    enable_testing()
    add_library(vkzg_vectors STATIC tests/vectors.cpp)
    target_include_directories(vkzg_vectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # "test" is a reserved target name once CTest is enabled, so the CMake
    # target is vkzg_test; the built binary is still named `test`.
    add_executable(vkzg_test tests/test.cpp)
    target_link_libraries(vkzg_test PRIVATE vkzg vkzg_vectors)
    set_target_properties(vkzg_test PROPERTIES OUTPUT_NAME test)

    set(VEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors)
    add_test(NAME test COMMAND vkzg_test ${VEC_DIR})
endif()

# --------------------------------------------------------------- benchmarks
if(VKZG_BUILD_BENCHMARKS)
    add_executable(bench bench/bench.cpp)
    target_link_libraries(bench PRIVATE vkzg)

    add_executable(profile bench/profile.cpp)
    target_link_libraries(profile PRIVATE vkzg)
endif()

# ------------------------------------------------------------------ install
install(TARGETS vkzg
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
