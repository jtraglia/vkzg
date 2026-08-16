cmake_minimum_required(VERSION 3.20)
project(vulkan-prover LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VULKAN_PROVER_BUILD_TESTS "Build the test executables" ON)
option(VULKAN_PROVER_BUILD_BENCHMARKS "Build the benchmark executables" ON)
option(VULKAN_PROVER_REGENERATE "Re-run the constant and shader generators at build time" ON)

find_package(Python3 COMPONENTS Interpreter)
find_package(Vulkan REQUIRED)

find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found; install glslang (e.g. `dnf install glslang`)")
endif()

# ---------------------------------------------------------------- generated
# Both generators are deterministic and their outputs are committed, so a
# checkout without Python still builds; regeneration is just kept honest here.
set(GEN_HOST_CONSTANTS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/bls12_381_constants.h)
set(GEN_DEV_CONSTANTS  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/bls12_381_constants.glsl.inc)
set(GEN_SHADER_SOURCE  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader_source.h)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
set(KERNEL_NAMES
    k_blob_to_fr k_ntt_pass k_build_circulant
    k_phase_a_sort k_phase_a k_ladder k_fold_ladder k_phase_b_split
    k_combine_split k_normalize k_compress_proofs k_bucket_reduce)
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

if(VULKAN_PROVER_REGENERATE AND Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT ${GEN_HOST_CONSTANTS} ${GEN_DEV_CONSTANTS}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
                ${GEN_HOST_CONSTANTS} ${GEN_DEV_CONSTANTS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
        COMMENT "Generating BLS12-381 constants")
    add_custom_command(
        OUTPUT ${GEN_SHADER_SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py
                ${GEN_SHADER_SOURCE} ${GLSLANG_VALIDATOR}
                "${CMAKE_CURRENT_SOURCE_DIR}/src:${SHADER_DIR}"
                ${KERNEL_SOURCES}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py ${SHADER_COMMON} ${KERNEL_SOURCES}
        COMMENT "Compiling and embedding SPIR-V kernels")
    add_custom_target(vulkan_prover_generated DEPENDS ${GEN_SHADER_SOURCE} ${GEN_HOST_CONSTANTS})
endif()

# ------------------------------------------------------------------ library
add_library(vulkan_prover
    src/vulkan_prover.cpp
    src/cpu/bls12_381.cpp
    src/cpu/setup.cpp
    src/setup_data.cpp)

if(TARGET vulkan_prover_generated)
    add_dependencies(vulkan_prover vulkan_prover_generated)
endif()

target_include_directories(vulkan_prover
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vulkan_prover PRIVATE -Wall -Wextra -O3)
target_link_libraries(vulkan_prover PUBLIC Vulkan::Vulkan)
set_target_properties(vulkan_prover PROPERTIES
    PUBLIC_HEADER include/vulkan_prover.h
    VERSION ${PROJECT_VERSION})

# The reference implementation is only needed by tests, so keep it out of the
# shipped library.
add_library(vulkan_prover_reference STATIC src/cpu/reference.cpp)
target_include_directories(vulkan_prover_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vulkan_prover_reference PRIVATE -O3)

# -------------------------------------------------------------------- tests
if(VULKAN_PROVER_BUILD_TESTS)
    enable_testing()
    add_library(vulkan_prover_vectors STATIC tests/vectors.cpp)
    target_include_directories(vulkan_prover_vectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cpu_field tests/test_cpu_field.cpp)
    target_link_libraries(test_cpu_field PRIVATE vulkan_prover)

    add_executable(test_reference tests/test_reference.cpp)
    target_link_libraries(test_reference PRIVATE vulkan_prover vulkan_prover_reference vulkan_prover_vectors)

    add_executable(test_gpu tests/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE vulkan_prover vulkan_prover_vectors)

    set(VEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors)
    add_test(NAME cpu_field COMMAND test_cpu_field)
    add_test(NAME reference COMMAND test_reference ${VEC_DIR})
    add_test(NAME gpu COMMAND test_gpu ${VEC_DIR})
endif()

# --------------------------------------------------------------- benchmarks
if(VULKAN_PROVER_BUILD_BENCHMARKS)
    add_executable(bench bench/bench.cpp)
    target_link_libraries(bench PRIVATE vulkan_prover)

    add_executable(profile_stages bench/profile_stages.cpp)
    target_link_libraries(profile_stages PRIVATE vulkan_prover)
endif()

add_executable(example examples/example.c)
target_link_libraries(example PRIVATE vulkan_prover)

# ------------------------------------------------------------------ install
include(GNUInstallDirs)
install(TARGETS vulkan_prover
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
