cmake_minimum_required(VERSION 3.20)
project(kzgpu LANGUAGES C CXX OBJCXX VERSION 0.1.0)

if(NOT APPLE)
    message(FATAL_ERROR "kzgpu targets Apple GPUs via Metal and only builds on macOS.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(KZGPU_BUILD_TESTS "Build the test executables" ON)
option(KZGPU_BUILD_BENCHMARKS "Build the benchmark executables" ON)
option(KZGPU_REGENERATE "Re-run the constant and shader generators at build time" ON)

find_package(Python3 COMPONENTS Interpreter)

# ---------------------------------------------------------------- generated
# Both generators are deterministic and their outputs are committed, so a
# checkout without Python still builds; regeneration is just kept honest here.
set(GEN_HOST_CONSTANTS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/bls12_381_constants.h)
set(GEN_DEV_CONSTANTS  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/bls12_381_constants.h.inc)
set(GEN_SHADER_SOURCE  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader_source.h)

set(GEN_SETUP_DATA ${CMAKE_CURRENT_SOURCE_DIR}/src/setup_data.h)

set(SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layout_defs.h
    ${GEN_DEV_CONSTANTS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/field.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/kernels.metal)

if(KZGPU_REGENERATE AND Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT ${GEN_HOST_CONSTANTS} ${GEN_DEV_CONSTANTS}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
                ${GEN_HOST_CONSTANTS} ${GEN_DEV_CONSTANTS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_constants.py
        COMMENT "Generating BLS12-381 constants")
    add_custom_command(
        OUTPUT ${GEN_SHADER_SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py
                ${GEN_SHADER_SOURCE} ${SHADER_SOURCES}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shaders.py ${SHADER_SOURCES}
        COMMENT "Embedding Metal shader source")
    add_custom_command(
        OUTPUT ${GEN_SETUP_DATA}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_setup.py
                ${CMAKE_CURRENT_SOURCE_DIR}/data/trusted_setup.txt ${GEN_SETUP_DATA}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_setup.py
                ${CMAKE_CURRENT_SOURCE_DIR}/data/trusted_setup.txt
        COMMENT "Embedding the mainnet trusted setup")
    add_custom_target(kzgpu_generated
        DEPENDS ${GEN_SHADER_SOURCE} ${GEN_HOST_CONSTANTS} ${GEN_SETUP_DATA})
endif()

# ------------------------------------------------------------------ library
add_library(kzgpu
    src/kzgpu_metal.mm
    src/cpu/bls12_381.cpp
    src/cpu/setup.cpp
    src/setup_data.cpp)

if(TARGET kzgpu_generated)
    add_dependencies(kzgpu kzgpu_generated)
endif()

target_include_directories(kzgpu
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kzgpu PRIVATE -Wall -Wextra -O3)
target_link_libraries(kzgpu PUBLIC "-framework Metal" "-framework Foundation")
set_target_properties(kzgpu PROPERTIES
    PUBLIC_HEADER include/kzgpu.h
    VERSION ${PROJECT_VERSION})
# ARC for the Objective-C++ translation unit only.
set_source_files_properties(src/kzgpu_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# The reference implementation is only needed by tests, so keep it out of the
# shipped library.
add_library(kzgpu_reference STATIC src/cpu/reference.cpp)
target_include_directories(kzgpu_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kzgpu_reference PRIVATE -O3)

# -------------------------------------------------------------------- tests
if(KZGPU_BUILD_TESTS)
    enable_testing()
    add_library(kzgpu_vectors STATIC tests/vectors.cpp)
    target_include_directories(kzgpu_vectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cpu_field tests/test_cpu_field.cpp)
    target_link_libraries(test_cpu_field PRIVATE kzgpu)

    add_executable(test_reference tests/test_reference.cpp)
    target_link_libraries(test_reference PRIVATE kzgpu kzgpu_reference kzgpu_vectors)

    add_executable(test_gpu tests/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE kzgpu kzgpu_vectors)

    add_executable(test_setup_data tests/test_setup_data.cpp)
    target_link_libraries(test_setup_data PRIVATE kzgpu)

    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/trusted_setup.txt)
    set(VEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors)
    add_test(NAME cpu_field COMMAND test_cpu_field)
    add_test(NAME reference COMMAND test_reference ${DATA_DIR} ${VEC_DIR})
    add_test(NAME gpu COMMAND test_gpu ${DATA_DIR} ${VEC_DIR})
    add_test(NAME setup_data COMMAND test_setup_data ${DATA_DIR})
endif()

# --------------------------------------------------------------- benchmarks
if(KZGPU_BUILD_BENCHMARKS)
    add_executable(bench_kzgpu bench/bench_kzgpu.cpp)
    target_link_libraries(bench_kzgpu PRIVATE kzgpu)

    add_executable(profile_stages bench/profile_stages.cpp)
    target_link_libraries(profile_stages PRIVATE kzgpu)
endif()

add_executable(kzgpu_example examples/example.c)
target_link_libraries(kzgpu_example PRIVATE kzgpu)

# ------------------------------------------------------------------ install
include(GNUInstallDirs)
install(TARGETS kzgpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES data/trusted_setup.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/kzgpu)
