cmake_minimum_required(VERSION 3.21)
project(phonon25
    VERSION 0.1.0
    DESCRIPTION "Pure-C inference engine for FermionResearch/Phonon-2.5"
    HOMEPAGE_URL "https://github.com/eschmidbauer/phonon-2.5-c"
    LANGUAGES C)

option(PHONON25_BUILD_CLI      "Build the phonon25 command-line binary" ${PROJECT_IS_TOP_LEVEL})
option(PHONON25_BUILD_EXAMPLES "Build examples/" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

set(PHONON25_LIB_SOURCES
    csrc/phonon25.h
    csrc/blas.h
    csrc/blas.c
    csrc/threadpool.h
    csrc/threadpool.c
    csrc/tensor_ops.c
    csrc/encoder.c
    csrc/decoder.c
    csrc/mel.c
    csrc/weights.c
    csrc/tokens.c
    csrc/wav.c
    csrc/resample.c
    csrc/segment.c
    csrc/engine.c
    csrc/json.c)

# The objects are built once (position-independent) and go into both the static library, what the CLI and C
# consumers link, and the shared one, what the Python bindings load. No dependencies beyond libc, libm and
# pthreads: every GEMM is in csrc/blas.c.
add_library(phonon25_objects OBJECT ${PHONON25_LIB_SOURCES})
set_target_properties(phonon25_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(phonon25_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(phonon25_objects PRIVATE -O3 -ffast-math -march=native -funroll-loops)
if(NOT APPLE)
    # glibc and musl hide POSIX (clock_gettime, setenv, sysconf, mmap …) under -std=c11 unless asked.
    target_compile_definitions(phonon25_objects PRIVATE _GNU_SOURCE)
endif()

add_library(phonon25 STATIC $<TARGET_OBJECTS:phonon25_objects>)
add_library(phonon25::phonon25 ALIAS phonon25)
add_library(phonon25_shared SHARED $<TARGET_OBJECTS:phonon25_objects>)
set_target_properties(phonon25_shared PROPERTIES OUTPUT_NAME phonon25)

find_package(Threads REQUIRED)
foreach(t phonon25 phonon25_shared)
    # Consumers include "phonon25.h" (and threadpool.h / blas.h for the pool and GEMM-mode calls) either way.
    target_include_directories(${t}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/phonon25>)
    target_link_libraries(${t} PUBLIC Threads::Threads m)
endforeach()

install(TARGETS phonon25 phonon25_shared EXPORT phonon25Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES csrc/phonon25.h csrc/threadpool.h csrc/blas.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phonon25)
install(EXPORT phonon25Targets
    NAMESPACE phonon25::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phonon25)

if(PHONON25_BUILD_CLI)
    add_executable(phonon25_cli csrc/main.c)
    set_target_properties(phonon25_cli PROPERTIES OUTPUT_NAME phonon25)
    target_link_libraries(phonon25_cli PRIVATE phonon25)
    install(TARGETS phonon25_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(PHONON25_BUILD_EXAMPLES)
    add_executable(phonon25_example examples/transcribe.c)
    set_target_properties(phonon25_example PROPERTIES OUTPUT_NAME transcribe)
    target_link_libraries(phonon25_example PRIVATE phonon25)
endif()
