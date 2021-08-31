cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(Thot VERSION 3.3.2 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)

include(CheckSymbolExists)

check_symbol_exists(getdelim "stdio.h" HAVE_GETDELIM)
if(HAVE_GETDELIM)
    add_compile_definitions(THOT_HAVE_GETDELIM)
endif()

check_symbol_exists(getline "stdio.h" HAVE_GETLINE)
if(HAVE_GETLINE)
    add_compile_definitions(THOT_HAVE_GETLINE)
endif()

include(CheckTypeSize)

check_type_size(ssize_t SSIZE_T)
if(HAVE_SSIZE_T)
    add_compile_definitions(THOT_HAVE_SSIZE_T)
endif()

include(CheckIncludeFiles)

check_include_files("sys/time.h;time.h" TIME_WITH_SYS_TIME)
if(TIME_WITH_SYS_TIME)
    add_compile_definitions(THOT_TIME_WITH_SYS_TIME)
else()
    check_include_files("sys/time.h" HAVE_SYS_TIME_H)
    if(HAVE_SYS_TIME_H)
        add_compile_definitions(THOT_HAVE_SYS_TIME_H)
    endif()
endif()

option(USE_GMP, "Use GMP library")
if(USE_GMP)
    find_package(GMP)
    if(GMP_FOUND)
        add_compile_definitions(THOT_HAVE_GMP)
        include_directories(${GMP_INCLUDE_DIRS})
    endif()
endif()

option(BUILD_SHARED_LIBRARY "Build shared library" ON)
option(BUILD_PYTHON_MODULE "Build Python module" ON)
option(BUILD_TESTS "Build unit tests" ON)

if(UNIX)
    add_compile_options(-Wall -Werror)
endif()

if(APPLE)
    add_compile_options(-mmacosx-version-min=10.13 -Werror=partial-availability)
    add_link_options(-mmacosx-version-min=10.13 -Wl,-no_weak_imports)
endif()

if(WIN32)
    add_compile_options(/WX)
endif()

include(FetchContent)

if(BUILD_PYTHON_MODULE)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.7.1.zip
    )
    FetchContent_MakeAvailable(pybind11)
endif()

add_subdirectory(src)

if(BUILD_TESTS)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()
