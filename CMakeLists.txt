cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 OLD)
endif()
if(POLICY CMP0127)
    set(CMAKE_POLICY_DEFAULT_CMP0127 OLD)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(Thot VERSION 3.4.5 LANGUAGES CXX C)

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

if(WIN32)
    add_compile_options(/D_ITERATOR_DEBUG_LEVEL=0)
endif()

if(APPLE)
    add_compile_options(-mmacosx-version-min=10.9 -Werror=partial-availability)
    add_link_options(-mmacosx-version-min=10.9 -Wl,-no_weak_imports)
endif()

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.5)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

include(FetchContent)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
)

set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_MakeAvailable(yaml-cpp)
if(UNIX)
    target_compile_options(yaml-cpp PRIVATE -Wno-effc++)
endif()
if(APPLE)
    target_compile_options(yaml-cpp PRIVATE -Wno-shadow)
endif()

if(BUILD_PYTHON_MODULE)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip
    )
    FetchContent_MakeAvailable(pybind11)
endif()

if(UNIX)
    add_compile_options(-Wall -Werror)
endif()

if(WIN32)
    add_compile_options(/WX)
    add_link_options(/WX)
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
