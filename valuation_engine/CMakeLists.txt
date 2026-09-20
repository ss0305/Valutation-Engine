cmake_minimum_required(VERSION 3.20)
project(valuation_engine
        VERSION 1.0.0
        DESCRIPTION "Real-time fair value, IPV and valuation reserves engine"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VE_BUILD_TESTS "Build the GoogleTest suite" ON)
option(VE_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(VE_ENABLE_WARNINGS "Enable a strict warning set" ON)

find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(valuation_core STATIC
  src/OptionPricer.cpp
  src/IPVEngine.cpp
  src/ReserveCalculator.cpp)

target_include_directories(valuation_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(valuation_core PUBLIC Threads::Threads)

if(VE_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(valuation_core PRIVATE /W4 /permissive-)
  else()
    target_compile_options(valuation_core PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast)
  endif()
endif()

add_library(valuation::core ALIAS valuation_core)

# -----------------------------------------------------------------------------
# Python bindings (Module 4)
# -----------------------------------------------------------------------------
if(VE_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(NOT pybind11_DIR AND Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _pybind11_dir
      ERROR_QUIET
      RESULT_VARIABLE _pybind11_found)
    if(_pybind11_found EQUAL 0)
      set(pybind11_DIR "${_pybind11_dir}" CACHE PATH "pybind11 CMake directory")
    endif()
  endif()

  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(valuation_engine MODULE src/bindings.cpp)
    target_link_libraries(valuation_engine PRIVATE valuation_core)
    target_compile_definitions(valuation_engine PRIVATE VERSION_INFO=${PROJECT_VERSION})
  else()
    message(WARNING "pybind11 not found - skipping the Python module. "
                    "Install it with: pip install pybind11")
  endif()
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(VE_BUILD_TESTS)
  enable_testing()

  set(_vendored_gtest "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest")
  if(EXISTS "${_vendored_gtest}/CMakeLists.txt")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory("${_vendored_gtest}" "${CMAKE_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
  else()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
      include(FetchContent)
      FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2)
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(googletest)
    endif()
  endif()

  add_executable(test_analytics tests/test_analytics.cpp)
  target_link_libraries(test_analytics PRIVATE valuation_core GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(test_analytics)
endif()
