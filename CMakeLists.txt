cmake_minimum_required(VERSION 3.26...3.27)

project(fastwave LANGUAGES CXX)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pdm install --no-lock
  =====================================================================
  If you are a software developer, you need to rerun the above after
  editing C++ files.")
endif()

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Force Release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # is -O3 worth it? -O2

# Find libs nanobind
# Detect the installed nanobind package and import it into CMake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)
find_package(nanobind CONFIG REQUIRED)

# https://nanobind.readthedocs.io/en/latest/api_cmake.html#low-level-interface
# nanobind_add_module(
#   # Name of the extension
#   _fastwave
#   # Target the stable ABI for Python 3.12+, which reduces
#   # the number of binary wheels that must be built. This
#   # does nothing on older Python versions
#   STABLE_ABI
#   # Source code goes here
#   src/_fastwave.cpp
# )
# https://nanobind.readthedocs.io/en/latest/faq.html#importing-fails-due-to-missing-lib-nanobind-dylib-so-dll
# ^ need to add -static to WA @rpath on MacOS
nanobind_build_library(nanobind-static-abi3 STATIC)

add_library(_fastwave MODULE src/_fastwave.cpp)

target_link_libraries(_fastwave PRIVATE nanobind-static-abi3)
nanobind_opt_size(_fastwave)
nanobind_lto(_fastwave)
nanobind_set_visibility(_fastwave)
nanobind_strip(_fastwave)
nanobind_disable_stack_protector(_fastwave)
nanobind_extension_abi3(_fastwave)  # with nanobind-abi3
nanobind_extension(_fastwave)
nanobind_compile_options(_fastwave)
nanobind_link_options(_fastwave)
nanobind_musl_static_libcpp(_fastwave)

# Install directive for scikit-build-core
install(TARGETS _fastwave LIBRARY DESTINATION fastwave)
