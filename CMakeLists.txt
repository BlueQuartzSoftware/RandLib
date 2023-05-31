cmake_minimum_required(VERSION 3.19)

project(RandLib
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(RandLib SHARED)

target_compile_features(RandLib
  PUBLIC
    cxx_std_17
)

target_include_directories(RandLib PUBLIC ${RandLib_SOURCE_DIR})

if(MSVC)
  target_compile_options(RandLib
    PRIVATE
      /MP
  )
endif()

if(UNIX)
  target_link_libraries(RandLib
    PRIVATE
      ${CMAKE_DL_LIBS}
  )
endif()

set(RANDLIB_EXPORT_HEADER ${RandLib_BINARY_DIR}/generated/RandLib_export.h)

include(GenerateExportHeader)
generate_export_header(RandLib EXPORT_FILE_NAME ${RANDLIB_EXPORT_HEADER})

target_include_directories(RandLib PUBLIC ${RandLib_BINARY_DIR}/generated)

set(RANDLIB_SRCS
    ${RandLib_SOURCE_DIR}/distributions/BasicRandGenerator.cpp
)

set(RANDLIB_HDRS
    ${RandLib_SOURCE_DIR}/distributions/ProbabilityDistribution.hpp
    ${RandLib_SOURCE_DIR}/distributions/BasicRandGenerator.hpp
    ${RandLib_SOURCE_DIR}/distributions/UniformRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/UniformDiscreteRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/UnivariateDistribution.hpp

    ${RandLib_SOURCE_DIR}/RandLib_global.h
)

target_sources(RandLib
  PRIVATE
    ${RANDLIB_HDRS}
    ${RANDLIB_SRCS}
    ${RANDLIB_EXPORT_HEADER}
)

set(SEARCH_INCLUDE_DIR ${RandLib_SOURCE_DIR}/external)

target_include_directories(RandLib
  PUBLIC
  $<BUILD_INTERFACE:${SEARCH_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# testing binary
add_executable(UniformDiscreteTesting ${RandLib_SOURCE_DIR}/testing/UniformDiscreteTesting.cpp)

target_link_libraries(UniformDiscreteTesting PRIVATE RandLib)

add_test(
  NAME UniformDiscreteTesting
  COMMAND $<TARGET_FILE:UniformDiscreteTesting>
)

add_executable(UniformContinuousTesting ${RandLib_SOURCE_DIR}/testing/UniformContinuousTesting.cpp)
target_link_libraries(UniformContinuousTesting PRIVATE RandLib)

add_test(
    NAME UniformContinuousTesting
    COMMAND $<TARGET_FILE:UniformContinuousTesting>
)

# enable testing functionality
enable_testing()

