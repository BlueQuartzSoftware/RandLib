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
generate_export_header(RandLib EXPORT_FILE_NAME ${RANDLIB_EXPORT_HEADER} EXPORT_MACRO_NAME RANDLIBSHARED_EXPORT)

target_include_directories(RandLib PUBLIC ${RandLib_BINARY_DIR}/generated)

set(RANDLIB_SRCS
    ${RandLib_SOURCE_DIR}/distributions/BasicRandGenerator.cpp

   ${RandLib_SOURCE_DIR}/math/BetaMath.cpp 
    ${RandLib_SOURCE_DIR}/math/GammaMath.cpp 
    ${RandLib_SOURCE_DIR}/math/MarcumMath.cpp
    ${RandLib_SOURCE_DIR}/math/NumericMath.cpp 
    ${RandLib_SOURCE_DIR}/math/RandMath.cpp 
)
set(RANDLIB_HDRS
    ${RandLib_SOURCE_DIR}/distributions/ProbabilityDistribution.h
    ${RandLib_SOURCE_DIR}/distributions/BasicRandGenerator.h 
    ${RandLib_SOURCE_DIR}/distributions/UniformRand.h 
    ${RandLib_SOURCE_DIR}/distributions/UniformDiscreteRand.h 
    ${RandLib_SOURCE_DIR}/distributions/UnivariateDistribution.h 

    ${RandLib_SOURCE_DIR}/external/checks.hpp 
    ${RandLib_SOURCE_DIR}/external/floor.hpp 
    ${RandLib_SOURCE_DIR}/external/gcem_options.hpp 
    ${RandLib_SOURCE_DIR}/external/helpers.hpp 
    ${RandLib_SOURCE_DIR}/external/log.hpp 
    ${RandLib_SOURCE_DIR}/external/pow.hpp 
    ${RandLib_SOURCE_DIR}/external/sgn.hpp 
    ${RandLib_SOURCE_DIR}/external/sqrt.hpp

    ${RandLib_SOURCE_DIR}/math/BetaMath.h 
    ${RandLib_SOURCE_DIR}/math/Constants.h 
    ${RandLib_SOURCE_DIR}/math/GammaMath.h 
    ${RandLib_SOURCE_DIR}/math/MarcumMath.h
    ${RandLib_SOURCE_DIR}/math/NumericMath.h
    ${RandLib_SOURCE_DIR}/math/RandMath.h 

    ${RandLib_SOURCE_DIR}/RandLib_global.h 
    ${RandLib_SOURCE_DIR}/RandLib.h 
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
