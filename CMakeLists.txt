cmake_minimum_required(VERSION 3.19)

project(RandLib
  VERSION 1.0.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(RandLib INTERFACE)

target_compile_features(RandLib
  INTERFACE
    cxx_std_17
)

set(RANDLIB_HDRS
  ${RandLib_SOURCE_DIR}/source/RandLib/distributions/ProbabilityDistribution.hpp
  ${RandLib_SOURCE_DIR}/source/RandLib/distributions/BasicRandGenerator.hpp
  ${RandLib_SOURCE_DIR}/source/RandLib/distributions/UniformRand.hpp
  ${RandLib_SOURCE_DIR}/source/RandLib/distributions/UniformDiscreteRand.hpp
  ${RandLib_SOURCE_DIR}/source/RandLib/distributions/UnivariateDistribution.hpp

  ${RandLib_SOURCE_DIR}/source/RandLib/RandLib_global.h
)

target_sources(RandLib
  INTERFACE
    ${RANDLIB_HDRS}
)

target_include_directories(RandLib
  INTERFACE
    ${RandLib_SOURCE_DIR}/source
    $<INSTALL_INTERFACE:include>
)

include(CTest)

if(BUILD_TESTING)
  # testing binary
  add_executable(UniformDiscreteTesting ${RandLib_SOURCE_DIR}/testing/UniformDiscreteTesting.cpp)

  target_link_libraries(UniformDiscreteTesting PRIVATE RandLib)

  add_test(NAME UniformDiscreteTesting
    COMMAND $<TARGET_FILE:UniformDiscreteTesting>
  )

  add_executable(UniformContinuousTesting ${RandLib_SOURCE_DIR}/testing/UniformContinuousTesting.cpp)
  target_link_libraries(UniformContinuousTesting PRIVATE RandLib)

  add_test(NAME UniformContinuousTesting
    COMMAND $<TARGET_FILE:UniformContinuousTesting>
  )
endif()
