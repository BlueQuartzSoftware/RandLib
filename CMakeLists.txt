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

target_include_directories(RandLib
  INTERFACE
    $<BUILD_INTERFACE:${RandLib_SOURCE_DIR}/source>
    $<INSTALL_INTERFACE:include>
)

include(CTest)

if(BUILD_TESTING)
  # testing binary
  add_executable(UniformDiscreteTesting ${RandLib_SOURCE_DIR}/testing/UniformDiscreteTesting.cpp)

  target_link_libraries(UniformDiscreteTesting PRIVATE RandLib)

  add_test(NAME UniformDiscreteTesting
    COMMAND UniformDiscreteTesting
  )

  add_executable(UniformContinuousTesting ${RandLib_SOURCE_DIR}/testing/UniformContinuousTesting.cpp)
  target_link_libraries(UniformContinuousTesting PRIVATE RandLib)

  add_test(NAME UniformContinuousTesting
    COMMAND UniformContinuousTesting
  )
endif()

option(RANDLIB_ENABLE_INSTALL "Enables RandLib install rules" ON)
if(RANDLIB_ENABLE_INSTALL)
  function(install_with_directory)
    set(optionsArgs)
    set(oneValueArgs DESTINATION COMPONENT BASE_DIR)
    set(multiValueArgs FILES)
    cmake_parse_arguments(ARGS "${optionsArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(file ${ARGS_FILES})
        file(RELATIVE_PATH relative_file ${ARGS_BASE_DIR} ${file})
        get_filename_component(destination_dir ${relative_file} DIRECTORY)
        install(FILES ${file}
          DESTINATION ${ARGS_DESTINATION}/${destination_dir}
          COMPONENT ${ARGS_COMPONENT}
        )
    endforeach()
  endfunction()

  install(TARGETS RandLib
    EXPORT RandLibTargets
  )

  set(RANDLIB_PACKAGE_NAMESPACE "RandLib::")
  export(EXPORT RandLibTargets NAMESPACE ${RANDLIB_PACKAGE_NAMESPACE})

  include(CMakePackageConfigHelpers)

  set(RANDLIB_INSTALL_CMAKE_PREFIX "lib/cmake" CACHE STRING "Install dir for CMake config")

  set(RANDLIB_CONFIG_FILE ${RandLib_BINARY_DIR}/RandLibConfig.cmake)
  set(RANDLIB_VERSION_FILE ${RandLib_BINARY_DIR}/RandLibConfigVersion.cmake)

  configure_package_config_file(${RandLib_SOURCE_DIR}/cmake/RandLibConfig.cmake.in ${RANDLIB_CONFIG_FILE}
    INSTALL_DESTINATION ${RANDLIB_INSTALL_CMAKE_PREFIX}
  )

  write_basic_package_version_file(${RANDLIB_VERSION_FILE}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
  )

  install(FILES ${RANDLIB_CONFIG_FILE} ${RANDLIB_VERSION_FILE}
    DESTINATION ${RANDLIB_INSTALL_CMAKE_PREFIX}
    COMPONENT develop
  )

  include(GNUInstallDirs)
  install(TARGETS RandLib
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT runtime
    LIBRARY
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      COMPONENT runtime
      NAMELINK_COMPONENT develop
    ARCHIVE
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      COMPONENT develop
  )

  install(EXPORT RandLibTargets
    NAMESPACE ${RANDLIB_PACKAGE_NAMESPACE}
    DESTINATION ${RANDLIB_INSTALL_CMAKE_PREFIX}
  )

  install_with_directory(FILES ${RANDLIB_HDRS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandLib
    COMPONENT develop
    BASE_DIR ${RandLib_SOURCE_DIR}/source/RandLib
  )
endif()
