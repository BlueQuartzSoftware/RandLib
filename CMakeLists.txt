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

set(RANDLIB_EXPORT_HEADER ${RandLib_BINARY_DIR}/generated/RandLib_export.hpp)

include(GenerateExportHeader)
generate_export_header(RandLib EXPORT_FILE_NAME ${RANDLIB_EXPORT_HEADER})

target_include_directories(RandLib PUBLIC ${RandLib_BINARY_DIR}/generated)

set(RANDLIB_HDRS
    # External Constexpr Math Library Functions
    ${RandLib_SOURCE_DIR}/external/checks.hpp
    ${RandLib_SOURCE_DIR}/external/floor.hpp
    ${RandLib_SOURCE_DIR}/external/gcem_options.hpp
    ${RandLib_SOURCE_DIR}/external/helpers.hpp
    ${RandLib_SOURCE_DIR}/external/log.hpp
    ${RandLib_SOURCE_DIR}/external/pow.hpp
    ${RandLib_SOURCE_DIR}/external/sgn.hpp
    ${RandLib_SOURCE_DIR}/external/sqrt.hpp

    # Overarching
    ${RandLib_SOURCE_DIR}/RandLib.hpp

    # Math
    ${RandLib_SOURCE_DIR}/math/BetaMath.hpp
    ${RandLib_SOURCE_DIR}/math/Constants.hpp
    ${RandLib_SOURCE_DIR}/math/GammaMath.hpp
    ${RandLib_SOURCE_DIR}/math/NumericMath.hpp
    ${RandLib_SOURCE_DIR}/math/RandMath.hpp

    # Abstract Distribution Classes
    ${RandLib_SOURCE_DIR}/distributions/ContinuousDistributions.hpp
    ${RandLib_SOURCE_DIR}/distributions/DiscreteDistributions.hpp
    ${RandLib_SOURCE_DIR}/distributions/ProbabilityDistribution.hpp

    # Bivariate Distributions
    ${RandLib_SOURCE_DIR}/distributions/bivariate/BivariateNormalRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/bivariate/NormalInverseGammaRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/bivariate/TrinomialRand.hpp

    # Univariate Distributions
    ## Helper
    ${RandLib_SOURCE_DIR}/distributions/univariate/BasicRandGenerator.hpp

    ## Abstract Parents
    ${RandLib_SOURCE_DIR}/distributions/univariate/ExponentialFamily.hpp

    ## Continuous
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/BetaPrimeRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/BetaRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/CauchyRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/DegenerateRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/ExponentiallyModifiedGaussianRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/ExponentialRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/FisherFRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/FrechetRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/GammaRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/GeometricStableRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/GumbelRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/InverseGammaRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/InverseGaussianRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/IrwinHallRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/KolmogorovSmirnovRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LaplaceRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LevyRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LogisticRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LogNormalRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/MarchenkoPasturRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NakagamiRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NoncentralChiSquaredRand.hpp
    #${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NoncentralTRand.hpp # This was left out of original library...
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NormalRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/ParetoRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/PlanckRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/RaisedCosineRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/SechRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/StableRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/StudentTRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/TriangularRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/UniformRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/WeibullRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/WignerSemicircleRand.hpp

    ### Circular
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/circular/VonMisesRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/circular/WrappedExponentialRand.hpp

    ## Discrete
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BernoulliRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BetaBinomialRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BinomialRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/CategoricalRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/GeometricRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/HyperGeometricRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/LogarithmicRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/NegativeBinomialRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/NegativeHyperGeometricRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/PoissonRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/RademacherRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/SkellamRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/UniformDiscreteRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/YuleRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/ZetaRand.hpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/ZipfRand.hpp

    ## Singular
    ${RandLib_SOURCE_DIR}/distributions/univariate/singular/CantorRand.hpp
)

set(RANDLIB_SRCS
    # Math
    ${RandLib_SOURCE_DIR}/math/BetaMath.cpp
    ${RandLib_SOURCE_DIR}/math/GammaMath.cpp
    ${RandLib_SOURCE_DIR}/math/NumericMath.cpp
    ${RandLib_SOURCE_DIR}/math/RandMath.cpp

    # Abstract Distribution
    ${RandLib_SOURCE_DIR}/distributions/ProbabilityDistribution.cpp

    # Univariate Distributions
    ## Helper
    ${RandLib_SOURCE_DIR}/distributions/univariate/BasicRandGenerator.cpp

    ## Abstract Parents
    ${RandLib_SOURCE_DIR}/distributions/univariate/ExponentialFamily.cpp

    ## Continuous
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/GeometricStableRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/GumbelRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/InverseGammaRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/InverseGaussianRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/IrwinHallRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/KolmogorovSmirnovRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LaplaceRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LevyRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LogisticRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/LogNormalRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/MarchenkoPasturRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NakagamiRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NoncentralChiSquaredRand.cpp
    #${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NoncentralTRand.cpp # This was left out of original library...
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/NormalRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/ParetoRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/PlanckRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/RaisedCosineRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/SechRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/StableRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/StudentTRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/TriangularRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/UniformRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/WeibullRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/continuous/WignerSemicircleRand.cpp

    ## Discrete
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BernoulliRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BetaBinomialRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/BinomialRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/CategoricalRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/GeometricRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/HyperGeometricRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/LogarithmicRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/NegativeBinomialRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/NegativeHyperGeometricRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/PoissonRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/RademacherRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/SkellamRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/UniformDiscreteRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/YuleRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/ZetaRand.cpp
    ${RandLib_SOURCE_DIR}/distributions/univariate/discrete/ZipfRand.cpp

    ## Singular
    ${RandLib_SOURCE_DIR}/distributions/univariate/singular/CantorRand.cpp
)

target_sources(RandLib
  PRIVATE
    ${RANDLIB_HDRS}
    ${RANDLIB_SRCS}
)

set(SEARCH_INCLUDE_DIR ${RandLib_SOURCE_DIR})
target_include_directories(RandLib
  PUBLIC
  $<BUILD_INTERFACE:${SEARCH_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
