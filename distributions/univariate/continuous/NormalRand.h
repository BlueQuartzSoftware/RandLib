#ifndef NORMALRAND_H
#define NORMALRAND_H

#include "InverseGammaRand.h"
#include "StableRand.h"

#include "math/Constants.h"

#include "external/log.hpp"
#include "external/sqrt.hpp"

#include "distributions/bivariate/NormalInverseGammaRand.h"

#include <array>

/**
 * @brief The NormalRand class <BR>
 * Normal distribution
 *
 * f(x | μ, σ^2) = 1 / ((2 π σ^2)^(1/2) * exp(-(x - μ)^2 / (2 σ^2))
 *
 * Notation: X ~ N(μ, σ^2)
 *
 * Related distributions: <BR>
 * X ~ S(2, 0, σ/√2, μ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT NormalRand : public StableDistribution<RealType>, public ExponentialFamily<RealType, DoublePair>
{
private:
  template <typename T = void>
  struct NormalZiggurat
  {
    struct Ziggurat
    {
      std::array<LongDoublePair, 257> table = {};

      constexpr Ziggurat()
      {
        constexpr long double A = 4.92867323399e-3l; /// area under rectangle
        /// coordinates of the implicit rectangle in base layer
        table[0].first = 0.001260285930498597l;   /// exp(-x1);
        table[0].second = 3.9107579595370918075l; /// A / stairHeight[0];
        /// implicit value for the top layer
        table[257 - 1].second = 0.0l;
        table[1].second = 3.6541528853610088l;
        table[1].first = 0.002609072746106362l;
        for(size_t i = 2; i < 257 - 1; ++i)
        {
          /// such y_i that f(x_{i+1}) = y_i
          table[i].second = nonstd::sqrt(-2 * nonstd::log(table[i - 1].first));
          table[i].first = table[i - 1].first + A / table[i].second;
        }
      }
    };
    static constexpr Ziggurat STATIC_ZIGGURAT = Ziggurat();
  };

  static constexpr auto ziggurat = NormalZiggurat<>::STATIC_ZIGGURAT.table;

  double sigma = 1; ///< scale σ

public:
  NormalRand(double location = 0, double variance = 1);
  String Name() const override;

public:
  void SetScale(double scale);
  void SetVariance(double variance);
  /**
   * @fn GetScale
   * @return σ
   */
  inline double GetScale() const
  {
    return sigma;
  }
  /**
   * @fn GetLogScale
   * @return log(σ)
   */
  inline double GetLogScale() const
  {
    return StableDistribution<RealType>::GetLogScale() - 0.5 * M_LN2;
  }
  /**
   * @fn GetPrecision
   * @return 1/σ^2
   */
  inline double GetPrecision() const
  {
    return 1.0 / this->Variance();
  }

  DoublePair SufficientStatistic(RealType x) const override;
  DoublePair SourceParameters() const override;
  DoublePair SourceToNatural(DoublePair sourceParameters) const override;
  double LogNormalizer(DoublePair theta) const override;
  DoublePair LogNormalizerGradient(DoublePair theta) const override;
  double CarrierMeasure(RealType) const override;

  double f(const RealType& x) const override;
  double logf(const RealType& x) const override;
  double F(const RealType& x) const override;
  double S(const RealType& x) const override;
  RealType Variate() const override;
  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator);
  void Sample(std::vector<RealType>& outputData) const override;

private:
  RealType quantileImpl(double p) const override;
  RealType quantileImpl1m(double p) const override;

  std::complex<double> CFImpl(double t) const override;

public:
  long double Moment(size_t n) const;
  long double ThirdMoment() const override
  {
    return Moment(3);
  }
  long double FourthMoment() const override
  {
    return Moment(4);
  }

  /**
   * @fn FitLocation
   * set location, returned by maximium-likelihood estimator
   * @param sample
   */
  void FitLocation(const std::vector<RealType>& sample);

  /**
   * @fn FitLocation
   * set location, returned by maximium-likelihood estimator
   * and return confidenceInterval for given significance level
   * @param sample
   * @param confidenceInterval
   * @param significanceLevel
   */
  void FitLocation(const std::vector<RealType>& sample, DoublePair& confidenceInterval, double significanceLevel);

  /**
   * @fn FitVariance
   * set variance, returned by maximium-likelihood estimator
   * @param sample
   */
  void FitVariance(const std::vector<RealType>& sample);

  /**
   * @fn FitVariance
   * @param sample
   * @param confidenceInterval
   * @param significanceLevel
   * @param unbiased
   */
  void FitVariance(const std::vector<RealType>& sample, DoublePair& confidenceInterval, double significanceLevel, bool unbiased = false);

  /**
   * @fn FitScale
   * set scale, returned via maximum-likelihood estimation or unbiased estimator
   * (which might be different from the unbiased estimator of variance)
   * @param sample
   * @param unbiased
   */
  void FitScale(const std::vector<RealType>& sample, bool unbiased = false);

  /**
   * @fn Fit
   * set parameters, returned by maximium-likelihood estimator if unbiased =
   * false, otherwise set parameters via UMVU estimator
   * @param sample
   * @param unbiased
   */
  void Fit(const std::vector<RealType>& sample, bool unbiased = false);

  /**
   * @fn Fit
   * set parameters, returned by maximium-likelihood estimator if unbiased =
   * false, otherwise set parameters via UMVU estimator, and return confidence
   * intervals for given significance level
   * @param sample
   * @param confidenceIntervalForMean
   * @param confidenceIntervalForVariance
   * @param significanceLevel
   * @param unbiased
   */
  void Fit(const std::vector<RealType>& sample, DoublePair& confidenceIntervalForMean, DoublePair& confidenceIntervalForVariance, double significanceLevel, bool unbiased = false);

  /**
   * @fn FitLocationBayes
   * set location, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  NormalRand<RealType> FitLocationBayes(const std::vector<RealType>& sample, const NormalRand<RealType>& priorDistribution, bool MAP = false);

  /**
   * @fn FitVarianceBayes
   * set variance, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  InverseGammaRand<RealType> FitVarianceBayes(const std::vector<RealType>& sample, const InverseGammaRand<RealType>& priorDistribution, bool MAP = false);

  /**
   * @fn FitBayes
   * set parameters, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  NormalInverseGammaRand<RealType> FitBayes(const std::vector<RealType>& sample, const NormalInverseGammaRand<RealType>& priorDistribution, bool MAP = false);
};

#endif // NORMALRAND_H
