#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/Constants.hpp"
#include "math/RandMath.hpp"

#include "distributions/univariate/BasicRandGenerator.hpp"

#include "distributions/bivariate/NormalInverseGammaRand.hpp"
#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/continuous/InverseGammaRand.hpp"
#include "distributions/univariate/continuous/StableRand.hpp"
#include "distributions/univariate/continuous/StudentTRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

#include "external/log.hpp"
#include "external/sqrt.hpp"

#include <array>

namespace randlib
{
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
public:
  NormalRand(double location = 0, double variance = 1)
  : StableDistribution<RealType>(2.0, 0.0, 1.0, location)
  {
    SetVariance(variance);
  }

  String Name() const override
  {
    return "Normal(" + this->toStringWithPrecision(this->GetLocation()) + ", " + this->toStringWithPrecision(this->Variance()) + ")";
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Normal distribution: scale should be positive, but it's equal to " + std::to_string(scale));
    sigma = scale;
    StableDistribution<RealType>::SetScale(sigma * M_SQRT1_2);
  }

  void SetVariance(double variance)
  {
    if(variance <= 0.0)
      throw std::invalid_argument("Variance of Normal distribution should be "
                                  "positive, but it's equal to " +
                                  std::to_string(variance));
    SetScale(std::sqrt(variance));
  }

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

  DoublePair SufficientStatistic(RealType x) const override
  {
    return {x, x * x};
  }

  DoublePair SourceParameters() const override
  {
    return {this->mu, this->sigma * this->sigma};
  }

  DoublePair SourceToNatural(DoublePair sourceParameters) const override
  {
    double mean = sourceParameters.first;
    double scaleSq = sourceParameters.second;
    return {mean / scaleSq, -0.5 / scaleSq};
  }

  double LogNormalizer(DoublePair theta) const override
  {
    double theta1 = theta.first, theta2 = theta.second;
    double F = -0.25 * theta1 * theta1 / theta2;
    F += 0.5 * std::log(-M_PI / theta2);
    return F;
  }

  DoublePair LogNormalizerGradient(DoublePair theta) const override
  {
    double theta1 = theta.first, theta2 = theta.second;
    double grad1 = -0.5 * theta1 / theta2;
    double grad2 = -0.5 / theta2 + grad1 * grad1;
    return {grad1, grad2};
  }

  double CarrierMeasure(RealType) const override
  {
    return 0.0;
  }

  double f(const RealType& x) const override
  {
    return this->pdfNormal(x);
  }

  double logf(const RealType& x) const override
  {
    return this->logpdfNormal(x);
  }

  double F(const RealType& x) const override
  {
    return this->cdfNormal(x);
  }

  double S(const RealType& x) const override
  {
    return this->cdfNormalCompl(x);
  }

  RealType Variate() const override
  {
    return this->mu + sigma * StandardVariate(this->localRandGenerator);
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    /// Ziggurat algorithm by George Marsaglia using 256 strips
    size_t iter = 0;
    do
    {
      unsigned long long B = randGenerator.Variate();
      int stairId = B & 255;
      RealType x = UniformRand<RealType>::StandardVariate(randGenerator) * ziggurat[stairId].second; /// Get horizontal coordinate
      if(x < ziggurat[stairId + 1].second)
        return ((signed)B > 0) ? x : -x;
      if(stairId == 0) /// handle the base layer
      {
        static thread_local RealType z = -1;
        if(z > 0) /// we don't have to generate another exponential variable as
                  /// we already have one
        {
          x = ExponentialRand<RealType>::StandardVariate(randGenerator) / ziggurat[1].second;
          z -= 0.5 * x * x;
        }
        if(z <= 0) /// if previous generation wasn't successful
        {
          do
          {
            x = ExponentialRand<RealType>::StandardVariate(randGenerator) / ziggurat[1].second;
            z = ExponentialRand<RealType>::StandardVariate(randGenerator) - 0.5 * x * x; /// we storage this value as after acceptance it
                                                                                         /// becomes exponentially distributed
          } while(z <= 0);
        }
        x += ziggurat[1].second;
        return ((signed)B > 0) ? x : -x;
      }
      /// handle the wedges of other stairs
      RealType height = ziggurat[stairId].first - ziggurat[stairId - 1].first;
      if(ziggurat[stairId - 1].first + height * UniformRand<RealType>::StandardVariate(randGenerator) < std::exp(-.5 * x * x))
        return ((signed)B > 0) ? x : -x;
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Normal distribution: sampling failed");
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    for(RealType& var : outputData)
      var = this->Variate();
  }

  long double Moment(size_t n) const
  {
    if(n == 0)
      return 1;
    return (n & 1) ? std::exp(n * this->GetLogScale() + RandMath::ldfact(n - 1)) : 0.0;
  }

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
  void FitLocation(const std::vector<RealType>& sample)
  {
    this->SetLocation(this->GetSampleMean(sample));
  }

  /**
   * @fn FitLocation
   * set location, returned by maximium-likelihood estimator
   * and return confidenceInterval for given significance level
   * @param sample
   * @param confidenceInterval
   * @param significanceLevel
   */
  void FitLocation(const std::vector<RealType>& sample, DoublePair& confidenceInterval, double significanceLevel)
  {
    if(significanceLevel <= 0 || significanceLevel > 1)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_LEVEL, "Input level is equal to " + this->toStringWithPrecision(significanceLevel)));

    FitLocation(sample);

    int n = sample.size();
    NormalRand<RealType> NormalRV(0, 1);
    double halfAlpha = 0.5 * significanceLevel;
    double interval = NormalRV.Quantile1m(halfAlpha) * sigma / std::sqrt(n);
    confidenceInterval.first = this->mu - interval;
    confidenceInterval.second = this->mu + interval;
  }

  /**
   * @fn FitVariance
   * set variance, returned by maximium-likelihood estimator
   * @param sample
   */
  void FitVariance(const std::vector<RealType>& sample)
  {
    this->SetVariance(this->GetSampleVariance(sample, this->mu));
  }

  /**
   * @fn FitVariance
   * @param sample
   * @param confidenceInterval
   * @param significanceLevel
   * @param unbiased
   */
  void FitVariance(const std::vector<RealType>& sample, DoublePair& confidenceInterval, double significanceLevel, bool unbiased = false)
  {
    if(significanceLevel <= 0 || significanceLevel > 1)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_LEVEL, "Input level is equal to " + this->toStringWithPrecision(significanceLevel)));

    FitVariance(sample);

    size_t n = sample.size();
    double halfAlpha = 0.5 * significanceLevel;
    ChiSquaredRand ChiSqRV(n);
    double numerator = sigma * sigma * (unbiased ? n : (n - 1));
    confidenceInterval.first = numerator / ChiSqRV.Quantile1m(halfAlpha);
    confidenceInterval.second = numerator / ChiSqRV.Quantile(halfAlpha);
  }

  /**
   * @fn FitScale
   * set scale, returned via maximum-likelihood estimation or unbiased estimator
   * (which might be different from the unbiased estimator of variance)
   * @param sample
   * @param unbiased
   */
  void FitScale(const std::vector<RealType>& sample, bool unbiased = false)
  {
    if(!unbiased)
      return FitVariance(sample);
    size_t n = sample.size();
    double halfN = 0.5 * n;
    double s = this->GetSampleVariance(sample, this->mu);
    s *= halfN;
    s = 0.5 * std::log(s);
    s -= std::lgammal(halfN + 0.5);
    s += std::lgammal(halfN);
    SetScale(std::exp(s));
  }

  /**
   * @fn Fit
   * set parameters, returned by maximium-likelihood estimator if unbiased =
   * false, otherwise set parameters via UMVU estimator
   * @param sample
   * @param unbiased
   */
  void Fit(const std::vector<RealType>& sample, bool unbiased = false)
  {
    double adjustment = 1.0;
    if(unbiased)
    {
      size_t n = sample.size();
      if(n <= 1)
        throw std::invalid_argument(this->fitErrorDescription(this->TOO_FEW_ELEMENTS, "There should be at least 2 elements"));
      adjustment = static_cast<double>(n) / (n - 1);
    }
    DoublePair stats = this->GetSampleMeanAndVariance(sample);
    this->SetLocation(stats.first);
    this->SetVariance(stats.second * adjustment);
  }

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
  void Fit(const std::vector<RealType>& sample, DoublePair& confidenceIntervalForMean, DoublePair& confidenceIntervalForVariance, double significanceLevel, bool unbiased = false)
  {
    if(significanceLevel <= 0 || significanceLevel > 1)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_LEVEL, "Input level is equal to " + this->toStringWithPrecision(significanceLevel)));

    Fit(sample, unbiased);

    size_t n = sample.size();
    double sigmaAdj = unbiased ? sigma : (sigma * n) / (n - 1);
    /// calculate confidence interval for mean
    double halfAlpha = 0.5 * significanceLevel;
    StudentTRand<RealType> tRV(n - 1);
    double interval = tRV.Quantile1m(halfAlpha) * sigmaAdj / std::sqrt(n);
    confidenceIntervalForMean.first = this->mu - interval;
    confidenceIntervalForMean.second = this->mu + interval;

    /// calculate confidence interval for variance
    ChiSquaredRand<RealType> ChiSqRV(n - 1);
    double numerator = (n - 1) * sigmaAdj * sigmaAdj;
    confidenceIntervalForVariance.first = numerator / ChiSqRV.Quantile1m(halfAlpha);
    confidenceIntervalForVariance.second = numerator / ChiSqRV.Quantile(halfAlpha);
  }

  /**
   * @fn FitLocationBayes
   * set location, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  NormalRand<RealType> FitLocationBayes(const std::vector<RealType>& sample, const NormalRand<RealType>& priorDistribution, bool MAP = false)
  {
    double mu0 = priorDistribution.GetLocation();
    double tau0 = priorDistribution.GetPrecision();
    double tau = GetPrecision();
    double numerator = this->GetSampleSum(sample) * tau + tau0 * mu0;
    double denominator = sample.size() * tau + tau0;
    NormalRand<RealType> posteriorDistribution(numerator / denominator, 1.0 / denominator);
    this->SetLocation(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

  /**
   * @fn FitVarianceBayes
   * set variance, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  InverseGammaRand<RealType> FitVarianceBayes(const std::vector<RealType>& sample, const InverseGammaRand<RealType>& priorDistribution, bool MAP = false)
  {
    double halfN = 0.5 * sample.size();
    double alphaPrior = priorDistribution.GetShape();
    double betaPrior = priorDistribution.GetRate();
    double alphaPosterior = alphaPrior + halfN;
    double betaPosterior = betaPrior + halfN * this->GetSampleVariance(sample, this->mu);
    InverseGammaRand<RealType> posteriorDistribution(alphaPosterior, betaPosterior);
    SetVariance(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

  /**
   * @fn FitBayes
   * set parameters, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  NormalInverseGammaRand<RealType> FitBayes(const std::vector<RealType>& sample, const NormalInverseGammaRand<RealType>& priorDistribution, bool MAP = false)
  {
    size_t n = sample.size();
    double alphaPrior = priorDistribution.GetShape();
    double betaPrior = priorDistribution.GetRate();
    double muPrior = priorDistribution.GetLocation();
    double lambdaPrior = priorDistribution.GetPrecision();
    DoublePair stats = this->GetSampleMeanAndVariance(sample);
    double lambdaPosterior = lambdaPrior + n;
    double muPosterior = (lambdaPrior * muPrior + n * stats.first) / lambdaPosterior;
    double halfN = 0.5 * n;
    double alphaPosterior = alphaPrior + halfN;
    double aux = muPrior - stats.first;
    double betaPosterior = betaPrior + halfN * (stats.second + lambdaPrior / lambdaPosterior * aux * aux);
    NormalInverseGammaRand<RealType> posteriorDistribution(muPosterior, lambdaPosterior, alphaPosterior, betaPosterior);
    DoublePair newParams = MAP ? static_cast<DoublePair>(posteriorDistribution.Mode()) : static_cast<DoublePair>(posteriorDistribution.Mean());
    this->SetLocation(newParams.first);
    this->SetVariance(newParams.second);
    return posteriorDistribution;
  }

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

  RealType quantileImpl(double p) const override
  {
    return this->quantileNormal(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    return this->quantileNormal1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    return this->cfNormal(t);
  }
};
} // namespace randlib
