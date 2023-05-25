#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The PoissonRand class <BR>
 * Poisson distribution
 *
 * P(X = k) = λ^k * exp(-λ) / k!
 *
 * Notation: X ~ Po(λ)
 */
template <typename IntType = int>
class RANDLIB_EXPORT PoissonRand : public randlib::DiscreteDistribution<IntType>, public ExponentialFamily<IntType, double>
{
  double lambda = 1;      ///< rate λ
  double logLambda = 0;   ///< ln(λ)
  double Fmu = 2 * M_1_E; ///< P(X < [λ])
  double Pmu = M_1_E;     ///< P(X = [λ])

  double mu = 1, delta = 6;
  double zeta{};
  long double c1{}, c2{}, c3{}, c4{}, c{};
  double sqrtMu = 1, sqrtMupHalfDelta = 2;
  double lfactMu = 0;

public:
  explicit PoissonRand(double rate = 1.0)
  {
    SetRate(rate);
  }

  String Name() const override
  {
    return "Poisson(" + this->toStringWithPrecision(GetRate()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  IntType MinValue() const override
  {
    return 0;
  }

  IntType MaxValue() const override
  {
    return std::numeric_limits<IntType>::max();
  }

  void SetRate(double rate)
  {
    if(rate <= 0.0)
      throw std::invalid_argument("Poisson distribution: rate should be positive, but it's equal to " + std::to_string(rate));
    lambda = rate;

    logLambda = std::log(lambda);
    mu = std::floor(lambda);
    Fmu = F(mu);
    Pmu = this->P(mu);

    if(!generateByInversion())
      SetGeneratorConstants();
  }

  inline double GetRate() const
  {
    return lambda;
  }

  double SufficientStatistic(IntType x) const override
  {
    return x;
  }

  double SourceParameters() const override
  {
    return lambda;
  }

  double SourceToNatural(double sourceParameters) const override
  {
    return std::log(sourceParameters);
  }

  double NaturalParameters() const override
  {
    return logLambda;
  }

  double LogNormalizer(double theta) const override
  {
    return std::exp(theta);
  }

  double LogNormalizerGradient(double theta) const override
  {
    return std::exp(theta);
  }

  double CarrierMeasure(IntType x) const override
  {
    return -RandMath::lfact(x);
  }

  double CrossEntropyAdjusted(double parameters) const override
  {
    return parameters - lambda * std::log(parameters);
  }

  double EntropyAdjusted() const override
  {
    return lambda - lambda * logLambda;
  }

  double logP(const IntType& k) const override
  {
    if(k < 0)
      return -INFINITY;
    double y = k * logLambda - lambda;
    return y - RandMath::lfact(k);
  }

  double F(const IntType& k) const override
  {
    return (k >= 0.0) ? RandMath::qgamma(k + 1, lambda, logLambda) : 0.0;
  }

  double S(const IntType& k) const override
  {
    return (k >= 0.0) ? RandMath::pgamma(k + 1, lambda, logLambda) : 1.0;
  }

  IntType Variate() const override
  {
    return generateByInversion() ? variateInversion() : variateRejection();
  }

  static IntType Variate(double rate, RandGenerator& randGenerator = ProbabilityDistribution<IntType>::staticRandGenerator)
  {
    /// check validness of parameter
    if(rate <= 0.0)
      throw std::invalid_argument("Poisson distribution: rate should be positive");
    if(rate > 1000)
    {
      /// approximate with normal distribution
      float X = NormalRand<float>::StandardVariate(randGenerator);
      return std::floor(rate + std::sqrt(rate) * X);
    }
    int k = -1;
    double s = 0;
    do
    {
      s += randlib::ExponentialRand<double>::StandardVariate(randGenerator);
      ++k;
    } while(s < rate);
    return k;
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
    if(generateByInversion())
    {
      for(IntType& var : outputData)
        var = variateInversion();
    }
    else
    {
      for(IntType& var : outputData)
        var = variateRejection();
    }
  }

  long double Mean() const override
  {
    return lambda;
  }

  long double Variance() const override
  {
    return lambda;
  }

  IntType Median() const override
  {
    /// this value is approximate
    return std::max(std::floor(lambda + 1.0 / 3 - 0.02 / lambda), 0.0);
  }

  IntType Mode() const override
  {
    if(RandMath::areClose(mu, lambda))
    {
      return (Pmu < this->P(mu + 1)) ? mu + 1 : mu;
    }
    return mu;
  }

  long double Skewness() const override
  {
    return 1.0 / std::sqrt(lambda);
  }

  long double ExcessKurtosis() const override
  {
    return 1.0 / lambda;
  }

  /**
   * @fn Fit
   * fit rate λ via maximum-likelihood method
   * @param sample
   */
  void Fit(const std::vector<IntType>& sample)
  {
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    SetRate(this->GetSampleMean(sample));
  }

  /**
   * @brief Fit
   * @param sample
   * @param confidenceInterval
   * @param significanceLevel
   */
  void Fit(const std::vector<IntType>& sample, DoublePair& confidenceInterval, double significanceLevel)
  {
    size_t n = sample.size();

    if(significanceLevel <= 0 || significanceLevel > 1)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_LEVEL, "Alpha is equal to " + this->toStringWithPrecision(significanceLevel)));

    Fit(sample);

    double halfAlpha = 0.5 * significanceLevel;
    ErlangRand<double> ErlangRV(n);
    confidenceInterval.first = ErlangRV.Quantile(halfAlpha);
    ErlangRV.SetShape(n + 1);
    confidenceInterval.second = ErlangRV.Quantile1m(halfAlpha);
  }

  /**
   * @fn FitBayes
   * fit rate λ via Bayes estimation
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior Gamma distribution
   */
  GammaRand<> FitBayes(const std::vector<IntType>& sample, const GammaDistribution<>& priorDistribution, bool MAP = false)
  {
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    double alpha = priorDistribution.GetShape();
    double beta = priorDistribution.GetRate();
    GammaRand<> posteriorDistribution(alpha + this->GetSampleSum(sample), beta + sample.size());
    SetRate(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

private:
  void SetGeneratorConstants()
  {
    delta = std::round(std::sqrt(2 * mu * std::log(M_1_PI * 128 * mu)));
    delta = std::max(6.0, std::min(mu, delta));

    c1 = std::sqrt(0.5 * M_PI * mu * M_E);

    c2 = 1.0 / (2 * mu + delta);
    c2 += 0.5 * (M_LNPI - M_LN2 + std::log(mu + 0.5 * delta));
    c2 = c1 + std::exp(c2);

    c3 = c2 + 1.0;
    c4 = c3 + 1.0;

    zeta = (4.0 * mu) / delta + 2;
    c = std::exp(-(2.0 + delta) / zeta);
    c *= zeta;
    c += c4;

    sqrtMu = std::sqrt(mu);
    sqrtMupHalfDelta = std::sqrt(mu + 0.5 * delta);
    lfactMu = RandMath::lfact(mu);
  }

  double acceptanceFunction(IntType X) const
  {
    if(X == 0)
      return 0.0;
    double q = X * logLambda;
    q += lfactMu;
    q -= RandMath::lfact(X + mu);
    return q;
  }

  bool generateByInversion() const
  {
    /// the inversion generator is much faster than rejection,
    /// however precision loss for large rate increases drastically
    return lambda < 10;
  }

  IntType variateRejection() const
  {
    size_t iter = 0;
    IntType X = 0;
    do
    {
      bool reject = false;
      float W = 0.0;
      float U = c * UniformRand<float>::StandardVariate(this->localRandGenerator);
      if(U <= c1)
      {
        float N = NormalRand<float>::StandardVariate(this->localRandGenerator);
        float Y = -std::fabs(N) * sqrtMu;
        X = std::floor(Y);
        if(X < -mu)
        {
          reject = true;
        }
        else
        {
          W = -0.5 * (N * N - 1.0);
        }
      }
      else if(U <= c2)
      {
        float N = NormalRand<float>::StandardVariate(this->localRandGenerator);
        float Y = 1.0 + std::fabs(N) * sqrtMupHalfDelta;
        X = std::ceil(Y);
        if(X > delta)
        {
          reject = true;
        }
        else
        {
          W = Y * (2.0 - Y) / (2.0 * mu + delta);
        }
      }
      else if(U <= c3)
      {
        return mu;
      }
      else if(U <= c4)
      {
        X = 1;
      }
      else
      {
        float V = randlib::ExponentialRand<float>::StandardVariate(this->localRandGenerator);
        float Y = delta + V * zeta;
        X = std::ceil(Y);
        W = -(2.0 + Y) / zeta;
      }

      if(!reject && W - ExponentialRand<float>::StandardVariate(this->localRandGenerator) <= acceptanceFunction(X))
      {
        return X + mu;
      }

    } while(++iter < ProbabilityDistribution<IntType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Poisson distribution: sampling failed");
  }

  IntType variateInversion() const
  {
    double U = UniformRand<double>::StandardVariate(this->localRandGenerator);
    IntType k = mu;
    double s = Fmu, p = Pmu;
    if(s < U)
    {
      do
      {
        ++k;
        p *= lambda / k;
        s += p;
      } while(s < U && p > 0);
    }
    else
    {
      s -= p;
      while(k > 0 && s > U)
      {
        p /= lambda / k;
        --k;
        s -= p;
      }
    }
    return k;
  }

  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> y(std::cos(t) - 1.0, std::sin(t));
    return std::exp(lambda * y);
  }
};
} // namespace randlib
