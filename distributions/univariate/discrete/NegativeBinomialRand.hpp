#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/continuous/BetaRand.hpp"
#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"
#include "distributions/univariate/discrete/PoissonRand.hpp"

namespace randlib
{
/**
 * @brief The NegativeBinomialDistribution class <BR>
 * Abstract class for Negative binomial distribution
 *
 * P(X = k) = C(k + r - 1, k) p^r (1-p)^k
 *
 * Notation: X ~ NB(r, p)
 *
 * Related distributions: <BR>
 * If X ~ NB(1, p), then X ~ Geometric(p)
 * If Y ~ Γ(r, p / (1 - p), then Po(Y) ~ NB(r, p)
 */
template <typename IntType = int, typename T = double>
class RANDLIB_EXPORT NegativeBinomialDistribution : public randlib::DiscreteDistribution<IntType>
{
protected:
  double p = 0.5;            ///< probability of failure
  double q = 0.5;            ///< probability of success
  double logProb = -M_LN2;   ///< log(p)
  double log1mProb = -M_LN2; ///< log(q)

  NegativeBinomialDistribution(T number = 1, double probability = 0.5)
  {
    SetParameters(number, probability);
  }

public:
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

  inline double GetProbability() const
  {
    return p;
  }

  inline T GetNumber() const
  {
    return r;
  }

  double P(const IntType& k) const override
  {
    return (k < 0) ? 0.0 : std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
    if(k < 0)
      return -INFINITY;
    double y = std::lgammal(r + k);
    y -= RandMath::lfact(k);
    y += k * log1mProb;
    y += pdfCoef;
    return y;
  }

  double F(const IntType& k) const override
  {
    if(k < 0)
      return 0.0;
    int kp1 = k + 1;
    double logBetaFun = RandMath::logBeta(r, kp1);
    return RandMath::ibeta(p, r, kp1, logBetaFun, logProb, log1mProb);
  }

  double S(const IntType& k) const override
  {
    if(k < 0)
      return 0.0;
    int kp1 = k + 1;
    double logBetaFun = RandMath::logBeta(kp1, r);
    return RandMath::ibeta(q, kp1, r, logBetaFun, log1mProb, logProb);
  }

  IntType Variate() const override
  {
    GENERATOR_ID genId = GetIdOfUsedGenerator();
    if(genId == TABLE)
      return variateByTable();
    return (genId == EXPONENTIAL) ? variateThroughExponential() : variateThroughGammaPoisson();
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
    GENERATOR_ID genId = GetIdOfUsedGenerator();
    if(genId == TABLE)
    {
      for(IntType& var : outputData)
        var = variateByTable();
    }
    else if(genId == EXPONENTIAL)
    {
      for(IntType& var : outputData)
        var = variateThroughExponential();
    }
    else
    {
      for(IntType& var : outputData)
        var = variateThroughGammaPoisson();
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    GammaRV.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return qDivP * r;
  }

  long double Variance() const override
  {
    return qDivP * r / p;
  }

  IntType Mode() const override
  {
    return (r > 1) ? std::floor((r - 1) * qDivP) : 0;
  }

  long double Skewness() const override
  {
    return (1 + q) / std::sqrt(q * r);
  }

  long double ExcessKurtosis() const override
  {
    long double kurtosis = p / qDivP;
    kurtosis += 6;
    return kurtosis / r;
  }

  /**
   * @fn FitProbabilityBayes
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  BetaRand<> FitProbabilityBayes(const std::vector<IntType>& sample, const BetaDistribution<>& priorDistribution, bool MAP = false)
  {
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    int n = sample.size();
    double alpha = priorDistribution.GetAlpha();
    double beta = priorDistribution.GetBeta();
    BetaRand<> posteriorDistribution(alpha + r * n, beta + this->GetSampleSum(sample));
    SetParameters(r, MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

protected:
  void SetParameters(T number, double probability)
  {
    if(r <= 0.0)
      throw std::invalid_argument("Negative-Binomial distribution: number parameter should be positive");
    if(probability <= 0.0 || probability >= 1.0)
      throw std::invalid_argument("Negative-Binomial distribution: probability "
                                  "parameter should be in interval (0, 1)");
    r = (number > 0) ? number : 1;
    p = probability;
    q = 1.0 - p;
    logProb = std::log(p);
    log1mProb = std::log1pl(-p);
    GammaRV.SetParameters(r, p / q);
    qDivP = GammaRV.GetScale();
    pdfCoef = r * logProb;
    pdfCoef -= GammaRV.GetLogGammaShape();

    if(GetIdOfUsedGenerator() == TABLE)
    {
      /// table method
      table[0] = p;
      double prod = p;
      for(int i = 1; i < tableSize; ++i)
      {
        prod *= q;
        table[i] = table[i - 1] + prod;
      }
    }
  }

  enum GENERATOR_ID
  {
    TABLE,
    EXPONENTIAL,
    GAMMA_POISSON
  };

  /**
   * @fn GetIdOfUsedGenerator
   * If r is small, we use two different generators for two different cases:
   * If p < 0.08 then the tail is too heavy (probability to be in main body is
   * smaller than 0.75), then we return highest integer, smaller than variate
   * from exponential distribution. Otherwise we choose table method
   * @return id of generator
   */
  GENERATOR_ID GetIdOfUsedGenerator() const
  {
    if((r < 10 || GammaRV.Mean() > 10) && r == std::round(r))
      return (p < 0.08) ? EXPONENTIAL : TABLE;
    return GAMMA_POISSON;
  }

  IntType variateGeometricByTable() const
  {
    double U = UniformRand<double>::StandardVariate(this->localRandGenerator);
    /// handle tail by recursion
    if(U > table[tableSize - 1])
      return tableSize + variateGeometricByTable();
    /// handle the main body
    IntType x = 0;
    while(U > table[x])
      ++x;
    return x;
  }

  IntType variateGeometricThroughExponential() const
  {
    float X = -randlib::ExponentialRand<float>::StandardVariate(this->localRandGenerator) / log1mProb;
    return std::floor(X);
  }

private:
  T r = 1;                 ///< number of failures until the experiment is stopped
  double pdfCoef = -M_LN2; ///< coefficient for faster pdf calculation
  double qDivP = 1;        ///< q / p
  static constexpr int tableSize = 16;
  double table[tableSize];
  GammaRand<float> GammaRV{};

  IntType variateByTable() const
  {
    IntType var = 0;
    for(int i = 0; i < r; ++i)
    {
      var += variateGeometricByTable();
    }
    return var;
  }

  IntType variateThroughExponential() const
  {
    float X = -ExponentialRand<float>::StandardVariate(this->localRandGenerator) / log1mProb;
    return std::floor(X);
  }

  IntType variateThroughGammaPoisson() const
  {
    return PoissonRand<IntType>::Variate(GammaRV.Variate(), this->localRandGenerator);
  }

  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> denominator(1.0 - q * std::cos(t), -q * std::sin(t));
    return std::pow(p / denominator, r);
  }
};

/**
 * @brief The NegativeBinomialRand class <BR>
 * Negative binomial distribution
 */
template <typename IntType = int, typename T = double>
class RANDLIB_EXPORT NegativeBinomialRand : public NegativeBinomialDistribution<IntType, T>
{
public:
  NegativeBinomialRand(T number = 1, double probability = 0.5)
  : NegativeBinomialDistribution<IntType, T>(number, probability)
  {
  }

  String Name() const override
  {
    if(std::is_integral_v<T>)
      return "Pascal(" + this->toStringWithPrecision(this->GetNumber()) + ", " + this->toStringWithPrecision(this->GetProbability()) + ")";
    return "Polya(" + this->toStringWithPrecision(this->GetNumber()) + ", " + this->toStringWithPrecision(this->GetProbability()) + ")";
  }

  using NegativeBinomialDistribution<IntType, T>::SetParameters;

  static constexpr char TOO_SMALL_VARIANCE[] = "Sample variance should be greater than sample mean";

  /**
   * @fn Fit
   * set number and probability, estimated via maximum-likelihood method
   * @param sample
   */
  void Fit(const std::vector<IntType>& sample)
  {
    /// Check positivity of sample
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    /// Initial guess by method of moments
    DoublePair stats = this->GetSampleMeanAndVariance(sample);
    double mean = stats.first, variance = stats.second;
    /// Method can't be applied in case of variance smaller than mean
    if(variance <= mean)
      throw std::invalid_argument(this->fitErrorDescription(this->NOT_APPLICABLE, this->TOO_SMALL_VARIANCE));
    double guess = mean * mean / (variance - mean);
    size_t n = sample.size();
    if(!RandMath::findRootNewtonFirstOrder<double>(
           [sample, mean, n](double x) {
             double first = 0.0, second = 0.0;
             for(const IntType& var : sample)
             {
               first += RandMath::digamma(var + x);
               second += RandMath::trigamma(var + x);
             }
             first -= n * (RandMath::digammamLog(x) + std::log(x + mean));
             second -= n * (RandMath::trigamma(x) - mean / (x * (mean + x)));
             return DoublePair(first, second);
           },
           guess))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding algorithm"));
    if(guess <= 0.0)
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, "Number should be positive, but returned value is " + this->toStringWithPrecision(guess)));
    SetParameters(guess, guess / (guess + mean));
  }
};

template <typename IntType = int>
using PascalRand = NegativeBinomialRand<IntType, int>;

template <typename IntType = int>
using PolyaRand = NegativeBinomialRand<IntType, double>;
} // namespace randlib
