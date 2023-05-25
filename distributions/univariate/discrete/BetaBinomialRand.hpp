#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/BinomialRand.hpp"
#include "distributions/univariate/discrete/UniformDiscreteRand.hpp"

#include <functional>

namespace randlib
{
/**
 * @brief The BetaBinomialRand class <BR>
 * Beta-Binomial distribution
 *
 * Notation: X ~ BB(n, α, β)
 *
 * Related distributions: <BR>
 * If X ~ Binomial(n, p), where p ~ Beta(α, β), then X ~ BB(n, α, β)
 */
template <typename IntType = int>
class RANDLIB_EXPORT BetaBinomialRand : public randlib::DiscreteDistribution<IntType>
{
  IntType n = 1;      ///< number of experiments
  double pmfCoef = 0; ///< log(n!) - log(Γ(α + β + n)) - log(B(α, β))
  BetaRand<double> B{};

public:
  BetaBinomialRand(IntType number = 1, double shape1 = 1, double shape2 = 1)
  {
    SetParameters(number, shape1, shape2);
  }

  String Name() const override
  {
    return "Beta-Binomial(" + this->toStringWithPrecision(GetNumber()) + ", " + this->toStringWithPrecision(GetAlpha()) + ", " + this->toStringWithPrecision(GetBeta()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  IntType MinValue() const override
  {
    return 0;
  }

  IntType MaxValue() const override
  {
    return n;
  }

  void SetParameters(IntType number, double shape1, double shape2)
  {
    if(shape1 <= 0.0 || shape2 <= 0.0)
      throw std::invalid_argument("Beta-Binomial distribution: shape parameters should be positive");
    if(number <= 0)
      throw std::invalid_argument("Beta-Binomial distribution: number should be positive");
    n = number;
    B.SetShapes(shape1, shape2);
    pmfCoef = RandMath::lfact(n);
    pmfCoef -= std::lgammal(B.GetAlpha() + B.GetBeta() + n);
    pmfCoef -= B.GetLogBetaFunction();
  }

  inline IntType GetNumber() const
  {
    return n;
  }

  inline double GetAlpha() const
  {
    return B.GetAlpha();
  }

  inline double GetBeta() const
  {
    return B.GetBeta();
  }

  double P(const IntType& k) const override
  {
    return (k < 0 || k > n) ? 0.0 : std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
    if(k < 0 || k > n)
      return -INFINITY;
    double y = std::lgammal(k + B.GetAlpha());
    y += std::lgammal(n - k + B.GetBeta());
    y -= RandMath::lfact(k);
    y -= RandMath::lfact(n - k);
    return pmfCoef + y;
  }

  double F(const IntType& k) const override
  {
    if(k < 0)
      return 0.0;
    if(k >= n)
    {
      return 1.0;
    }
    double sum = 0.0;
    int i = 0;
    do
    {
      sum += P(i);
    } while(++i <= k);
    return sum;
  }

  IntType Variate() const override
  {
    return (B.GetAlpha() == 1 && B.GetBeta() == 1) ? VariateUniform() : VariateBeta();
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
    if(B.GetAlpha() == 1 && B.GetBeta() == 1)
    {
      for(IntType& var : outputData)
        var = VariateUniform();
    }
    else
    {
      for(IntType& var : outputData)
        var = VariateBeta();
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    B.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    double alpha = B.GetAlpha();
    double beta = B.GetBeta();
    return n * alpha / (alpha + beta);
  }

  long double Variance() const override
  {
    double alpha = B.GetAlpha();
    double beta = B.GetBeta();
    double alphaPBeta = alpha + beta;
    double numerator = n * alpha * beta * (alphaPBeta + n);
    double denominator = alphaPBeta * alphaPBeta;
    denominator *= (alphaPBeta + 1);
    return numerator / denominator;
  }

  IntType Mode() const override
  {
    IntType mode = (IntType)(n * B.Mode());
    double logPmode = this->logP(mode);
    if(this->logP(mode + 1) > logPmode)
      return mode + 1;
    if(this->logP(mode - 1) > logPmode)
      return mode - 1;
    return mode;
  }

  long double Skewness() const override
  {
    long double alpha = B.GetAlpha();
    long double beta = B.GetBeta();
    long double alphaPBeta = alpha + beta;
    long double res = (1 + alphaPBeta) / (n * alpha * beta * (alphaPBeta + n));
    res = std::sqrt(res);
    res *= (alphaPBeta + 2 * n) * (beta - alpha);
    res /= alphaPBeta + 2;
    return res;
  }

  long double ExcessKurtosis() const override
  {
    long double alpha = B.GetAlpha();
    long double beta = B.GetBeta();
    long double alphaPBeta = alpha + beta;
    long double alphaBetaN = alpha * beta * n;
    long double res = alpha * beta * (n - 2);
    res += 2 * (double)n * n;
    res -= alphaBetaN * (6 - n) / alphaPBeta;
    res -= 6 * alphaBetaN * n / (alphaPBeta * alphaPBeta);
    res *= 3;
    res += alphaPBeta * (alphaPBeta - 1 + 6 * n);
    res *= alphaPBeta * alphaPBeta * (1 + alphaPBeta);
    res /= (alphaBetaN * (alphaPBeta + 2) * (alphaPBeta + 3) * (alphaPBeta + n));
    return res - 3.0;
  }

private:
  IntType VariateUniform() const
  {
    return UniformDiscreteRand<IntType>::StandardVariate(0, n, this->localRandGenerator);
  }

  IntType VariateBeta() const
  {
    double p = B.Variate();
    return BinomialDistribution<IntType>::Variate(n, p, this->localRandGenerator);
  }
};
} // namespace randlib
