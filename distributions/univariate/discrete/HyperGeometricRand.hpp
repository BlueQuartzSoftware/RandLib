#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/BernoulliRand.hpp"
#include "distributions/univariate/discrete/BetaBinomialRand.hpp"

namespace randlib
{
/**
 * @brief The HyperGeometricRand class <BR>
 * Hypergeometric distribution
 *
 * X ~ HG(N, K, n)
 */
template <typename IntType = int>
class RANDLIB_EXPORT HyperGeometricRand : public randlib::DiscreteDistribution<IntType>
{
  IntType N = 1;      ///< population size
  IntType K = 1;      /// number of possible successes
  IntType n = 1;      /// number of draws
  double pmfCoef = 1; ///< C(N, n)
  double p0 = 1;      ///< K/N

public:
  HyperGeometricRand(IntType totalSize = 1, IntType drawsNum = 1, IntType successesNum = 1)
  {
    SetParameters(totalSize, drawsNum, successesNum);
  }

  String Name() const override
  {
    return "Hypergeometric(" + this->toStringWithPrecision(N) + ", " + this->toStringWithPrecision(n) + ", " + this->toStringWithPrecision(K) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  IntType MinValue() const override
  {
    return std::max(static_cast<IntType>(0), n - N + K);
  }

  IntType MaxValue() const override
  {
    return std::min(n, K);
  }

  void SetParameters(IntType totalSize, IntType drawsNum, IntType successesNum)
  {
    if(totalSize <= 0 || drawsNum <= 0 || successesNum <= 0)
      throw std::invalid_argument("HyperGeometric distribution: all parameters should be positive");
    if(drawsNum > totalSize)
      throw std::invalid_argument("HyperGeometric distribution: total size "
                                  "should be greater than draws number");
    if(successesNum > totalSize)
      throw std::invalid_argument("HyperGeometric distribution: total size "
                                  "should be greater than successes number");

    N = totalSize;
    n = drawsNum;
    K = successesNum;

    p0 = static_cast<double>(K) / N;
    pmfCoef = RandMath::lfact(K);
    pmfCoef += RandMath::lfact(N - K);
    pmfCoef += RandMath::lfact(N - n);
    pmfCoef += RandMath::lfact(n);
    pmfCoef -= RandMath::lfact(N);
  }

  inline IntType GetTotalSize()
  {
    return N;
  }

  inline IntType GetDrawsNum()
  {
    return n;
  }

  inline IntType GetSuccessesNum()
  {
    return K;
  }

  double P(const IntType& k) const override
  {
    return (k < MinValue() || k > MaxValue()) ? 0.0 : std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
    if(k < MinValue() || k > MaxValue())
      return -INFINITY;
    double y = RandMath::lfact(k);
    y += RandMath::lfact(K - k);
    y += RandMath::lfact(n - k);
    y += RandMath::lfact(N - K - n + k);
    return pmfCoef - y;
  }

  double F(const IntType& k) const override
  {
    if(k < MinValue())
      return 0.0;
    IntType maxVal = MaxValue();
    if(k >= maxVal)
      return 1.0;
    if(k <= 0.5 * maxVal)
    {
      /// sum P(X = i) going forward until k
      double sum = 0;
      for(IntType i = 0; i <= k; ++i)
        sum += P(i);
      return sum;
    }
    /// going backwards is faster
    double sum = 1.0;
    for(IntType i = k + 1; i <= maxVal; ++i)
      sum -= P(i);
    return sum;
  }

  IntType Variate() const override
  {
    double p = p0;
    IntType sum = 0;
    for(int i = 1; i <= n; ++i)
    {
      if(BernoulliRand::Variate(p, this->localRandGenerator) && ++sum >= K)
        return sum;
      p = K - sum;
      p /= N - i;
    }
    return sum;
  }

  long double Mean() const override
  {
    return static_cast<double>(n * K) / N;
  }

  long double Variance() const override
  {
    long double numerator = n;
    numerator *= K;
    numerator *= N - K;
    numerator *= N - n;
    long double denominator = N;
    denominator *= N;
    denominator *= N - 1;
    return numerator / denominator;
  }

  IntType Mode() const override
  {
    double mode = (n + 1) * (K + 1);
    return std::floor(mode / (N + 2));
  }

  long double Skewness() const override
  {
    long double skewness = N - 1;
    skewness /= n;
    skewness /= K;
    skewness /= N - K;
    skewness /= N - n;
    skewness = std::sqrt(skewness);
    skewness *= N - 2 * K;
    skewness *= N - 2 * n;
    return skewness / (N - 2);
  }

  long double ExcessKurtosis() const override
  {
    long double numerator = N;
    numerator *= (N + 1);
    long double a1 = K;
    a1 *= N - K;
    long double a2 = n;
    a2 *= N - n;
    numerator -= 6 * (a1 + a2);
    numerator *= N;
    numerator *= N;
    numerator *= N - 1;
    long double aux = n;
    aux *= K;
    aux *= N - K;
    aux *= N - n;
    numerator += 6 * aux * (5 * N - 6);
    long double denominator = aux * (N - 2) * (N - 3);
    return numerator / denominator;
  }
};
} // namespace randlib
