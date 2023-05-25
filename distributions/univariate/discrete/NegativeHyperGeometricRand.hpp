#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/BernoulliRand.hpp"

namespace randlib
{
/**
 * @brief The NegativeHyperGeometricRand class <BR>
 * Negative hypergeometric distribution
 *
 * Notation: X ~ NHG(N, M, m)
 */
template <typename IntType = int>
class RANDLIB_EXPORT NegativeHyperGeometricRand : public randlib::DiscreteDistribution<IntType>
{
  IntType N = 1;      ///< size of population
  IntType M = 1;      ///< total amount of successes
  IntType m = 1;      ///< limiting number of successes
  double pmfCoef = 1; ///< C(N, M)
  double p0 = 1;      ///< M / N

public:
  NegativeHyperGeometricRand(IntType totalSize = 1, IntType totalSuccessesNum = 1, IntType limitSuccessesNum = 1)
  {
    SetParameters(totalSize, totalSuccessesNum, limitSuccessesNum);
  }

  String Name() const override
  {
    return "Negative hypergeometric(" + this->toStringWithPrecision(N) + ", " + this->toStringWithPrecision(M) + ", " + this->toStringWithPrecision(m) + ")";
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
    return N - M;
  }

  void SetParameters(IntType totalSize, IntType totalSuccessesNum, IntType limitSuccessesNum)
  {
    if(totalSize <= 0 || totalSuccessesNum <= 0 || limitSuccessesNum <= 0)
      throw std::invalid_argument("Negative-HyperGeometric distribution: all "
                                  "parameters should be positive");
    if(totalSuccessesNum > totalSize)
      throw std::invalid_argument("Negative-HyperGeometric distribution: total size shouldn't be smaller "
                                  "than total successes number");
    if(limitSuccessesNum > totalSuccessesNum)
      throw std::invalid_argument("Negative-HyperGeometric distribution: total successes number "
                                  "shouldn't be smaller than limit successes number");

    N = totalSize;
    M = totalSuccessesNum;
    m = limitSuccessesNum;

    p0 = static_cast<double>(M) / N;
    pmfCoef = RandMath::lfact(M);
    pmfCoef += RandMath::lfact(N - M);
    pmfCoef -= RandMath::lfact(m - 1);
    pmfCoef -= RandMath::lfact(M - m);
    pmfCoef -= RandMath::lfact(N);
  }

  inline IntType GetTotalSize()
  {
    return N;
  }

  inline IntType GetTotalSuccessesNum()
  {
    return M;
  }

  inline IntType GetLimitSuccessesNum()
  {
    return m;
  }

  double P(const IntType& k) const override
  {
    return (k < MinValue() || k > MaxValue()) ? 0.0 : std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
    if(k < MinValue() || k > MaxValue())
      return -INFINITY;
    double p = RandMath::lfact(k + m - 1);
    p += RandMath::lfact(N - m - k);
    p -= RandMath::lfact(k);
    p -= RandMath::lfact(N - M - k);
    return p + pmfCoef;
  }

  double F(const IntType& k) const override
  {
    // relation with hypergeometric distribution can be used here instead
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
    IntType successesNum = 0;
    IntType num = 0;
    while(successesNum < m)
    {
      ++num;
      if(BernoulliRand::Variate(p, this->localRandGenerator) && ++successesNum == num - N + M)
        return N - M;
      p = M - successesNum;
      p /= N - num;
    }
    return num - successesNum;
  }

  long double Mean() const override
  {
    double mean = m;
    mean *= N - M;
    mean /= M + 1;
    return mean;
  }

  long double Variance() const override
  {
    double Mp1 = M + 1;
    double var = 1 - m / Mp1;
    var *= N + 1;
    var *= N - M;
    var /= Mp1 * (Mp1 + 1);
    return m * var;
  }
};
} // namespace randlib
