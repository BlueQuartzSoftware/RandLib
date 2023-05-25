#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

namespace randlib
{
/**
 * @brief The LogarithmicRand class <BR>
 * Logarithmic distribution
 *
 * P(X = k) = -p^k / [k log(1 - p)]
 *
 * X ~ Log(p)
 */
template <typename IntType = int>
class RANDLIB_EXPORT LogarithmicRand : public randlib::DiscreteDistribution<IntType>
{
  double p = 0.5;            ///< parameter of distribution
  double logProb = -M_LN2;   ///< log(p)
  double log1mProb = -M_LN2; ///< log(q)
public:
  explicit LogarithmicRand(double probability)
  {
    SetProbability(probability);
  }

  String Name() const override
  {
    return "Logarithmic(" + this->toStringWithPrecision(GetProbability()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  IntType MinValue() const override
  {
    return 1;
  }

  IntType MaxValue() const override
  {
    return std::numeric_limits<IntType>::max();
  }

  void SetProbability(double probability)
  {
    if(probability <= 0.0 || probability >= 1.0)
      throw std::invalid_argument("Logarithmic distribution: probability "
                                  "parameter should in interval (0, 1)");
    p = probability;
    logProb = std::log(p);
    log1mProb = std::log1pl(-p);
  }

  inline double GetProbability() const
  {
    return p;
  }

  double P(const IntType& k) const override
  {
    return (k < 1) ? 0.0 : -std::pow(p, k) / (k * log1mProb);
  }

  double logP(const IntType& k) const override
  {
    return (k < 1) ? -INFINITY : k * logProb - std::log(-k * log1mProb);
  }

  double F(const IntType& k) const override
  {
    return (k < 1) ? 0.0 : 1 + betaFun(k + 1) / log1mProb;
  }

  double S(const IntType& k) const override
  {
    return (k < 1) ? 1.0 : -betaFun(k + 1) / log1mProb;
  }

  IntType Variate() const override
  {
    /// Kemp's second accelerated generator
    /// p. 548, "Non-Uniform Random Variate Generation" by Luc Devroye
    float V = UniformRand<float>::StandardVariate(this->localRandGenerator);
    if(V >= p)
      return 1.0;
    float U = UniformRand<float>::StandardVariate(this->localRandGenerator);
    double y = -std::expm1l(U * log1mProb);
    if(V > y)
      return 1.0;
    if(V > y * y)
      return 2.0;
    return std::floor(1.0 + std::log(V) / std::log(y));
  }

  long double Mean() const override
  {
    return -p / (1.0 - p) / log1mProb;
  }

  long double Variance() const override
  {
    long double var = p / log1mProb + 1;
    var /= log1mProb;
    var *= p;
    long double q = 1.0 - p;
    return -var / (q * q);
  }

  IntType Mode() const override
  {
    return 1.0;
  }

private:
  /**
   * @fn betaFun
   * @param a
   * @return B(p, a, 0), where B(x, a, b) denotes incomplete beta function,
   * using series expansion (converges for x < 1)
   */
  double betaFun(IntType a) const
  {
    double denom = a + 1;
    double sum = p * p / (a + 2) + p / (a + 1) + 1.0 / a;
    double add = 1;
    int i = 3;
    do
    {
      add = std::exp(i * logProb) / (++denom);
      sum += add;
      ++i;
    } while(add > MIN_POSITIVE * sum);
    return std::exp(a * logProb) * sum;
  }

  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> y(std::cos(t), std::sin(t));
    y = std::log(1.0 - p * y);
    return y / log1mProb;
  }
};
} // namespace randlib
