#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/BasicRandGenerator.hpp"

#include "distributions/univariate/discrete/BernoulliRand.hpp"

namespace randlib
{
/**
 * @brief The RademacherRand class <BR>
 * Rademacher distribution
 *
 * P(X = k) = 0.5 * 1_{|k| = 1}
 *
 * Notation: X ~ Rademacher
 *
 * Related distributions: <BR>
 * If Y ~ Bernoulli(0.5), then 2Y - 1 ~ Rademacher
 */
class RANDLIB_EXPORT RademacherRand : public randlib::DiscreteDistribution<int>
{
public:
  RademacherRand() = default;

  String Name() const override
  {
    return "Rademacher";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  int MinValue() const override
  {
    return -1;
  }

  int MaxValue() const override
  {
    return 1;
  }

  double P(const int& k) const override
  {
    return (k == 1 || k == -1) ? 0.5 : 0.0;
  }

  double logP(const int& k) const override
  {
    return (k == 1 || k == -1) ? -M_LN2 : -INFINITY;
  }

  double F(const int& k) const override
  {
    if(k < -1)
      return 0;
    return (k < 1) ? 0.5 : 1.0;
  }

  int Variate() const override
  {
    return BernoulliRand::StandardVariate(this->localRandGenerator) ? 1 : -1;
  }

  long double Mean() const override
  {
    return 0;
  }

  long double Variance() const override
  {
    return 1;
  }

  int Median() const override
  {
    return -1;
  }

  int Mode() const override
  {
    /// any from {-1, 1}
    return Variate();
  }

  long double Skewness() const override
  {
    return 0.0;
  }

  long double ExcessKurtosis() const override
  {
    return -2.0;
  }

  double Entropy()
  {
    return M_LN2;
  }

private:
  int quantileImpl(double p) const override
  {
    return (p <= 0.5) ? -1 : 1;
  }

  int quantileImpl1m(double p) const override
  {
    return (p >= 0.5) ? -1 : 1;
  }

  std::complex<double> CFImpl(double t) const override
  {
    return std::cos(t);
  }
};
} // namespace randlib
