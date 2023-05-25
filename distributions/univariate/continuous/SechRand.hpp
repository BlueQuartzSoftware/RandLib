#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/CauchyRand.hpp"

namespace randlib
{
/**
 * @brief The SechRand class <BR>
 * Hyperbolic secant distribution
 *
 * Notation: X ~ Sech
 */
template <typename RealType = double>
class RANDLIB_EXPORT SechRand : public randlib::ContinuousDistribution<RealType>
{
public:
  SechRand() = default;

  String Name() const override
  {
    return "Hyperbolic secant";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::INFINITE_T;
  }

  RealType MinValue() const override
  {
    return -INFINITY;
  }

  RealType MaxValue() const override
  {
    return INFINITY;
  }

  double f(const RealType& x) const override
  {
    return 0.5 / std::cosh(M_PI_2 * x);
  }

  double logf(const RealType& x) const override
  {
    return M_PI_2 * x - RandMath::softplus(M_PI * x);
  }

  double F(const RealType& x) const override
  {
    double y = std::exp(M_PI_2 * x);
    return M_2_PI * RandMath::atan(y);
  }

  RealType Variate() const override
  {
    RealType y = std::fabs(CauchyRand<RealType>::StandardVariate(this->localRandGenerator));
    return M_2_PI * std::log(y);
  }

  long double Mean() const override
  {
    return 0.0;
  }

  long double Variance() const override
  {
    return 1.0;
  }

  RealType Median() const override
  {
    return 0.0;
  }

  RealType Mode() const override
  {
    return 0.0;
  }

  long double Skewness() const override
  {
    return 0.0;
  }

  long double ExcessKurtosis() const override
  {
    return 2.0;
  }

  long double Entropy() const
  {
    return 2.0 * M_2_PI * M_CATALAN;
  }

private:
  RealType quantileImpl(double p) const
  {
    RealType x = M_PI_2 * p;
    x = std::tan(x);
    x = std::log(x);
    return M_2_PI * x;
  }

  RealType quantileImpl1m(double p) const
  {
    RealType x = M_PI_2 * p;
    x = std::tan(x);
    x = -std::log(x);
    return M_2_PI * x;
  }

  std::complex<double> CFImpl(double t) const override
  {
    return 1.0 / std::cosh(t);
  }
};
} // namespace randlib
