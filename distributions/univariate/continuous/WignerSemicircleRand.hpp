#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/BetaRand.hpp"

namespace randlib
{
/**
 * @brief The WignerSemicircleRand class <BR>
 * Wigner-Semicircle distribution
 *
 * Notation: X ~ Wigner-Sc(R)
 *
 * Related distributions:
 * If Y ~ Beta(1.5, 1.5), then R * (2Y - 1) ~ Wigner-Sc(R)
 */
template <typename RealType = double>
class RANDLIB_EXPORT WignerSemicircleRand : public randlib::ContinuousDistribution<RealType>
{
  RealType R = 1;    ///< radius
  double RSq = 1;    ///< R^2
  double logRSq = 0; /// log(R^2)
  BetaRand<RealType> X{1.5, 1.5};

public:
  explicit WignerSemicircleRand(double radius)
  {
      SetRadius(radius);
  }

  String Name() const override
  {
      return "Wigner Semicircle(" + this->toStringWithPrecision(GetRadius()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return -R;
  }

  RealType MaxValue() const override
  {
    return R;
  }

  void SetRadius(double radius)
  {
      if(radius <= 0.0)
          throw std::invalid_argument("Wigner-Semicircle distribution: radius should be positive");
      R = radius;
      RSq = R * R;
      logRSq = std::log(RSq);
  }

  inline double GetRadius() const
  {
    return R;
  }

  double f(const RealType& x) const override
  {
      double xSq = x * x;
      if(xSq >= RSq)
          return 0.0;
      double y = RSq - xSq;
      y = std::sqrt(y);
      y *= M_1_PI / RSq;
      return 2 * y;
  }

  double logf(const RealType& x) const override
  {
      double xSq = x * x;
      if(xSq >= RSq)
          return -INFINITY;
      return M_LN2 + 0.5 * std::log(RSq - xSq) - M_LNPI - logRSq;
  }

  double F(const RealType& x) const override
  {
      if(x <= -R)
          return 0.0;
      if(x >= R)
          return 1.0;
      double y = RSq - x * x;
      y = x * std::sqrt(y) / RSq;
      double z = std::asin(x / R);
      return 0.5 + (y + z) / M_PI;
  }

  RealType Variate() const override
  {
      RealType x = X.Variate();
      x += x - 1;
      return R * x;
  }

  void Reseed(unsigned long seed) const override
  {
      X.Reseed(seed);
  }

  long double Mean() const override
  {
      return 0.0;
  }

  long double Variance() const override
  {
      return 0.25 * RSq;
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
      return -1.0;
  }

  long double Entropy() const
  {
      return M_LNPI + 0.5 * logRSq - 0.5;
  }
};
} // namespace randlib
