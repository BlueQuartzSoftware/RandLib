#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The TriangularRand class <BR>
 * Triangular distribution
 *
 * Notation: X ~ Tri(a, b, c)
 */
template <typename RealType = double>
class RANDLIB_EXPORT TriangularRand : public randlib::ContinuousDistribution<RealType>
{
  double a = 0;                 ///< min value
  double b = 2;                 ///< max value
  double c = 1;                 ///< mode
  double constForGenerator = 1; ///< (c - a) / (b - a)
  double coefGenerator1 = 1;    ///< (b - a) * (c - a)
  double coefGenerator2 = 1;    ///< (b - a) * (b - c)

  void SetConstantsForGenerator()
  {
      constForGenerator = (c - a) / (b - a);
      coefGenerator1 = (b - a) * (c - a);
      coefGenerator2 = (b - a) * (b - c);
  }

public:
  TriangularRand(double lowerLimit = 0, double mode = 0.5, double upperLimit = 1)
  {
      SetParameters(lowerLimit, mode, upperLimit);
  }

  String Name() const override
  {
      return "Triangular(" + this->toStringWithPrecision(MinValue()) + ", " + this->toStringWithPrecision(Mode()) + ", " + this->toStringWithPrecision(MaxValue()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return a;
  }

  RealType MaxValue() const override
  {
    return b;
  }

  void SetParameters(double lowerLimit, double mode, double upperLimit)
  {
      if(lowerLimit >= mode)
          throw std::invalid_argument("Triangular distribution: lower limit should be greater than mode");
      if(mode >= upperLimit)
          throw std::invalid_argument("Triangular distribution: upper limit should be smaller than mode");
      a = lowerLimit;
      c = mode;
      b = upperLimit;
      SetConstantsForGenerator();
  }

  double f(const RealType& x) const override
  {
      if(x <= a)
          return 0;
      if(x < c)
          return 2.0 * (x - a) / coefGenerator1;
      if(x == c)
          return 2.0 / (b - a);
      if(x < b)
          return 2.0 * (b - x) / coefGenerator2;
      return 0;
  }

  double logf(const RealType& x) const override
  {
      return std::log(f(x));
  }

  double F(const RealType& x) const override
  {
      if(x <= a)
          return 0.0;
      if(x <= c)
          return (x - a) * (x - a) / coefGenerator1;
      if(x < b)
          return 1.0 - (b - x) * (b - x) / coefGenerator2;
      return 1.0;
  }

  double S(const RealType& x) const override
  {
      if(x <= a)
          return 1.0;
      if(x <= c)
          return 1.0 - (x - a) * (x - a) / coefGenerator1;
      if(x < b)
          return (b - x) * (b - x) / coefGenerator2;
      return 0.0;
  }

  RealType Variate() const override
  {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      if(U < constForGenerator)
          return a + std::sqrt(U * coefGenerator1);
      return b - std::sqrt((1 - U) * coefGenerator2);
  }

  long double Mean() const override
  {
      return (a + b + c) / 3.0;
  }

  long double Variance() const override
  {
      return (a * (a - b) + b * (b - c) + c * (c - a)) / 18.0;
  }

  RealType Median() const override
  {
      if(c + c > a + b)
          return a + std::sqrt(0.5 * coefGenerator1);
      return b - std::sqrt(0.5 * coefGenerator2);
  }

  RealType Mode() const override
  {
      return c;
  }

  long double Skewness() const override
  {
      double numerator = M_SQRT2;
      numerator *= (a + b - c - c);
      numerator *= (a + a - b - c);
      numerator *= (a - b - b + c);
      double denominator = a * (a - b);
      denominator += b * (b - c);
      denominator += c * (c - a);
      denominator *= std::sqrt(denominator);
      return 0.2 * numerator / denominator;
  }

  long double ExcessKurtosis() const override
  {
      return -0.6;
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
      double bmc = b - c, bma = b - a, cma = c - a;
      double at = a * t, bt = b * t, ct = c * t;
      std::complex<double> x(bmc * std::cos(at), bmc * std::sin(at));
      std::complex<double> y(bma * std::cos(ct), bma * std::sin(ct));
      std::complex<double> z(cma * std::cos(bt), cma * std::sin(bt));
      std::complex<double> numerator = x - y + z;
      /// in order to avoid numerical errors
      if(t < 1e-10 && std::fabs(numerator.real()) < 1e-10)
          return 1;
      double denominator = bma * cma * bmc * t * t;
      std::complex<double> frac = -numerator / denominator;
      return frac + frac;
  }
};
} // namespace randlib
